"""Generate a ResBench v1 field_scale submission part.

v1 compares model fields against ResMill fields of the SAME extent, because a
64-cube can never contain a body wider than 64 cells -- the old protocol cut
64-cubes out of a large field and compared them to natively built ones, which
charged every model for a gap ResMill opens against itself (STATUS.md).

Each field is conditioned on the same test-split row the reference used, so the
two ensembles span the same parameter range. Fields are sharded so the 32 can
be spread over nodes.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import pyarrow.parquet as pq

REPO = str(Path(__file__).resolve().parents[2])   # repository root
for p in (REPO, f'{REPO}/scripts/rebuttal_eval', f'{REPO}/scripts/tier2'):
    sys.path.insert(0, p)
from resflow.assembly.wholefield import generate_wholefield            # noqa: E402
from resflow.assembly.big_reservoir_multi import (                     # noqa: E402
    BlockSpec, build_cond_vector, LAYER_TYPE_TO_IDX, CONT_COLS)
from arch_loader import load_any                                       # noqa: E402
from model_factory import set_inference_attention                      # noqa: E402

DATA_DIR = Path(os.environ.get('RESERVOIR_DATA_DIR',
                               os.environ.get('SILICICLASTIC_ROOT', 'SiliciclasticReservoirs')))
CKPT_DIR = Path(os.environ.get('RESFLOW_COND_STATS_DIR', f'{REPO}/assets'))   # cond_stats.npz: the fixed condition normalisation
EXTENT = {'lobe': (512, 512, 32), 'delta': (512, 512, 32)}
SEED_BASE = 2026091800
ENVS = ('lobe', 'channel:PV_SHOESTRING', 'channel:CB_LABYRINTH',
        'channel:CB_JIGSAW', 'channel:SH_DISTAL', 'channel:SH_PROXIMAL',
        'channel:MEANDER_OXBOW', 'delta')


def pick_rows(env, n):
    """Exactly ResBench tools/gen_field_reference.py:pick_rows -- the same n
    test-split conditions, spread evenly across realized sand fraction."""
    test = pd.read_parquet(DATA_DIR / 'splits' / 'test.parquet')
    test = test[test['layer_type'] == env].reset_index(drop=True)
    rows = [r.to_dict() for _, r in test.iterrows()]
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    out = []
    for i in idx:
        r = rows[int(i)]
        t = pq.read_table(DATA_DIR / r['shard_dir'] / 'params.parquet')
        out.append({c: t[c][int(r['sample_idx'])].as_py() for c in t.column_names})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='submission root')
    ap.add_argument('--env', default='lobe')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--shard', default='0/1', help='k/N: generate this slice')
    ap.add_argument('--radius', type=str, default=None,
                    help='sliding-attention radius at inference; omit for trained mode')
    ap.add_argument('--ref', default=os.environ.get('RESBENCH_REF', 'resbench_v1_ref'),
                    help='reference root: manifest.csv gives ntg (field realised) and azimuth, '
                         'fields/<slug>/manifest.json gives the extent')
    ap.add_argument('--n-steps', type=int, default=100)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--compile', action='store_true', help='torch.compile the model forward (same math, faster)')
    a = ap.parse_args()

    k, nsh = (int(x) for x in a.shard.split('/'))
    dev = 'cuda'
    model, arch = load_any(a.ckpt, dev)
    if a.radius is not None:
        rad = tuple(int(v) for v in a.radius.split(',')) if ',' in a.radius else int(a.radius)
        set_inference_attention(model, rad)
        print(f'inference attention: sliding radius {a.radius}', flush=True)
    if a.compile:
        model.forward = torch.compile(model.forward)

    stats = np.load(CKPT_DIR / 'cond_stats.npz', allow_pickle=True)
    cmin, cmax = stats['cont_min'], stats['cont_max']
    rows = pick_rows(a.env, a.n)
    slug = a.env.replace(':', '_')
    ext = tuple(json.load(open(Path(a.ref) / 'fields' / slug / 'manifest.json'))['extent'])

    # The condition is published by the reference manifest, not inferred from the
    # source 64-cube: at field extent the engine's realized sand fraction drifts
    # from the cube's by ~0.02, which is 2x net_to_gross's tolerance. A submitter
    # reads the target it will be scored against. Width/depth/family columns are
    # not in the manifest's field rows; they come from the same test-split rows
    # the reference generator used (pick_rows, identical code).
    import csv
    man = {}
    for r in csv.DictReader(open(Path(a.ref) / 'manifest.csv', newline='')):
        if r.get('environment') == a.env and r.get('task') == 'field_scale':
            man[int(str(r['id']).split('|')[-1]) - SEED_BASE - 1000 * ENVS.index(a.env)] = r
    assert len(man) == len(rows) == a.n, (len(man), len(rows), a.n)
    for i, row in enumerate(rows):
        row['ntg'] = float(man[i]['ntg'])
        row['azimuth'] = float(man[i]['azimuth'])
        row['_id'] = man[i]['id']
    print(f'conditioning on manifest targets for {len(man)} fields', flush=True)

    mine = [(i, r) for i, r in enumerate(rows) if i % nsh == k]
    print(f'{arch} {a.env} extent {ext}: shard {k}/{nsh} -> {len(mine)} fields', flush=True)

    ids, vols = [], []
    for i, row in mine:
        # SPEC.md: channel fields are elongated along flow at azimuth 0 (the
        # manifest row carries exactly that value); lobe/delta keep the row's.
        az = float(row['azimuth'])
        spec = BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[a.env],
                         azimuth_deg=az,
                         raw_scalars={c: row[c] for c in CONT_COLS if c in row})
        cond = build_cond_vector(spec, cmin, cmax)
        g = torch.Generator(device=dev).manual_seed(SEED_BASE + 1000 * ENVS.index(a.env) + i)
        t0 = time.time()
        f, _ = generate_wholefield(model, cond, block_shape=(64, 64, 32),
                                   overlap=12, n_steps=a.n_steps, cfg_scale=a.cfg,
                                   device=dev, solver='heun', generator=g,
                                   verbose=False, field_shape=ext)
        b = (np.asarray(f) > 0).astype(np.int8)
        ids.append(row['_id'])
        vols.append(b)
        print(f'  field {i} ntg {b.mean():.4f} {time.time()-t0:.0f}s', flush=True)

    d = Path(a.out) / 'field_scale' / slug / 'fields'
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'shard{k:02d}.npz.tmp.npz'
    np.savez_compressed(tmp, ids=np.array(ids, dtype=object), volumes=np.stack(vols))
    os.replace(tmp, d / f'shard{k:02d}.npz')
    print(f'WROTE {d}/shard{k:02d}.npz  {len(vols)} fields', flush=True)


if __name__ == '__main__':
    main()
