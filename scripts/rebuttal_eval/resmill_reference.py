"""ResMill reference ensembles for entropy calibration (EVAL.md Addendum A.3).

Wraps resmill.dataset.generate.generate_sample — engine source untouched.

Subcommands:
  validate  — regenerate one dataset instance per environment from its
              (params, seed) and require bit-exact facies equality with the
              stored dataset copy. Doubles as the per-environment timing
              probe. MUST pass before `generate` is trusted.
  generate  — for each entropy condition (manifest rows 0..3 per env),
              N unconditional-under-parameters realizations with logged
              seeds, multiprocessing over CPU cores.

Engine kwargs are rebuilt from the full params.parquet row: strip meta/
derived columns (REPRODUCIBILITY.md §3), restore the requested NTG target
(meta['ntg'] holds the realized value; the engine input is preserved in
'requested_ntg') under the engine's own key name ('NTGtarget' if the row
has that column, else 'ntg'), drop per-row nulls.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RESMILL_REPO = Path(os.environ.get(
    'RESMILL_REPO',
    Path(__file__).resolve().parents[3] / 'ResMill_ls6'))
sys.path.insert(0, str(RESMILL_REPO))

CONFIG_DIR = RESMILL_REPO / 'examples' / 'dataset_generation'
CONFIG_FOR_ENV = {
    'lobe': 'config_full_lobes.json',
    'channel:PV_SHOESTRING': 'config_full_pv_shoestring.json',
    'channel:CB_LABYRINTH': 'config_full_cb_labyrinth.json',
    'channel:CB_JIGSAW': 'config_full_cb_jigsaw.json',
    'channel:SH_DISTAL': 'config_full_sh_distal.json',
    'channel:SH_PROXIMAL': 'config_full_sh_proximal.json',
    'channel:MEANDER_OXBOW': 'config_full_meander_oxbow.json',
    'delta': 'config_full_delta.json',
}

# Meta/derived columns that are NOT create_geology kwargs (REPRODUCIBILITY §3).
ENGINE_IGNORE = {
    'layer_type', 'seed', 'caption', 'ntg', 'requested_ntg',
    'poro_ave', 'perm_ave',
    'r_ave_m', 'r_ave_cells', 'r_major_m', 'r_major_cells',
    'dh_ave_m', 'dh_ave_cells',
    'mCHdepth_m', 'mCHdepth_cells', 'mCHwidth_m', 'mCHwidth_cells',
    'width_cells', 'depth_cells',
}

DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs'))
MASTER_SEED = 20260730


def load_grid(layer_type):
    cfg = json.loads((CONFIG_DIR / CONFIG_FOR_ENV[layer_type]).read_text())
    return cfg['grid']


def engine_job(row: dict, seed: int):
    """(job, grid_cfg) for generate_sample from a full params.parquet row."""
    kwargs = {k: v for k, v in row.items()
              if k not in ENGINE_IGNORE and v is not None
              and not (isinstance(v, float) and np.isnan(v))}
    # Restore the NTG target the engine actually received (see module doc).
    if 'NTGtarget' not in row or row.get('NTGtarget') is None:
        kwargs['ntg'] = row['requested_ntg']
    family = row['layer_type'].split(':')[0]
    if family == 'lobe':
        # LobeLayer requires poro_ave/perm_ave. The stored columns are the
        # REALIZED post-crop means (they overwrite the inputs in meta), but
        # lobe properties are filled after the facies geometry, so facies
        # stays bit-exact; only poro/perm are approximate — irrelevant here.
        kwargs['poro_ave'] = row['poro_ave']
        kwargs['perm_ave'] = row['perm_ave']
    job = {'layer_type': family, 'params': kwargs, 'seed': int(seed)}
    return job, load_grid(row['layer_type'])


def _gen_one(task):
    row, seed = task
    from resmill.dataset.generate import generate_sample
    t0 = time.time()
    facies, _, _, _, _ = generate_sample(*engine_job(row, seed))
    return facies.astype(np.int8), seed, time.time() - t0


def load_manifest_rows(manifest, data_dir, row_lo, row_hi):
    """Conditions with row_lo <= row_index < row_hi, in manifest order
    (canonical env order, row_index ascending) — the GLOBAL condition
    enumeration used for seeding, independent of any --envs filter."""
    mf = pd.read_csv(manifest, keep_default_na=False)
    mf = mf[(mf['row_index'] >= row_lo) & (mf['row_index'] < row_hi)]
    out = []
    for _, r in mf.iterrows():
        t = pq.read_table(Path(data_dir) / r['shard_dir'] / 'params.parquet')
        row = {c: t[c][int(r['sample_idx'])].as_py() for c in t.column_names}
        out.append((r.to_dict(), row))
    return out


def cmd_validate(args):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    rows = load_manifest_rows(args.manifest, args.data_dir, 0, 1)  # row 0 per env
    ok = True
    for mrow, prow in rows:
        stored = np.load(Path(args.data_dir) / mrow['shard_dir'] / 'facies.npy',
                         mmap_mode='r')[int(mrow['sample_idx'])]
        facies, seed, dt = _gen_one((prow, int(prow['seed'])))
        match = np.array_equal(facies, np.asarray(stored))
        ok &= match
        print(f"{mrow['environment']:24s} seed={seed} {dt:6.1f}s "
              f"{'BIT-EXACT' if match else 'MISMATCH (%.4f frac diff)' % (facies != stored).mean()}",
              flush=True)
    print('VALIDATION', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


def cmd_generate(args):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    lo, hi = (int(x) for x in args.rows.split(':'))
    rows = load_manifest_rows(args.manifest, args.data_dir, lo, hi)
    envs = set(args.envs.split(',')) if args.envs else None
    out = Path(args.out_dir)
    t_start = time.time()
    for ci, (mrow, prow) in enumerate(rows):   # ci: GLOBAL condition index
        if envs is not None and mrow['environment'] not in envs:
            continue
        slug = mrow['environment'].replace(':', '_')
        d = out / slug
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"cond_r{int(mrow['row_index']):04d}_{mrow['well_config']}.npz"
        if path.exists():
            print(f'skip existing {path.name}', flush=True)
            continue
        seeds = np.random.default_rng([args.seed_base, ci]).integers(
            1, 2**31 - 1, size=args.n_realizations)
        with Pool(args.workers) as pool:
            res = pool.map(_gen_one, [(prow, int(s)) for s in seeds],
                           chunksize=1)
        vols = np.stack([r[0] for r in res])
        times = [r[2] for r in res]
        np.savez_compressed(
            path, volumes=vols, seeds=seeds,
            ref_id=mrow['row_id'], well_config=mrow['well_config'],
            params_pointer=f"{mrow['shard_dir']}/params.parquet#row={mrow['sample_idx']}")
        print(f"[{ci + 1}/{len(rows)}] {slug} r{int(mrow['row_index'])}: "
              f"{len(vols)} vols, {np.mean(times):.1f}s/vol/core, "
              f"total {time.time() - t_start:.0f}s", flush=True)
    (out / f'resmill_ref_manifest_{lo}-{hi}.json').write_text(json.dumps({
        'seed_base': args.seed_base, 'rows': args.rows, 'envs': args.envs,
        'n_realizations': args.n_realizations,
        'workers': args.workers, 'n_conditions': len(rows),
        'wall_clock_s': round(time.time() - t_start, 1)}, indent=2))


def cmd_reject(args):
    """Targeted rejection sampling (Addendum A.4) for the conditions listed in
    --conditions-json: [{environment, row_index, well_config, ...}, ...].
    Draws in chunks until --target-accepted or --max-draws per condition.
    Acceptance = realization matches the reference volume at all well voxels.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')

    # Figure-3 XZ well masks (same frozen fractions as generate_ensembles.py)
    # rebuilt numpy-only: torch is not available in the resmill env.
    def well_mask(config, shape=(64, 64, 32), y=32):
        fr = {'1well': [0.5], '2wells': [0.33, 0.66],
              '3wells': [0.25, 0.5, 0.75]}[config]
        m = np.zeros(shape, dtype=bool)
        for f in fr:
            m[int(round(shape[0] * f)), y, :] = True
        return m

    wanted = json.loads(Path(args.conditions_json).read_text())
    rows = load_manifest_rows(args.manifest, args.data_dir, 0, 4)
    by_key = {(m['environment'], int(m['row_index'])): (m, p) for m, p in rows}
    out = Path(args.out_dir)
    for w in wanted:
        mrow, prow = by_key[(w['environment'], int(w['row_index']))]
        slug = mrow['environment'].replace(':', '_')
        stem = f"cond_r{int(mrow['row_index']):04d}_{mrow['well_config']}"
        path = out / slug / f'{stem}.npz'
        if path.exists():
            print(f'skip existing {path.name}', flush=True)
            continue
        ref = np.load(Path(args.ref_volumes) / slug
                      / 'volumes_r0000-r0511.npz', allow_pickle=True)
        ref_vol = dict(zip([str(i) for i in ref['ids']],
                           ref['volumes']))[mrow['row_id']]
        wells = well_mask(mrow['well_config'])
        w_ref = ref_vol[wells]

        accepted, seeds_used, drawn, chunk_i = [], [], 0, 0
        t0 = time.time()
        while len(accepted) < args.target_accepted and drawn < args.max_draws:
            n = min(args.workers * 8, args.max_draws - drawn)
            seeds = np.random.default_rng(
                [20260732, int(mrow['row_index']),
                 sum(ord(c) for c in slug) % 10007,   # salted hash() is not reproducible
                 chunk_i]).integers(1, 2**31 - 1, size=n)
            with Pool(args.workers) as pool:
                res = pool.map(_gen_one, [(prow, int(s)) for s in seeds],
                               chunksize=1)
            for facies, seed, _ in res:
                if np.array_equal(facies[wells], w_ref):
                    accepted.append(facies)
                    seeds_used.append(seed)
            drawn += n
            chunk_i += 1
            print(f'{slug} {stem}: {len(accepted)}/{args.target_accepted} '
                  f'accepted after {drawn} draws ({time.time() - t0:.0f}s)',
                  flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, volumes=np.stack(accepted) if accepted
                            else np.empty((0, 64, 64, 32), np.int8),
                            seeds=np.array(seeds_used), n_draws=drawn)
        print(f'{stem}: saved {len(accepted)} accepted / {drawn} draws',
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['validate', 'generate', 'reject'])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir')
    ap.add_argument('--n-realizations', type=int, default=512)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--rows', default='0:4',
                    help='generate: manifest row_index range LO:HI')
    ap.add_argument('--envs', default=None,
                    help='generate: comma-separated environment filter '
                         '(global condition indices/seeds are unaffected)')
    ap.add_argument('--seed-base', type=int, default=MASTER_SEED,
                    help='generate: rng seed base (Addendum B uses 20260801)')
    ap.add_argument('--conditions-json', help='reject: conditions to target')
    ap.add_argument('--ref-volumes', help='reject: reference volume root')
    ap.add_argument('--target-accepted', type=int, default=200)
    ap.add_argument('--max-draws', type=int, default=20000)
    args = ap.parse_args()
    if args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'generate':
        assert args.out_dir, '--out-dir required'
        cmd_generate(args)
    else:
        assert args.out_dir and args.conditions_json and args.ref_volumes
        cmd_reject(args)


if __name__ == '__main__':
    main()
