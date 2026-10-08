"""Generate the 64-cube parts of a ResBench v1 submission for ONE environment.

  --part uncond : unconditional/<slug>/samples/shard00.npz      512 rows, ids = manifest ids,
                                                                starting noise = the row's noise_seed
                  unconditional/<slug>/repeats/cond0..4.npz     K_REPEATS runs of manifest row <env>|cond<i>
  --part well   : well_conditioned/<slug>/samples/shard00.npz   512 rows; borehole = the reference volume's
                                                                column at (well_x, well_y), held exactly
                  well_conditioned/<slug>/repeats/well1..5.npz  K_REPEATS runs of row <env>|well<i>; the well
                                                                is the reference file's well_mask + pattern

Conditions come from REF/manifest.csv only (ntg = realised, azimuth, width/depth/family columns).
Sampler: Heun, n_steps (100), CFG 3.0, bf16 autocast, the model's trained attention (global on a
64-cube). Velocity/CFG/Heun math is the one in resflow.assembly.wholefield.generate_wholefield
and rope_native_heun.py; well conditioning is the training-time set_inpaint_context(mask, x*mask)
followed by apply_inpaint_output (hard replace of the known cells).
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = str(Path(__file__).resolve().parents[2])   # repository root
for p in (REPO, f'{REPO}/scripts/rebuttal_eval', f'{REPO}/scripts/tier2'):
    sys.path.insert(0, p)
from resflow.assembly.big_reservoir_multi import (                     # noqa: E402
    BlockSpec, build_cond_vector, LAYER_TYPE_TO_IDX, CONT_COLS)
from arch_loader import load_any                                       # noqa: E402

CKPT_DIR = Path(os.environ.get('RESFLOW_COND_STATS_DIR', f'{REPO}/assets'))   # cond_stats.npz: the fixed condition normalisation
ENVS = ('lobe', 'channel:PV_SHOESTRING', 'channel:CB_LABYRINTH',
        'channel:CB_JIGSAW', 'channel:SH_DISTAL', 'channel:SH_PROXIMAL',
        'channel:MEANDER_OXBOW', 'delta')
SHAPE = (64, 64, 32)
K_REPEATS = 128
REPEAT_SEED = 20260923      # fresh noise for repeats: default_rng([REPEAT_SEED, env_index, cond_index, k])


def cond_from_row(row, env, cmin, cmax):
    raw = {c: float(row[c]) for c in CONT_COLS if row.get(c, '') not in ('', None)}
    spec = BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[env],
                     azimuth_deg=float(row['azimuth']), raw_scalars=raw)
    return build_cond_vector(spec, cmin, cmax)


def noise(seeds):
    return torch.from_numpy(np.stack(
        [np.random.default_rng(s).standard_normal(SHAPE, dtype=np.float32) for s in seeds]))[:, None]


@torch.no_grad()
def heun_sample(model, x, cond, mask, known, n_steps, cfg, amp=True):
    """x, mask, known: (B,1,X,Y,Z) on device; cond: (B,18). Returns x with known cells replaced."""
    model.set_inpaint_context(mask, known)
    B = x.shape[0]

    def vel(xs, t):
        tt = torch.full((B,), t, device=xs.device) * 1000
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            v_c = model(xs, tt, cond).float()
            v_u = model(xs, tt).float()
        return v_u + cfg * (v_c - v_u)

    dt = 1.0 / n_steps
    for s in range(n_steps):
        t = s * dt
        v = vel(x, t)
        v = 0.5 * (v + vel(x + v * dt, t + dt))
        x = x + v * dt
    model.clear_inpaint_context()
    return x * (1 - mask) + known * mask


def run(model, conds, x0, masks, knowns, batch, n_steps, cfg, dev, tag):
    out = []
    t0 = time.time()
    for b in range(0, len(conds), batch):
        sl = slice(b, b + batch)
        c = torch.as_tensor(np.asarray(conds[sl], dtype=np.float32), device=dev)
        x = heun_sample(model, x0[sl].to(dev), c, masks[sl].to(dev), knowns[sl].to(dev), n_steps, cfg)
        out.append((x[:, 0] > 0).to(torch.int8).cpu().numpy())
        done = min(b + batch, len(conds))
        print(f'  {tag}: {done}/{len(conds)}  {time.time()-t0:.0f}s  '
              f'{(time.time()-t0)/done:.2f}s/cube', flush=True)
    return np.concatenate(out)


def save(path, ids, vols):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, ids=np.array(ids, dtype=object), volumes=vols)
    os.replace(tmp, path)
    print(f'WROTE {path}  {vols.shape}  mean sand {vols.mean():.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='submission root')
    ap.add_argument('--ref', default=os.environ.get('RESBENCH_REF', 'resbench_v1_ref'))
    ap.add_argument('--env', required=True)
    ap.add_argument('--part', choices=['uncond', 'well'], required=True)
    ap.add_argument('--n-steps', type=int, default=100)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=None, help='smoke: first N rows / N repeats')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    env, slug, ei = a.env, a.env.replace(':', '_'), ENVS.index(a.env)
    ref, out, dev = Path(a.ref), Path(a.out), 'cuda'
    K = min(K_REPEATS, a.limit) if a.limit else K_REPEATS
    task = 'unconditional' if a.part == 'uncond' else 'well_conditioned'
    conds_names = [f'cond{i}' for i in range(5)] if a.part == 'uncond' else [f'well{i}' for i in range(1, 6)]
    p_samples = out / task / slug / 'samples' / 'shard00.npz'
    p_reps = {c: out / task / slug / 'repeats' / f'{c}.npz' for c in conds_names}
    if not a.force and p_samples.exists() and all(p.exists() for p in p_reps.values()):
        print(f'CACHED {task} {slug}'); return

    rows = [r for r in csv.DictReader(open(ref / 'manifest.csv', newline='')) if r['environment'] == env]
    urows = [r for r in rows if r['task'] == 'unconditional']
    if a.limit:
        urows = urows[:a.limit]
    byid = {}
    if a.part == 'well':
        z = np.load(ref / 'volumes' / slug / 'volumes.npz', allow_pickle=True)
        byid = dict(zip([str(i) for i in z['ids']], z['volumes']))

    model, arch = load_any(a.ckpt, dev)
    stats = np.load(CKPT_DIR / 'cond_stats.npz', allow_pickle=True)
    cmin, cmax = stats['cont_min'], stats['cont_max']
    print(f'{arch} {env} part={a.part} heun{a.n_steps} cfg{a.cfg} batch{a.batch}', flush=True)

    # ---- samples: one volume per reference row -------------------------------------------------
    if a.force or not p_samples.exists():
        n = len(urows)
        conds = np.stack([cond_from_row(r, env, cmin, cmax) for r in urows])
        x0 = noise([int(r['noise_seed']) for r in urows])
        masks = torch.zeros((n, 1) + SHAPE); knowns = torch.zeros((n, 1) + SHAPE)
        if a.part == 'well':
            for b, r in enumerate(urows):
                x, y = int(r['well_x']), int(r['well_y'])
                col = byid[r['id']][x, y, :].astype(np.float32)
                masks[b, 0, x, y, :] = 1.0
                knowns[b, 0, x, y, :] = torch.from_numpy(col * 2.0 - 1.0)
        vols = run(model, conds, x0, masks, knowns, a.batch, a.n_steps, a.cfg, dev, f'{slug} {task} samples')
        if a.part == 'well':
            for b, r in enumerate(urows):
                x, y = int(r['well_x']), int(r['well_y'])
                assert (vols[b, x, y, :] == byid[r['id']][x, y, :]).all(), f'well mismatch row {b}'
        save(p_samples, [r['id'] for r in urows], vols)

    # ---- repeats: K runs of one fixed input ----------------------------------------------------
    for ci, cname in enumerate(conds_names):
        if not a.force and p_reps[cname].exists():
            continue
        (row,) = [r for r in rows if r['id'] == f'{env}|{cname}']
        conds = np.repeat(cond_from_row(row, env, cmin, cmax)[None], K, axis=0)
        x0 = noise([[REPEAT_SEED, ei, ci, k] for k in range(K)])
        masks = torch.zeros((K, 1) + SHAPE); knowns = torch.zeros((K, 1) + SHAPE)
        if a.part == 'well':
            w = np.load(ref / 'repeats' / slug / f'{cname}.npz', allow_pickle=True)
            m = torch.from_numpy(w['well_mask'].astype(np.float32))
            pat = torch.from_numpy(w['pattern'].astype(np.float32) * 2.0 - 1.0)
            masks[:, 0] = m
            knowns[:, 0] = m * pat.view(1, 1, -1)       # the pattern along z at the well column
            assert int(m.sum()) == SHAPE[2] and tuple(np.argwhere(w['well_mask'].any(2))[0]) == tuple(w['well_xy'])
        vols = run(model, conds, x0, masks, knowns, a.batch, a.n_steps, a.cfg, dev, f'{slug} {task} {cname}')
        if a.part == 'well':
            wx, wy = (int(v) for v in w['well_xy'])
            assert (vols[:, wx, wy, :] == w['pattern']).all(), f'{cname}: well cells differ from pattern'
        save(p_reps[cname], [f'{cname}|{k}' for k in range(K)], vols)


if __name__ == '__main__':
    main()
