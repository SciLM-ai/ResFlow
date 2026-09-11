"""Generate assembly ensembles under any block scheduler (Addendum G).

One checkpoint, several schedulers, everything else held fixed -- same
condition, seeds, grid, overlap, step count and CFG -- so the scheduler
is the only variable, the same discipline that isolated the fusion rule
in Addendum E.

  multi    velocity averaging (MultiDiffusion)
  raster   outpainting in raster order, anti-diagonal wavefront
  stage4   4-colour staging, constant depth in grid size
  coupled  trajectory conditioning, fully parallel (needs a 4-channel model)
  spot     shifted non-overlapping windows, one partition per ODE step
           (SpotDiffusion-style; no averaging, no staging)

Also emits the native (A) ensemble so A-vs-C is available per model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'rebuttal_eval'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resflow.methods.flow_matching import FlowMatching        # noqa: E402
from resflow.assembly import (                                # noqa: E402
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX, generate_big_reservoir_multi,
    generate_staged, generate_coupled, generate_spot)
from resflow.assembly.big_reservoir_multi import build_cond_vector  # noqa: E402
import generate_ensembles as ge                               # noqa: E402
from model_factory import load_checkpoint                     # noqa: E402

AZIMUTH = 95.0
VOLUME_SHAPE = (64, 64, 32)
CKPT_DIR = Path('/work/08405/ilgar/vista/genflows_runs_backup_ls6/'
                'reservoirs_inpainting/checkpoints')
NATIVE_SEED_BASE = 20260814000      # E.7 lobe values
ASM_SEED_BASE = 20260813000
N_NATIVE = 250
N_ASM = 10
GRID = (10, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sampler', required=True,
                    choices=['native', 'multi', 'raster', 'stage4', 'coupled',
                             'hybrid', 'spot'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--engine-manifest', required=True)
    ap.add_argument('--overlap', type=int, default=24)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--cfg', type=float, default=None,
                    help='classifier-free guidance scale; defaults to the '
                         'paper Table-6 value. Never swept for ASSEMBLY '
                         'before -- only for native generation.')
    ap.add_argument('--n-steps', type=int, default=None)
    ap.add_argument('--hybrid-order', default='raster',
                    choices=['raster', 'stage4'],
                    help='which conditioning schedule the hybrid switches '
                         'to after the MultiDiffusion warm-up')
    ap.add_argument('--md-frac', type=float, default=0.3,
                    help='hybrid only: fraction of the trajectory run as '
                         'MultiDiffusion before switching to 4-stage '
                         'conditioning')
    ap.add_argument('--solver', default='euler', choices=['euler', 'heun'],
                    help="ODE solver for the assembly samplers ('heun' = "
                         "2 evaluations per step; use half the steps for "
                         "equal cost). Native generation stays Euler.")
    ap.add_argument('--no-shared-noise', action='store_true',
                    help='draw fresh noise per block group instead of '
                         'cropping one global field. Shared noise gives '
                         'spatial coherence for free, since flow matching '
                         'maps correlated noise to correlated geology.')
    args = ap.parse_args()
    cfg_scale = ge.CFG if args.cfg is None else args.cfg
    n_steps = ge.N_STEPS if args.n_steps is None else args.n_steps

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, arch = load_checkpoint(args.ckpt, COND_DIM, VOLUME_SHAPE, device)
    in_ch = getattr(model, 'in_channels', 3)
    print(f'ckpt={Path(args.ckpt).name} arch={arch} in_channels={in_ch} '
          f'sampler={args.sampler} overlap={args.overlap}', flush=True)

    stats = np.load(CKPT_DIR / 'cond_stats.npz', allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    slim = json.loads(Path(args.engine_manifest).read_text())['slim_params']
    spec = BlockSpec(layer_idx=LAYER_TYPE_TO_IDX['lobe'], azimuth_deg=AZIMUTH,
                     raw_scalars=dict(slim))
    cond_np = build_cond_vector(spec, cont_min, cont_max)

    info = {'sampler': args.sampler, 'ckpt': str(args.ckpt), 'arch': arch,
            'in_channels': in_ch, 'overlap': args.overlap, 'grid': GRID,
            'n_steps': n_steps, 'cfg': cfg_scale, 'solver': args.solver,
            'shared_noise': not args.no_shared_noise,
            'cond': cond_np.tolist(), 'assemblies': []}

    if args.sampler == 'native':
        t0 = time.time()
        vols = np.empty((N_NATIVE, *VOLUME_SHAPE), dtype=np.int8)
        with torch.no_grad():
            for lo in range(0, N_NATIVE, args.batch_size):
                hi = min(lo + args.batch_size, N_NATIVE)
                x0 = ge.seeded_noise([NATIVE_SEED_BASE + k
                                      for k in range(lo, hi)],
                                     (1, *VOLUME_SHAPE)).to(device)
                cond = torch.from_numpy(
                    np.tile(cond_np, (hi - lo, 1))).float().to(device)
                z = torch.zeros((hi - lo, 1, *VOLUME_SHAPE), device=device)
                model.set_inpaint_context(z, z)
                x = ge.euler_cfg_sample(model, x0, cond,
                                        cfg_scale=cfg_scale,
                                        n_steps=n_steps)
                vols[lo:hi] = (x[:, 0] > 0).to(torch.int8).cpu().numpy()
                print(f'{hi}/{N_NATIVE}', flush=True)
        env_dir = out / 'lobe'
        env_dir.mkdir(exist_ok=True)
        ids = np.array([f'G-native|{k}' for k in range(N_NATIVE)], dtype=object)
        np.savez_compressed(env_dir / f'volumes_r0000-r{N_NATIVE - 1:04d}.npz',
                            ids=ids, volumes=vols)
        info['n'] = N_NATIVE
        info['mean_ntg'] = float(vols.mean())
        info['wall_s'] = round(time.time() - t0, 1)
    else:
        method = FlowMatching(model)
        ny, nx = GRID
        for i in range(N_ASM):
            torch.manual_seed(ASM_SEED_BASE + i)
            t0 = time.time()
            if args.sampler == 'multi':
                grid = [[spec for _ in range(nx)] for _ in range(ny)]
                xg, _ = generate_big_reservoir_multi(
                    method, grid, cont_min, cont_max,
                    block_shape=VOLUME_SHAPE, overlap_xy=args.overlap,
                    n_steps=n_steps, cfg_scale=cfg_scale, max_batch=24,
                    device=device, solver=args.solver)
            elif args.sampler in ('raster', 'stage4', 'hybrid'):
                order = (args.hybrid_order if args.sampler == 'hybrid'
                         else args.sampler)
                extra = {}
                if args.sampler == 'hybrid':
                    extra = dict(md_frac=args.md_frac, method=method,
                                 grid_specs=[[spec] * nx for _ in range(ny)],
                                 cont_min=cont_min, cont_max=cont_max)
                xg, _ = generate_staged(
                    model, cond_np, grid_shape=GRID,
                    block_shape=VOLUME_SHAPE, overlap=args.overlap,
                    n_steps=n_steps, cfg_scale=cfg_scale, device=device,
                    order=order,
                    shared_noise=not args.no_shared_noise, verbose=False,
                    solver=args.solver, **extra)
            elif args.sampler == 'spot':
                xg, _ = generate_spot(
                    model, cond_np, grid_shape=GRID,
                    block_shape=VOLUME_SHAPE, overlap=args.overlap,
                    n_steps=n_steps, cfg_scale=cfg_scale, device=device,
                    verbose=False, solver=args.solver)
            else:
                xg, _ = generate_coupled(
                    model, cond_np, grid_shape=GRID,
                    block_shape=VOLUME_SHAPE, overlap=args.overlap,
                    n_steps=n_steps, cfg_scale=cfg_scale, device=device,
                    verbose=False, solver=args.solver)
            binary = (xg.numpy() > 0).astype(np.int8)
            np.savez_compressed(out / f'assembly_{i:02d}.npz', binary=binary)
            info['assemblies'].append({'i': i, 'shape': list(binary.shape),
                                       'ntg': float(binary.mean()),
                                       'wall_s': round(time.time() - t0, 1)})
            print(f'{args.sampler} {i}: {binary.shape} '
                  f'ntg={binary.mean():.4f} '
                  f'{info["assemblies"][-1]["wall_s"]}s', flush=True)

    (out / f'generation_manifest_{args.sampler}.json').write_text(
        json.dumps(info, indent=2))
    print('DONE', args.sampler)


if __name__ == '__main__':
    main()
