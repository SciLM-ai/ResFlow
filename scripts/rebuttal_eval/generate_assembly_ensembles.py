"""Model-side ensembles for the assembly-consistency benchmark (EVAL.md Addendum E.3).

Subcommands:
  native   -- (A) 250 native (64,64,32) volumes, Table 6 settings, empty mask,
              noise seeds 20260811000+k, conditioned on the E.2 selected
              instance's slim parameters (read from the engine manifest).
  assembly -- (B) 10 MultiDiffusion assemblies, 10x10 blocks, overlap 24,
              uniform conditioning (same instance, azimuth 95), torch seed
              20260809000+i; saves each 424x424x32 binary volume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RESFLOW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESFLOW))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resflow.models.unet3d import UNet3D  # noqa: E402
from resflow.methods.flow_matching import FlowMatching  # noqa: E402
from resflow.assembly import (  # noqa: E402
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX, generate_big_reservoir_multi)
from resflow.assembly.big_reservoir_multi import build_cond_vector  # noqa: E402
import generate_ensembles as ge  # noqa: E402  (Table 6 sampler + seeded noise)

ENV_CFG = {
    'pv': {'env': 'channel:PV_SHOESTRING', 'slug': 'channel_PV_SHOESTRING',
           'native_seed_base': 20260811000, 'asm_seed_base': 20260809000},
    'lobe': {'env': 'lobe', 'slug': 'lobe',
             'native_seed_base': 20260814000, 'asm_seed_base': 20260813000},
}
AZIMUTH = 95.0
VOLUME_SHAPE = (64, 64, 32)
CKPT_DIR = Path('/work/08405/ilgar/vista/genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')

N_NATIVE = 250
N_ASM = 10
GRID = (10, 10)
OVERLAP = 24


def load_model(device, ckpt=None):
    """Load a checkpoint, rebuilding whichever architecture it holds.

    Addendum-G runs may be UNet3D, UNet3D+attention or DiT3D; the
    architecture is inferred from the state_dict keys so callers do not
    have to track it.
    """
    path = Path(ckpt) if ckpt else CKPT_DIR / 'flow_matching.pt'
    sys.path.insert(0, str(RESFLOW / 'scripts' / 'tier2'))
    from model_factory import load_checkpoint  # noqa: E402
    model, arch = load_checkpoint(path, COND_DIM, VOLUME_SHAPE, device)
    print(f'loaded {path.name} (arch={arch})', flush=True)
    return model


def make_spec(slim, env):
    return BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[env], azimuth_deg=AZIMUTH,
                     raw_scalars=dict(slim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['native', 'assembly', 'outpaint'])
    ap.add_argument('--engine-manifest', required=True,
                    help='engine_manifest.json from resmill_median_ensemble.py')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--env', default='pv', choices=list(ENV_CFG))
    ap.add_argument('--overlap', type=int, default=None,
                    help='override the deployed overlap 24. Addendum-G '
                         'models randomise the training slab width over '
                         '[8,32], so any value in that range is in '
                         'distribution and needs no retraining.')
    ap.add_argument('--ckpt', default=None,
                    help='override the foundation flow_matching.pt (e.g. a '
                         'specialist inference_epochNNN.pt). cond_stats.npz '
                         'is always the foundation one, per Addendum E.2.')
    args = ap.parse_args()

    global OVERLAP
    if args.overlap is not None:
        OVERLAP = args.overlap
    cfg = ENV_CFG[args.env]
    NATIVE_SEED_BASE = cfg['native_seed_base']
    ASM_SEED_BASE = cfg['asm_seed_base']
    slim = json.loads(Path(args.engine_manifest).read_text())['slim_params']
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device, args.ckpt)

    stats = np.load(CKPT_DIR / 'cond_stats.npz', allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    spec = make_spec(slim, cfg['env'])
    cond_np = build_cond_vector(spec, cont_min, cont_max)

    if args.command == 'native':
        t0 = time.time()
        vols = np.empty((N_NATIVE, *VOLUME_SHAPE), dtype=np.int8)
        with torch.no_grad():
            for lo in range(0, N_NATIVE, args.batch_size):
                hi = min(lo + args.batch_size, N_NATIVE)
                x0 = ge.seeded_noise([NATIVE_SEED_BASE + k for k in range(lo, hi)],
                                     (1, *VOLUME_SHAPE)).to(device)
                cond = torch.from_numpy(np.tile(cond_np, (hi - lo, 1))).float().to(device)
                z = torch.zeros((hi - lo, 1, *VOLUME_SHAPE), device=device)
                model.set_inpaint_context(z, z)
                x = ge.euler_cfg_sample(model, x0, cond)
                vols[lo:hi] = (x[:, 0] > 0).to(torch.int8).cpu().numpy()
                print(f'{hi}/{N_NATIVE}', flush=True)
        env_dir = out / cfg['slug']
        env_dir.mkdir(exist_ok=True)
        ids = np.array([f'E-native|{k}' for k in range(N_NATIVE)], dtype=object)
        np.savez_compressed(env_dir / f'volumes_r0000-r{N_NATIVE - 1:04d}.npz',
                            ids=ids, volumes=vols)
        info = {'ensemble': 'E.3 A native', 'n': N_NATIVE,
                'seed_base': NATIVE_SEED_BASE, 'cond': cond_np.tolist(),
                'mean_ntg': float(vols.mean()),
                'wall_s': round(time.time() - t0, 1)}
    elif args.command == 'outpaint':
        # Sequential raster-order outpainting: same grid, overlap, steps,
        # CFG and per-assembly torch seed as the 'assembly' command, so the
        # ONLY difference is the fusion rule (conditioning vs velocity
        # averaging). Requires a slab-trained checkpoint.
        from resflow.assembly.outpaint import (  # noqa: E402
            generate_big_reservoir_outpaint)
        info = {'ensemble': 'E.3 B assemblies (outpaint fusion)',
                'grid': GRID, 'overlap': OVERLAP,
                'seed_base': ASM_SEED_BASE, 'cond': cond_np.tolist(),
                'n_steps': ge.N_STEPS, 'cfg': ge.CFG, 'assemblies': []}
        for i in range(N_ASM):
            torch.manual_seed(ASM_SEED_BASE + i)
            t0 = time.time()
            x_global, _ = generate_big_reservoir_outpaint(
                model, cond_np, grid_shape=GRID, block_shape=VOLUME_SHAPE,
                overlap=OVERLAP, n_steps=ge.N_STEPS, cfg_scale=ge.CFG,
                device=device)
            binary = (x_global.numpy() > 0).astype(np.int8)
            np.savez_compressed(out / f'assembly_{i:02d}.npz', binary=binary)
            info['assemblies'].append({'i': i, 'shape': list(binary.shape),
                                       'ntg': float(binary.mean()),
                                       'wall_s': round(time.time() - t0, 1)})
            print(f'outpaint assembly {i}: shape={binary.shape} '
                  f'ntg={binary.mean():.4f}', flush=True)
    else:
        method = FlowMatching(model)
        ny, nx = GRID
        grid = [[make_spec(slim, cfg['env']) for _ in range(nx)] for _ in range(ny)]
        info = {'ensemble': 'E.3 B assemblies', 'grid': GRID,
                'overlap': OVERLAP, 'seed_base': ASM_SEED_BASE,
                'cond': cond_np.tolist(), 'assemblies': []}
        for i in range(N_ASM):
            torch.manual_seed(ASM_SEED_BASE + i)
            t0 = time.time()
            x_global, _ = generate_big_reservoir_multi(
                method, grid, cont_min, cont_max,
                block_shape=VOLUME_SHAPE, overlap_xy=OVERLAP,
                n_steps=ge.N_STEPS, cfg_scale=ge.CFG,
                max_batch=24, device=device)
            binary = (x_global.numpy() > 0).astype(np.int8)
            np.savez_compressed(out / f'assembly_{i:02d}.npz', binary=binary)
            info['assemblies'].append({'i': i, 'shape': list(binary.shape),
                                       'ntg': float(binary.mean()),
                                       'wall_s': round(time.time() - t0, 1)})
            print(f'assembly {i}: shape={binary.shape} ntg={binary.mean():.4f}',
                  flush=True)
    info['ckpt'] = str(Path(args.ckpt) if args.ckpt
                       else CKPT_DIR / 'flow_matching.pt')
    (out / f'generation_manifest_{args.command}.json').write_text(
        json.dumps(info, indent=2))
    print('DONE', args.command)


if __name__ == '__main__':
    main()
