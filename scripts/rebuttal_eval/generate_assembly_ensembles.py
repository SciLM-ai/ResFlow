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

ENV = 'channel:PV_SHOESTRING'
AZIMUTH = 95.0
VOLUME_SHAPE = (64, 64, 32)
CKPT_DIR = Path('/work/08405/ilgar/vista/genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')

N_NATIVE = 250
NATIVE_SEED_BASE = 20260811000
N_ASM = 10
ASM_SEED_BASE = 20260809000
GRID = (10, 10)
OVERLAP = 24


def load_model(device):
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / 'flow_matching.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    return model


def make_spec(slim):
    raw = {k: slim[k] for k in ('ntg', 'width_cells', 'depth_cells',
                                'mCHsinu', 'mFFCHprop', 'probAvulInside')}
    return BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[ENV], azimuth_deg=AZIMUTH,
                     raw_scalars=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['native', 'assembly'])
    ap.add_argument('--engine-manifest', required=True,
                    help='engine_manifest.json from resmill_median_ensemble.py')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    slim = json.loads(Path(args.engine_manifest).read_text())['slim_params']
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)

    stats = np.load(CKPT_DIR / 'cond_stats.npz', allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    spec = make_spec(slim)
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
        env_dir = out / 'channel_PV_SHOESTRING'
        env_dir.mkdir(exist_ok=True)
        ids = np.array([f'E-native|{k}' for k in range(N_NATIVE)], dtype=object)
        np.savez_compressed(env_dir / f'volumes_r0000-r{N_NATIVE - 1:04d}.npz',
                            ids=ids, volumes=vols)
        info = {'ensemble': 'E.3 A native', 'n': N_NATIVE,
                'seed_base': NATIVE_SEED_BASE, 'cond': cond_np.tolist(),
                'mean_ntg': float(vols.mean()),
                'wall_s': round(time.time() - t0, 1)}
    else:
        method = FlowMatching(model)
        ny, nx = GRID
        grid = [[make_spec(slim) for _ in range(nx)] for _ in range(ny)]
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
    (out / f'generation_manifest_{args.command}.json').write_text(
        json.dumps(info, indent=2))
    print('DONE', args.command)


if __name__ == '__main__':
    main()
