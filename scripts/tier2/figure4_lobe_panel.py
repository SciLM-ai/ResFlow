"""Paper Figure 4's LOBE PANEL, regenerated with an Addendum-G model.

Everything about the generation is taken from the published pipeline
rather than re-specified here:

  * per-block conditioning comes from `big_reservoir/lobes/generate.py`'s
    own `build_grid_specs` -- lobe everywhere, NTG 0.7 fixed, width_cells
    0.8->0.2 / depth_cells 0.6->0.2 / asp 0.7->0.3 (normalised) along Y,
    azimuth 45->135 deg along X;
  * grid 10x10, blocks 64x64x32, overlap 24, MultiDiffusion ("hard"),
    NFE 50, CFG 3.0, torch seed 11 (gen_seeds.py's default);
  * cond_stats.npz is the foundation's, as in the paper;
  * rendering reuses figure4.py's `render_panel` unchanged, so the
    colormap, blue overlap hatching and block borders are the published
    code.

Only the checkpoint differs.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples' / 'reservoirs' / 'paper_figures'))
sys.path.insert(0, str(REPO / 'examples' / 'reservoirs' / 'big_reservoir' / 'lobes'))
sys.path.insert(0, str(REPO / 'scripts' / 'tier2'))

import figure4 as F4                                          # noqa: E402
import generate as LOBEGEN            # big_reservoir/lobes/generate.py
from resflow.methods.flow_matching import FlowMatching        # noqa: E402
from resflow.assembly import generate_big_reservoir_multi     # noqa: E402
from resflow.utils.plotting_reservoirs import (               # noqa: E402
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS)
from model_factory import load_checkpoint                     # noqa: E402

FOUND = Path('/work/08405/ilgar/vista/genflows_runs_backup_ls6/'
             'reservoirs_inpainting/checkpoints')
PAPER_NPZ = (REPO / 'examples/reservoirs/paper_figures/gen_seeds/seed_11/'
             'lobes/results/reservoir_hard_ov24.npz')


def generate(ckpt, seed, overlap, out_npz):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    stats = np.load(FOUND / 'cond_stats.npz', allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    grid = LOBEGEN.build_grid_specs(cont_min, cont_max)       # identical specs

    print(f'  width_cells Y: {grid[0][0].raw_scalars["width_cells"]:.1f} -> '
          f'{grid[-1][0].raw_scalars["width_cells"]:.1f}')
    print(f'  depth_cells Y: {grid[0][0].raw_scalars["depth_cells"]:.1f} -> '
          f'{grid[-1][0].raw_scalars["depth_cells"]:.1f}')
    print(f'  asp         Y: {grid[0][0].raw_scalars["asp"]:.2f} -> '
          f'{grid[-1][0].raw_scalars["asp"]:.2f}')
    print(f'  azimuth     X: {grid[0][0].azimuth_deg:.1f} -> '
          f'{grid[0][-1].azimuth_deg:.1f}')

    model, arch = load_checkpoint(ckpt, LOBEGEN.COND_DIM, (64, 64, 32), dev)
    method = FlowMatching(model)
    ny, nx = LOBEGEN.GRID_SHAPE
    torch.manual_seed(seed)
    t0 = time.time()
    volume, _ = generate_big_reservoir_multi(
        method, grid, cont_min, cont_max,
        block_shape=LOBEGEN.BLOCK_SHAPE, overlap_xy=overlap,
        n_steps=LOBEGEN.N_STEPS, cfg_scale=LOBEGEN.CFG_SCALE,
        max_batch=24, device=dev)
    volume = volume.numpy()
    binary = (volume > 0).astype(np.int8)
    el = time.time() - t0
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, volume=volume.astype(np.float32), binary=binary,
             mode='hard', overlap=overlap,
             block_shape=np.array(LOBEGEN.BLOCK_SHAPE), ny=ny, nx=nx,
             pure_x_indices=np.array(list(range(nx)), dtype=np.int32),
             pure_x_layer=np.array(['lobe'] * nx, dtype=object),
             trans_x_indices=np.array([], dtype=np.int32), trans_kind='hard',
             row_azimuths_deg=np.array(
                 [grid[i][0].azimuth_deg for i in range(ny)], dtype=np.float32),
             elapsed_s=el)
    print(f'  saved {out_npz}  shape={binary.shape} NTG={binary.mean():.3f} '
          f'{el:.1f}s  (arch={arch})')
    return binary


def as_run(npz):
    d = np.load(npz, allow_pickle=True)
    return {'binary': d['binary'], 'overlap': int(d['overlap']),
            'block_shape': tuple(int(s) for s in d['block_shape']),
            'ny': int(d['ny']), 'nx': int(d['nx'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', default='Addendum-G model')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--overlap', type=int, default=24)
    ap.add_argument('--z', type=int, default=F4.Z_SLICE)
    ap.add_argument('--out-dir', default='/work/08405/ilgar/vista/resbench_eval')
    ap.add_argument('--tag', default='new')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    out = Path(args.out_dir)
    npz = out / f'fig4lobe_{args.tag}_ov{args.overlap}_seed{args.seed}.npz'
    if not (args.skip_existing and npz.exists()):
        print(f'Generating with {Path(args.ckpt).name} (seed {args.seed}, '
              f'overlap {args.overlap}) ...')
        generate(args.ckpt, args.seed, args.overlap, npz)

    runs = [('Paper model', as_run(PAPER_NPZ)), (args.label, as_run(npz))]
    Tx, Ty, _ = runs[0][1]['binary'].shape

    panel_h = 7.5
    fig_w = 2 * panel_h * Tx / Ty + 1.2
    fig = plt.figure(figsize=(fig_w, panel_h + 1.1))
    gs = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.93,
                          bottom=0.13, wspace=0.10)
    for k, (name, run) in enumerate(runs):
        ax = fig.add_subplot(gs[0, k])
        F4.render_panel(ax, run, args.z,
                        title=f'{name}  ({Tx}×{Ty})',
                        show_ylabel=(k == 0), show_yticklabels=(k == 0))

    cax = fig.add_axes([0.055, 0.045, 0.22, 0.024])
    sm = plt.cm.ScalarMappable(cmap=CMAP_FACIES, norm=NORM_FACIES)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 1])
    cb.ax.set_xticklabels(FACIES_BINARY_LABELS, fontsize=11)
    fig.text(0.32, 0.050,
             'Identical conditioning: lobe everywhere, NTG 0.7, '
             'width/depth/asp gradient along Y, azimuth 45°→135° along X;  '
             f'10×10 blocks of 64×64×32, overlap {args.overlap}, '
             f'NFE 50, CFG 3.0, seed {args.seed}, z={args.z}',
             fontsize=9.5, va='bottom')

    for ext in ('png', 'pdf'):
        p = out / f'figure4_lobe_panel.{ext}'
        fig.savefig(p, dpi=170, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
