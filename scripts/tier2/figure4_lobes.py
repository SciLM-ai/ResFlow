"""Paper Figure 4, rendered for lobes with the Addendum-G models.

Reuses figure4.py's own helpers unchanged -- `render_panel`, `shade_overlaps`,
`draw_borders`, `_pcm`, the CMAP_FACIES/NORM_FACIES colormap, the axis
convention (binary[:, :, z].T with extent [0, Tx, 0, Ty], X horizontal,
Y vertical), the blue overlap hatching (/// for X, \\\\\\ for Y) and the
block-border lines -- so the visual language is identical to the published
figure. Only the source cubes differ.

Figure 4 in the paper devotes one big square panel to the lobe environment
and six narrow strips to the channel families. Here every panel is a lobe
assembly, so the same layout is reused with the panels holding, in order,
the published approach and the Addendum-G model at matched seeds.

Usage:
    python scripts/tier2/figure4_lobes.py --out-dir <dir>
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples' / 'reservoirs' / 'paper_figures'))

import figure4 as F4                                        # noqa: E402
from resflow.utils.plotting_reservoirs import (              # noqa: E402
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS)

EVAL = Path('/work/08405/ilgar/vista/resbench_eval')
BLOCK = (64, 64, 32)
OVERLAP = 24
GRID = 10


def load(assembly_dir, idx=0):
    """Wrap an Addendum-G assembly in the dict figure4.load_run returns."""
    f = sorted(Path(assembly_dir).glob('assembly_*.npz'))[idx]
    b = np.load(f)['binary']
    return {'binary': b, 'overlap': OVERLAP, 'block_shape': BLOCK,
            'ny': GRID, 'nx': GRID, 'src': str(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(EVAL))
    ap.add_argument('--z', type=int, default=F4.Z_SLICE)
    ap.add_argument('--seeds', type=int, default=5,
                    help='number of assembly seeds shown as strips')
    ap.add_argument('--vector', action='store_true')
    args = ap.parse_args()
    F4.VECTOR = args.vector

    big = load(EVAL / 'ep_065', 0)              # best Addendum-G model
    prev = load(EVAL / 'assembly_lobe_specialist', 0)   # published approach
    strips = [('Published\napproach', prev)]
    for k in range(1, args.seeds):
        strips.append((f'New model\nseed {k}', load(EVAL / 'ep_065', k)))

    Tx, Ty, _ = big['binary'].shape
    Txs, Tys, _ = strips[0][1]['binary'].shape
    n = len(strips)

    # Geometry copied from figure4.main(): pin the common panel height and
    # derive widths from aspect='equal' so every panel shares a vertical
    # extent.
    panel_h_in = 7.5
    big_w_in = panel_h_in * Tx / Ty
    st_w_in = panel_h_in * Txs / Tys / 3.0      # narrow strips, as in fig 4
    wsp = 0.10
    stack_w_in = n * st_w_in + (n - 1) * wsp
    fig_w = big_w_in + 1.0 + stack_w_in + 0.5
    fig_h = panel_h_in + 1.1

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=2,
                          width_ratios=[big_w_in, stack_w_in],
                          left=0.06, right=0.985, top=0.93, bottom=0.12,
                          wspace=0.08)

    ax_big = fig.add_subplot(gs[0, 0])
    F4.render_panel(ax_big, big, args.z,
                    title=f'Lobe  ({Tx}×{Ty})')

    sub = gs[0, 1].subgridspec(nrows=1, ncols=n, wspace=wsp)
    for i, (name, run) in enumerate(strips):
        ax = fig.add_subplot(sub[0, i])
        # crop each strip to a 64-wide window so the aspect matches the
        # channel strips of the published figure
        r = dict(run)
        r['binary'] = run['binary'][:64, :, :]
        r['nx'] = 1
        F4.render_panel(ax, r, args.z, title=name,
                        show_xlabel=True, show_ylabel=(i == 0),
                        show_yticklabels=(i == 0))

    # colourbar + overlap legend, same as figure4
    cax = fig.add_axes([0.06, 0.045, 0.28, 0.022])
    sm = plt.cm.ScalarMappable(cmap=CMAP_FACIES, norm=NORM_FACIES)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 1])
    cb.ax.set_xticklabels(FACIES_BINARY_LABELS, fontsize=11)
    fig.text(0.40, 0.050,
             'blue hatching = block overlap (/// X, \\\\\\ Y);  '
             f'blocks {BLOCK[0]}×{BLOCK[1]}, overlap {OVERLAP}, '
             f'{GRID}×{GRID} grid, z={args.z}',
             fontsize=10, va='bottom')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out / f'figure4_lobes.{ext}'
        fig.savefig(p, dpi=170, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
