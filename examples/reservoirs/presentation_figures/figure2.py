"""Presentation Figure 2: well-conditioned FM samples for the first 4 reservoir
architectures (Lobe, Delta, Meander, PV-shoestring), top 3 facies rows only.

Subset of NeurIPS Figure 3:
  - 3 rows (XY @ z=18, XZ @ y=32, YZ @ x=32) of binary facies.
  - 4 cols (first half of figure 3's 8 columns).
  - Per-row well configuration (3 wells at fractions 0.25/0.5/0.75 by default,
    matching paper_figures/figure3.py's canonical 3-well config):
      Row 0 (XY @ z=z_slice): horizontal wells along Y at fixed X.
      Row 1 (XZ @ y=y_slice): vertical wells at fixed X, full z extent.
      Row 2 (YZ @ x=x_slice): vertical wells at fixed Y, full z extent.
  - No entropy block.

Cubes are reused from paper_figures/figs/figure3_3wells_cubes.npz when
available; otherwise they are regenerated from the FM-inpaint checkpoint.

Run:
    python examples/reservoirs/presentation_figures/figure2.py
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PRES_DIR = Path(__file__).resolve().parent
PAPER_DIR = PRES_DIR.parent / 'paper_figures'
sys.path.insert(0, str(PAPER_DIR))
import figure2 as paper_fig2  # noqa: E402
import figure3 as paper_fig3  # noqa: E402

from resflow.utils.data_reservoirs import VOLUME_SHAPE  # noqa: E402
from resflow.utils.plotting_reservoirs import (  # noqa: E402
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
    slice_xy, slice_xz, slice_yz, imshow_slice,
)


# Use the 4-layer subset.
COLUMN_ORDER_FULL = paper_fig2.COLUMN_ORDER
COLUMN_ORDER = COLUMN_ORDER_FULL[:4]  # Lobe, Delta, Meander, PV-shoestring

# Match figure 3's defaults for the canonical 3-well config.
DEFAULT_CUBES_CACHE = PAPER_DIR / 'figs' / 'figure3_3wells_cubes.npz'
DEFAULT_WELL_FRACTIONS = [0.25, 0.5, 0.75]
Z_SLICE_DEFAULT = paper_fig3.Z_SLICE_DEFAULT
Y_SLICE_DEFAULT = paper_fig3.Y_SLICE_DEFAULT
X_SLICE_DEFAULT = paper_fig3.X_SLICE_DEFAULT
WELL_COLOR = paper_fig3.WELL_COLOR
WELL_LW = paper_fig3.WELL_LW
WELL_ALPHA = paper_fig3.WELL_ALPHA
ROW_KEYS = paper_fig3.ROW_KEYS  # ('xy', 'xz', 'yz')


def load_cubes_cache(cache_path: Path):
    """Return cubes[row_key][lt] -> (X, Y, Z) np.float32 for the 4-layer subset."""
    if not cache_path.exists():
        sys.exit(
            f'Cubes cache not found at {cache_path}. Run paper_figures/'
            'figure3.py once (with the 3-well config) to generate it, or '
            'pass --cubes-cache pointing at a custom .npz.')
    d = np.load(cache_path, allow_pickle=True)
    cubes = {k: {} for k in ROW_KEYS}
    for row_key in ROW_KEYS:
        for _, lt in COLUMN_ORDER:
            key = f'{row_key}__{lt.replace(":", "_")}'
            if key not in d.files:
                sys.exit(f'cache {cache_path} is missing {key!r}')
            cubes[row_key][lt] = d[key]
    return cubes


def well_positions_x(well_fractions):
    return [int(round(VOLUME_SHAPE[0] * f)) for f in well_fractions]


def well_positions_y(well_fractions):
    return [int(round(VOLUME_SHAPE[1] * f)) for f in well_fractions]


def add_well_outlines(ax, row_key, *, well_fractions,
                       z_slice, y_slice, x_slice, vector=False):
    SX, SY, SZ = VOLUME_SHAPE
    x_off = 0.0 if vector else -0.5
    y_off = 0.0 if vector else -0.5
    if row_key == 'xy':
        for x in well_positions_x(well_fractions):
            ax.add_patch(Rectangle((x + x_off, y_off), 1, SY, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))
    elif row_key == 'xz':
        for x in well_positions_x(well_fractions):
            ax.add_patch(Rectangle((x + x_off, y_off), 1, SZ, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))
    elif row_key == 'yz':
        for y in well_positions_y(well_fractions):
            ax.add_patch(Rectangle((y + x_off, y_off), 1, SZ, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))


def _draw_panel(ax, img, *, cmap, norm=None, vmin=None, vmax=None,
                 vector=False):
    if not vector:
        return imshow_slice(ax, img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    H, W = img.shape
    xs = np.linspace(0, W, W + 1)
    ys = np.linspace(0, H, H + 1)
    kw = dict(cmap=cmap, shading='flat', edgecolors='none', linewidth=0,
              antialiased=False, rasterized=False)
    if norm is not None:
        kw['norm'] = norm
    else:
        kw['vmin'] = vmin
        kw['vmax'] = vmax
    artist = ax.pcolormesh(xs, ys, img, **kw)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def render_figure(cubes, out_png, out_pdf=None, *,
                  z_slice, y_slice, x_slice,
                  well_fractions, vector=False,
                  draw_outlines=True, dpi=200):
    n_cols = len(COLUMN_ORDER)
    SX, SY, SZ = VOLUME_SHAPE

    panel_w = 1.10
    row_heights = [panel_w, panel_w / 2.0, panel_w / 2.0]
    label_col_w = 0.55
    fig_w = label_col_w + n_cols * panel_w + 0.20
    fig_h = 0.55 + sum(row_heights) + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h))
    master_widths = [label_col_w] + [panel_w] * n_cols
    master_heights = [0.45, sum(row_heights), 0.45]
    master_gs = fig.add_gridspec(
        nrows=3, ncols=len(master_widths),
        width_ratios=master_widths,
        height_ratios=master_heights,
        left=0.005, right=0.995, top=0.965, bottom=0.04,
        hspace=0.06, wspace=0.0)

    for ci, (pretty, _) in enumerate(COLUMN_ORDER):
        ax = fig.add_subplot(master_gs[0, 1 + ci])
        ax.text(0.5, 0.30, pretty, ha='center', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.set_axis_off()

    if vector:
        xt_64, yt_64 = (0, 32, 64), (0, 32, 64)
        yt_32 = (0, 16, 32)
    else:
        xt_64, yt_64 = (-0.5, 31.5, 63.5), (-0.5, 31.5, 63.5)
        yt_32 = (-0.5, 15.5, 31.5)
    row_specs = [
        ('xy', f'XY (z={z_slice})', slice_xy, z_slice,
         xt_64, ('0', '32', '64'),
         yt_64, ('0', '32', '64')),
        ('xz', f'XZ (y={y_slice})', slice_xz, y_slice,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
        ('yz', f'YZ (x={x_slice})', slice_yz, x_slice,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
    ]

    data_gs = master_gs[1, 1:].subgridspec(
        nrows=3, ncols=n_cols,
        height_ratios=row_heights,
        hspace=0.20, wspace=0.10)

    handle_im = None
    for ri, (row_key, _, slicer, idx,
             xticks, xlabels, yticks, ylabels) in enumerate(row_specs):
        for ci, (_, lt) in enumerate(COLUMN_ORDER):
            cube = cubes[row_key][lt]
            img = slicer(cube, idx)
            ax = fig.add_subplot(data_gs[ri, ci])
            im = _draw_panel(ax, img, cmap=CMAP_FACIES, norm=NORM_FACIES,
                             vector=vector)
            handle_im = im
            if draw_outlines:
                add_well_outlines(ax, row_key,
                                  well_fractions=well_fractions,
                                  z_slice=z_slice, y_slice=y_slice,
                                  x_slice=x_slice, vector=vector)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if ri == 2:
                ax.set_xticklabels(xlabels, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(ylabels, fontsize=7)
            ax.tick_params(axis='both', length=2.5, pad=1.5, direction='in')
            if ri == 2:
                ax_letter = 'Y' if slicer is slice_yz else 'X'
                ax.set_xlabel(ax_letter, fontsize=7, labelpad=1)

    label_sub_gs = master_gs[1, 0].subgridspec(
        nrows=3, ncols=1, height_ratios=row_heights, hspace=0.20)
    for ri, (_rk, row_label, *_rest) in enumerate(row_specs):
        ax = fig.add_subplot(label_sub_gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.85, 0.5, row_label, ha='right', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

    cbar_sub_gs = master_gs[2, 1:].subgridspec(
        nrows=2, ncols=5,
        width_ratios=[1, 1, 4, 1, 1],
        height_ratios=[1, 1],
        hspace=0.0)
    cax = fig.add_subplot(cbar_sub_gs[1, 2])
    cbar = fig.colorbar(handle_im, cax=cax, orientation='horizontal',
                        ticks=[0, 1])
    cbar.set_ticklabels(FACIES_BINARY_LABELS)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cubes-cache', default=str(DEFAULT_CUBES_CACHE),
                    help='path to figure3-style cubes .npz (default: '
                         'paper_figures/figs/figure3_3wells_cubes.npz)')
    ap.add_argument('--out-dir', default=str(PRES_DIR / 'figs'))
    ap.add_argument('--out-name', default='figure2')
    ap.add_argument('--z', type=int, default=Z_SLICE_DEFAULT)
    ap.add_argument('--y', type=int, default=Y_SLICE_DEFAULT)
    ap.add_argument('--x', type=int, default=X_SLICE_DEFAULT)
    ap.add_argument('--well-fractions', type=float, nargs='+',
                    default=DEFAULT_WELL_FRACTIONS,
                    help='well positions as axis fractions (must match the '
                         'config used to generate --cubes-cache; defaults '
                         'to 0.25 0.5 0.75 for the 3-wells cache)')
    ap.add_argument('--vector', action='store_true',
                    help='render with pcolormesh(rasterized=False) so each '
                         'voxel is a true vector cell (sharp at any zoom; '
                         'larger PDF). Default uses imshow (rasterized).')
    ap.add_argument('--no-outlines', action='store_true',
                    help='skip the black well-footprint outline rectangles')
    ap.add_argument('--dpi', type=int, default=400,
                    help='PNG resolution (default: 400)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name
    if args.no_outlines and '_no_outlines' not in out_name:
        out_name = f'{out_name}_no_outlines'
    if args.vector and not out_name.endswith('_vector'):
        out_name = f'{out_name}_vector'

    cubes_cache = Path(args.cubes_cache)
    print(f'Loading cubes cache -> {cubes_cache}')
    cubes = load_cubes_cache(cubes_cache)
    for _, lt in COLUMN_ORDER:
        for row_key in ROW_KEYS:
            assert lt in cubes[row_key]
    print(f'Layers: {[lt for _, lt in COLUMN_ORDER]}')

    out_png = out_dir / f'{out_name}.png'
    out_pdf = out_dir / f'{out_name}.pdf'
    print(f'Rendering -> {out_png}')
    render_figure(cubes, out_png, out_pdf,
                  z_slice=args.z, y_slice=args.y, x_slice=args.x,
                  well_fractions=args.well_fractions,
                  vector=args.vector,
                  draw_outlines=not args.no_outlines,
                  dpi=args.dpi)
    print(f'Done. {out_png}  /  {out_pdf}')


if __name__ == '__main__':
    main()
