"""Presentation Figure 1: real samples for all 8 reservoir architectures,
laid out like NeurIPS Figure 2.

Layout: 3 rows (XY, XZ, YZ) x 8 cols (one per layer type), single field
group rendered with a configurable property colormap. The cubes are the
exact same picks used in `paper_figures/figure1.py` -- the four train-split
medoids (lobe / delta / MEANDER_OXBOW / PV_SHOESTRING) plus the four
test-split overrides for the remaining channel architectures (CB_JIGSAW /
CB_LABYRINTH / SH_PROXIMAL / SH_DISTAL). Picks are read from the existing
`paper_figures/figs/figure1_picks.txt` so the figures are guaranteed to
show identical samples.

Property options (`--property`):
    alluvsim   6-class categorical facies (default)
    binary     2-class binary facies (sand / shale)
    poro       continuous porosity in [0, 0.5]
    perm       continuous log10(permeability), range derived from the cubes

Run:
    python examples/reservoirs/presentation_figures/figure1.py --property alluvsim
    python examples/reservoirs/presentation_figures/figure1.py --property binary
    python examples/reservoirs/presentation_figures/figure1.py --property poro
    python examples/reservoirs/presentation_figures/figure1.py --property perm
    # outputs -> figs/figure1_<property>.{png,pdf}
"""
import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from resflow.utils.plotting_reservoirs import (
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
    CMAP_ALLUVSIM, NORM_ALLUVSIM, ALLUVSIM_LABELS,
    CMAP_PORO, CMAP_PERM, perm_to_logperm,
    slice_xy, slice_xz, slice_yz, imshow_slice,
)


VECTOR = False  # set via --vector for true-vector PDF cells.


def _draw_panel(ax, img, *, cmap, norm=None, vmin=None, vmax=None):
    """imshow (rasterized) or pcolormesh (vector) per global VECTOR flag."""
    if not VECTOR:
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


# Column order = same row order used in paper_figures/figure1.py.
COLUMN_ORDER = [
    ('Lobe',          'lobe'),
    ('Delta',         'delta'),
    ('Meander',       'channel:MEANDER_OXBOW'),
    ('PV\nshoestring','channel:PV_SHOESTRING'),
    ('CB\njigsaw',    'channel:CB_JIGSAW'),
    ('CB\nlabyrinth', 'channel:CB_LABYRINTH'),
    ('SH\nproximal',  'channel:SH_PROXIMAL'),
    ('SH\ndistal',    'channel:SH_DISTAL'),
]

SX, SY, SZ = 64, 64, 32

PAPER_FIG_DIR = Path(__file__).resolve().parent.parent / 'paper_figures' / 'figs'
PICKS_LOG = PAPER_FIG_DIR / 'figure1_picks.txt'


# Per-property config: array name on disk, dtype, optional transform, cmap,
# colorbar config (norm / vmin / vmax / ticks / tick-labels / label).
PROPERTY_SPECS = {
    'alluvsim': {
        'array_name':   'facies_alluvsim',
        'dtype':         np.int8,
        'transform':     None,
        'cmap':          CMAP_ALLUVSIM,
        'norm':          NORM_ALLUVSIM,
        'vlim':          None,
        'cbar_ticks':    [-1, 0, 1, 2, 3, 4],
        'cbar_labels':   ALLUVSIM_LABELS,
        'cbar_title':    None,
    },
    'binary': {
        'array_name':   'facies',
        'dtype':         np.int8,
        'transform':     None,
        'cmap':          CMAP_FACIES,
        'norm':          NORM_FACIES,
        'vlim':          None,
        'cbar_ticks':    [0, 1],
        'cbar_labels':   FACIES_BINARY_LABELS,
        'cbar_title':    None,
    },
    'poro': {
        'array_name':   'poro',
        'dtype':         np.float32,
        'transform':     None,
        'cmap':          CMAP_PORO,
        'norm':          None,
        'vlim':          (0.0, 0.5),
        'cbar_ticks':    None,
        'cbar_labels':   None,
        'cbar_title':    'Porosity',
    },
    'perm': {
        'array_name':   'perm',
        'dtype':         np.float32,
        'transform':     perm_to_logperm,
        'cmap':          CMAP_PERM,
        'norm':          None,
        'vlim':          'auto',  # derived from data
        'cbar_ticks':    None,
        'cbar_labels':   None,
        'cbar_title':    r'$\log_{10}$ Permeability [mD]',
    },
}


def parse_picks_log(path: Path):
    """Parse figure1_picks.txt -> dict[layer_type] = (shard_dir, sample_idx)."""
    out = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                cur = None
                continue
            if not line.startswith(' '):
                cur = line.strip()
                out[cur] = {}
            else:
                m = re.match(r'\s*([A-Za-z_]+)\s*=\s*(\S+)', line)
                if m and cur is not None:
                    out[cur][m.group(1)] = m.group(2)
    picks = {}
    for lt, kv in out.items():
        if 'shard_dir' in kv and 'sample_idx' in kv:
            picks[lt] = {
                'shard_dir': kv['shard_dir'],
                'sample_idx': int(kv['sample_idx']),
            }
    return picks


def ensure_arrays_on_disk(data_dir, shard_dir, names):
    """Make sure each `<name>.npy` exists under data_dir/shard_dir, fetching
    from HuggingFace if missing. Returns dict name -> Path."""
    out = {}
    missing = []
    for name in names:
        p = Path(data_dir) / shard_dir / f'{name}.npy'
        if p.exists():
            out[name] = p
        else:
            missing.append((name, p))
    if not missing:
        return out

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            'huggingface_hub is required to fetch property arrays. '
            'Install with `pip install huggingface_hub`.') from e

    for name, p in missing:
        rel = f'{shard_dir}/{name}.npy'
        print(f'  fetching {rel} ...', flush=True)
        local = hf_hub_download(
            repo_id='SciLM/SiliciclasticReservoirs',
            repo_type='dataset',
            filename=rel,
            local_dir=str(data_dir),
        )
        out[name] = Path(local)
    return out


def load_property_cube(data_dir, shard_dir, idx, *, array_name, dtype,
                        transform):
    paths = ensure_arrays_on_disk(data_dir, shard_dir, names=[array_name])
    arr = np.asarray(np.load(paths[array_name], mmap_mode='r')[idx], dtype=dtype)
    if transform is not None:
        arr = transform(arr)
    return arr


def render_figure(cubes, out_png, out_pdf=None, *,
                  z_idx=18, y_idx=SY // 2, x_idx=SX // 2,
                  cmap, norm=None, vlim=None,
                  cbar_ticks=None, cbar_labels=None, cbar_title=None):
    """3 rows (XY/XZ/YZ) x 8 cols (per layer type), single property cmap."""
    n_cols = len(COLUMN_ORDER)

    panel_w = 1.10
    row_heights_in = [panel_w, panel_w / 2.0, panel_w / 2.0]
    label_col_w = 0.55
    fig_w = label_col_w + n_cols * panel_w + 0.20
    fig_h = 0.55 + sum(row_heights_in) + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h))
    master_widths = [label_col_w] + [panel_w] * n_cols
    master_heights = [0.45, sum(row_heights_in), 0.45]
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

    data_gs = master_gs[1, 1:].subgridspec(
        nrows=3, ncols=n_cols,
        height_ratios=row_heights_in,
        hspace=0.20, wspace=0.10)

    if VECTOR:
        xt_64 = (0, 32, 64); yt_64 = (0, 32, 64); yt_32 = (0, 16, 32)
    else:
        xt_64 = (-0.5, 31.5, 63.5)
        yt_64 = (-0.5, 31.5, 63.5)
        yt_32 = (-0.5, 15.5, 31.5)
    row_specs = [
        (f'XY (z={z_idx})', slice_xy, z_idx,
         xt_64, ('0', '32', '64'),
         yt_64, ('0', '32', '64')),
        (f'XZ (y={y_idx})', slice_xz, y_idx,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
        (f'YZ (x={x_idx})', slice_yz, x_idx,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
    ]

    handle_im = None
    vmin, vmax = (vlim if vlim is not None else (None, None))
    for ri, (row_label, slicer, idx, xticks, xlabels,
             yticks, ylabels) in enumerate(row_specs):
        for ci, (_, lt) in enumerate(COLUMN_ORDER):
            cube = cubes[lt]
            img = slicer(cube, idx)
            ax = fig.add_subplot(data_gs[ri, ci])
            im = _draw_panel(ax, img, cmap=cmap, norm=norm,
                              vmin=vmin, vmax=vmax)
            handle_im = im
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
        nrows=3, ncols=1, height_ratios=row_heights_in, hspace=0.20)
    for ri, (row_label, *_rest) in enumerate(row_specs):
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
    cbar_kw = dict(cax=cax, orientation='horizontal')
    if cbar_ticks is not None:
        cbar_kw['ticks'] = cbar_ticks
    cbar = fig.colorbar(handle_im, **cbar_kw)
    if cbar_labels is not None:
        cbar.set_ticklabels(cbar_labels)
    if cbar_title is not None:
        cbar.set_label(cbar_title, fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',
                   default=os.environ.get(
                       'RESERVOIR_DATA_DIR',
                       os.path.join(os.environ.get('SCRATCH', '/tmp'),
                                    'SiliciclasticReservoirs')))
    p.add_argument('--out-dir',
                   default=str(Path(__file__).resolve().parent / 'figs'))
    p.add_argument('--property', dest='prop',
                   choices=list(PROPERTY_SPECS.keys()), default='alluvsim',
                   help='which property to render (default: alluvsim)')
    p.add_argument('--out-name', default=None,
                   help='override output basename (default: figure1_<property>)')
    p.add_argument('--picks-log', default=str(PICKS_LOG),
                   help='picks log written by paper_figures/figure1.py')
    p.add_argument('--z', type=int, default=18)
    p.add_argument('--y', type=int, default=SY // 2)
    p.add_argument('--x', type=int, default=SX // 2)
    p.add_argument('--vector', action='store_true',
                   help='render with pcolormesh(rasterized=False) so each '
                        'voxel is a true vector cell (sharp at any zoom; '
                        'much larger PDF). Default uses imshow (rasterized).')
    args = p.parse_args()
    global VECTOR
    VECTOR = args.vector

    spec = PROPERTY_SPECS[args.prop]
    out_name = args.out_name or f'figure1_{args.prop}'
    if VECTOR and not out_name.endswith('_vector'):
        out_name = f'{out_name}_vector'

    picks_log = Path(args.picks_log)
    if not picks_log.exists():
        sys.exit(f'Picks log not found at {picks_log}. Run '
                 'paper_figures/figure1.py once to generate it.')

    print(f'Reading picks log -> {picks_log}')
    picks = parse_picks_log(picks_log)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {spec["array_name"]} cubes (auto-fetching from HF if needed) ...')
    cubes = {}
    for pretty, lt in COLUMN_ORDER:
        if lt not in picks:
            sys.exit(f'Picks log is missing entry for {lt!r}; '
                     'regenerate paper_figures/figure1.py first.')
        info = picks[lt]
        cubes[lt] = load_property_cube(
            data_dir, info['shard_dir'], info['sample_idx'],
            array_name=spec['array_name'], dtype=spec['dtype'],
            transform=spec['transform'])
        print(f'  {lt:<28s}  {info["shard_dir"]}#{info["sample_idx"]}')

    vlim = spec['vlim']
    if vlim == 'auto':
        all_vals = np.concatenate([c.ravel() for c in cubes.values()])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        vlim = (vmin, vmax)
        print(f'  auto vlim = ({vmin:.3f}, {vmax:.3f})')

    out_png = out_dir / f'{out_name}.png'
    out_pdf = out_dir / f'{out_name}.pdf'
    print(f'Rendering -> {out_png}')
    render_figure(
        cubes, out_png, out_pdf,
        z_idx=args.z, y_idx=args.y, x_idx=args.x,
        cmap=spec['cmap'], norm=spec['norm'], vlim=vlim,
        cbar_ticks=spec['cbar_ticks'],
        cbar_labels=spec['cbar_labels'],
        cbar_title=spec['cbar_title'],
    )
    print(f'Done. {out_png}  /  {out_pdf}')


if __name__ == '__main__':
    main()
