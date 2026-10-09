"""NeurIPS Figure 1: real samples from 4 reservoir types at typical (medoid) cond.

Layout: 4 rows (lobe, delta, channel:MEANDER_OXBOW, channel:PV_SHOESTRING)
        x 9 columns = 3 fields x 3 orthogonal slices each.

Fields shown (left-to-right column groups, each is XY @ z_mid | XZ @ y_mid | YZ @ x_mid):
    1. binary facies              (RESERVOIR_CMAP, range [0, 1])
    2. 6-class facies_alluvsim   (categorical -1..4, ALLUVSIM_FACIES_COLORS)
    3. porosity                   (RESERVOIR_CMAP, range [0, 0.5])

Color schemes follow a canonical earth-science palette
(grey for mud, mustard / burnt-orange for sand).

"Typical mean conditioning" = sample whose param vector (ntg, width_cells,
depth_cells, family-specific cols when present; azimuth excluded since it just
rotates the same geology) is closest to the per-family mean after per-column
z-scoring. Search runs over the train split via the cached condition arrays.

The script auto-fetches poro/facies_alluvsim for the 4 picked shards from
HuggingFace if they aren't already on disk locally.

Run:
    python examples/reservoirs/paper_figures/figure1.py
    # output -> examples/reservoirs/paper_figures/figs/figure1.{png,pdf}
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from resflow.utils.data_reservoirs import (
    LAYER_TYPES, LAYER_TYPE_TO_IDX, CONT_COLS,
)
from resflow.utils.plotting_reservoirs import (
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
    CMAP_PORO,
    CMAP_ALLUVSIM, NORM_ALLUVSIM, ALLUVSIM_LABELS,
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


# Per-type pretty names + which layer_type strings to use. Row order
# matches Figure 2's column order so the two figures read consistently.
TYPE_SPECS = [
    ('Lobe',          'lobe'),
    ('Delta',         'delta'),
    ('Meander',       'channel:MEANDER_OXBOW'),
    ('PV\nshoestring','channel:PV_SHOESTRING'),
    ('CB\njigsaw',    'channel:CB_JIGSAW'),
    ('CB\nlabyrinth', 'channel:CB_LABYRINTH'),
    ('SH\nproximal',  'channel:SH_PROXIMAL'),
    ('SH\ndistal',    'channel:SH_DISTAL'),
]

# Cube shape (X, Y, Z) per dataset README.
SX, SY, SZ = 64, 64, 32

# Per-type overrides for slice indices. Keys are layer_type strings; values
# are dicts with optional 'z', 'y', 'x' overriding SZ//2 / SY//2 / SX//2.
SLICE_OVERRIDES = {
    'lobe':                   {'z': 18},
    'delta':                  {'z': 18},
    'channel:MEANDER_OXBOW':  {'z': 18},
    'channel:PV_SHOESTRING':  {'z': 18},
    'channel:CB_JIGSAW':      {'z': 18},
    'channel:CB_LABYRINTH':   {'z': 18},
    'channel:SH_PROXIMAL':    {'z': 18},
    'channel:SH_DISTAL':      {'z': 18},
}

# Per-type sample overrides. If a layer_type appears here, the medoid search
# is bypassed and the listed sample is used instead. Useful when the medoid
# is geologically uninteresting (e.g. amalgamated delta -> hand-pick a fanned
# example via find_delta_candidate.py). 'split' selects which cond cache to
# look the row up in (defaults to 'train'); the four channel-type picks below
# are the same TEST-split rows that Figure 2 uses (mirroring
# inpainting/sample.py with rng=default_rng(0)).
SAMPLE_OVERRIDES = {
    # k=11 from find_delta_candidate.py: NTG=0.40 (~family mean), avul=0.79,
    # short trunk, az=3 -- spread-out distributary fan with separated channels.
    'delta': {'shard_dir': 'delta/shard_0078', 'sample_idx': 376},
    # Mirror Figure 2's inpainting picks (TEST split, np.random.default_rng(0)
    # iterated through LAYER_TYPES exactly as in inpainting/sample.py) so the
    # real-vs-generated comparison uses identical conditioning for these four
    # channel architectures.
    'channel:CB_JIGSAW':    {'shard_dir': 'channel_cb_jigsaw/shard_0203',
                             'sample_idx': 257, 'split': 'test'},
    'channel:CB_LABYRINTH': {'shard_dir': 'channel_cb_labyrinth/shard_0242',
                             'sample_idx': 288, 'split': 'test'},
    'channel:SH_PROXIMAL':  {'shard_dir': 'channel_sh_proximal/shard_0082',
                             'sample_idx': 262, 'split': 'test'},
    'channel:SH_DISTAL':    {'shard_dir': 'channel_sh_distal/shard_0166',
                             'sample_idx': 0,   'split': 'test'},
}

# Per-column mask of which cont_raw columns are universal (use for every type).
# CONT_COLS = ['ntg','width_cells','depth_cells','asp','mCHsinu','mFFCHprop',
#              'probAvulInside','trunk_length_fraction']
UNIVERSAL_IDX = [0, 1, 2]            # ntg, width_cells, depth_cells


def find_medoid(cache_npz, layer_type):
    """Return (shard_dir, sample_idx_within_shard, cond_dict) for the train-split
    sample whose conditioning is closest to the family mean.

    For lobe rows we additionally use 'asp'. For delta rows we additionally use
    {mCHsinu, mFFCHprop, probAvulInside, trunk_length_fraction}. For channel
    rows {mCHsinu, mFFCHprop, probAvulInside}. Azimuth is excluded -- it just
    rotates the same geology.
    """
    d = np.load(cache_npz, allow_pickle=True)
    shard_dirs = list(d['shard_dirs'])
    layer_idx = d['layer_idx']
    sample_idx = d['sample_idx']
    shard_keys = d['shard_keys']
    cont = d['cont_raw']

    target_idx = LAYER_TYPE_TO_IDX[layer_type]
    mask = layer_idx == target_idx
    rows = np.where(mask)[0]
    sub_cont = cont[rows]                   # (n_rows, 8)

    # Cols to use: universal + any column that has a real value for >50% of rows
    # in this family (i.e., the family-specific columns that apply here).
    nan_frac = np.isnan(sub_cont).mean(axis=0)
    use_cols = [c for c in range(len(CONT_COLS))
                if c in UNIVERSAL_IDX or nan_frac[c] < 0.5]

    X = sub_cont[:, use_cols]
    # Z-score per column over this family.
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    # Replace NaNs with 0 (column mean after z-score) just in case any leak in.
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    dists = np.linalg.norm(X_norm, axis=1)
    best_local = int(np.argmin(dists))
    best_global = int(rows[best_local])

    cond_used = {CONT_COLS[c]: float(sub_cont[best_local, c])
                 for c in use_cols if not np.isnan(sub_cont[best_local, c])}
    family_mean = {CONT_COLS[c]: float(mu[k]) for k, c in enumerate(use_cols)}

    return {
        'shard_dir': str(shard_dirs[int(shard_keys[best_global])]),
        'sample_idx': int(sample_idx[best_global]),
        'cond_used': cond_used,
        'family_mean': family_mean,
        'azimuth': float(d['azimuth'][best_global]),
    }


def lookup_sample(cache_npz, layer_type, shard_dir, sample_idx):
    """Find the row in the train cond cache matching (shard_dir, sample_idx)
    for a given layer_type, and return the same dict shape as find_medoid.
    Used by SAMPLE_OVERRIDES so the picks log still has cond/azimuth values.
    """
    d = np.load(cache_npz, allow_pickle=True)
    shard_dirs = list(d['shard_dirs'])
    layer_idx = d['layer_idx']
    sample_idx_arr = d['sample_idx']
    shard_keys = d['shard_keys']
    cont = d['cont_raw']

    target_layer = LAYER_TYPE_TO_IDX[layer_type]
    try:
        target_shard_key = shard_dirs.index(shard_dir)
    except ValueError:
        raise ValueError(
            f'shard_dir {shard_dir!r} not present in cond cache') from None

    hit = np.where(
        (layer_idx == target_layer)
        & (shard_keys == target_shard_key)
        & (sample_idx_arr == sample_idx))[0]
    if len(hit) == 0:
        raise ValueError(
            f'No row in cond cache for layer_type={layer_type!r} '
            f'shard_dir={shard_dir!r} sample_idx={sample_idx}')
    gi = int(hit[0])

    # Compute family mean over rows of this layer_type for the same column set.
    family_rows = np.where(layer_idx == target_layer)[0]
    sub_cont = cont[family_rows]
    nan_frac = np.isnan(sub_cont).mean(axis=0)
    use_cols = [c for c in range(len(CONT_COLS))
                if c in UNIVERSAL_IDX or nan_frac[c] < 0.5]
    mu = np.nanmean(sub_cont[:, use_cols], axis=0)

    cond_used = {CONT_COLS[c]: float(cont[gi, c])
                 for c in use_cols if not np.isnan(cont[gi, c])}
    family_mean = {CONT_COLS[c]: float(mu[k]) for k, c in enumerate(use_cols)}

    return {
        'shard_dir': shard_dir,
        'sample_idx': int(sample_idx),
        'cond_used': cond_used,
        'family_mean': family_mean,
        'azimuth': float(d['azimuth'][gi]),
    }


def ensure_arrays_on_disk(data_dir, shard_dir, names):
    """Make sure each `<name>.npy` exists under data_dir/shard_dir, fetching from
    HuggingFace if missing. Returns dict name -> Path."""
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
            'huggingface_hub is required to fetch poro/perm/facies_alluvsim. '
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


def load_sample_arrays(data_dir, shard_dir, idx):
    """Return dict with binary facies, 6-class facies, poro, perm cubes for one
    sample. Each is a (64, 64, 32) array."""
    paths = ensure_arrays_on_disk(
        data_dir, shard_dir,
        names=['facies', 'facies_alluvsim', 'poro'])

    out = {}
    out['facies'] = np.asarray(np.load(paths['facies'], mmap_mode='r')[idx],
                               dtype=np.int8)
    out['facies_alluvsim'] = np.asarray(
        np.load(paths['facies_alluvsim'], mmap_mode='r')[idx], dtype=np.int8)
    out['poro'] = np.asarray(np.load(paths['poro'], mmap_mode='r')[idx],
                             dtype=np.float32)
    return out


def build_figure(picks, cubes, out_path_png, out_path_pdf=None):
    """Render the 4-row x 9-col figure. `cubes[type_name]` is the dict returned
    by load_sample_arrays.

    Layout: 1 left column for row labels + 3 field-group blocks side-by-side.
    Each field-group block is a 3-col sub-gridspec [XY | XZ | YZ] with
    width_ratios = [1, 2, 2] (XZ/YZ are 64x32 vs XY's 64x64 so they need 2x
    the axes width to render the same height with aspect='equal' and not leave
    big white margins).
    """
    n_rows = len(picks)
    field_groups = [
        ('Binary facies',           'facies',           CMAP_FACIES,   None,       NORM_FACIES),
        ('6-class facies (alluvsim)', 'facies_alluvsim', CMAP_ALLUVSIM, None,       NORM_ALLUVSIM),
        ('Porosity',                'poro',             CMAP_PORO,     (0.0, 0.5), None),
    ]
    n_fg = len(field_groups)

    # Each row: ~1.6 inches tall. XY panel = ~1.6" wide, XZ/YZ = ~1.6" wide
    # (same axes width via width_ratios but image fills it horizontally because
    # cube X = 2 * Z, with aspect='equal' the image height is half the axes
    # width — a long thin band, exactly what we want).
    label_col_w = 1.10        # wider so horizontal labels fit
    # panel_unit controls the column width unit (XY=1u, XZ=YZ=2u). With
    # aspect='equal', if columns are wider than the per-row height, axes
    # shrink and leave horizontal whitespace inside each gridspec cell.
    # Shrinking panel_unit to match the per-row height (after data-row
    # hspace eats some vertical space) keeps the axes filling their cells
    # and tightens the inter-column gaps.
    panel_unit = 0.85
    fg_w = panel_unit * (1 + 2 + 2)   # width per field group
    sep_w = 0.18              # gap between field groups
    fig_w = label_col_w + n_fg * fg_w + (n_fg - 1) * sep_w + 0.15
    row_h = panel_unit * 1.0
    fig_h = 0.55 + n_rows * row_h + 0.55  # header + rows + colorbar

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Master gridspec: cols = [labels, fg0, sep, fg1, sep, fg2, sep, fg3]
    master_widths = [label_col_w]
    for fi in range(n_fg):
        master_widths.append(fg_w)
        if fi < n_fg - 1:
            master_widths.append(sep_w)
    master_gs = fig.add_gridspec(
        nrows=3, ncols=len(master_widths),
        width_ratios=master_widths,
        height_ratios=[0.35, n_rows * row_h, 0.45],
        left=0.005, right=0.995, top=0.965, bottom=0.04,
        hspace=0.06, wspace=0.0)

    # Slice indices: middle of each axis.
    z_mid = SZ // 2
    y_mid = SY // 2
    x_mid = SX // 2

    cbar_handles = {}

    # Helper: master-gs col index for field-group fi (account for label col + separators).
    def fg_master_col(fi):
        return 1 + 2 * fi

    # --- Header row: one cell per field group with the group title + slice subtitles ---
    for fi, (fg_label, _, _, _, _) in enumerate(field_groups):
        ax = fig.add_subplot(master_gs[0, fg_master_col(fi)])
        ax.text(0.5, 0.65, fg_label, ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        # x-positions for sub-panel labels = cumulative width-ratio midpoints in [1,2,2]
        sub_widths = [1.0, 2.0, 2.0]
        total = sum(sub_widths)
        x_centers = []
        running = 0.0
        for w in sub_widths:
            x_centers.append((running + w / 2) / total)
            running += w
        for x, lab in zip(x_centers, ['XY (z=18)', 'XZ (y=32)', 'YZ (x=32)']):
            ax.text(x, -0.18, lab, ha='center', va='bottom',
                    fontsize=11, color='0.30', transform=ax.transAxes)
        ax.set_axis_off()

    # --- Data rows: each field-group block is a (n_rows x 3) sub-gridspec ---
    for fi, (fg_label, key, cmap, vlim, norm) in enumerate(field_groups):
        # All field groups show y-tick labels for every slice type, so we
        # need a tiny wspace so the XZ/YZ tick labels don't overlap the
        # neighbouring panel.
        sub_gs = master_gs[1, fg_master_col(fi)].subgridspec(
            nrows=n_rows, ncols=3,
            width_ratios=[1, 2, 2],
            hspace=0.20, wspace=0.10)

        for ri, (_, layer_type) in enumerate(TYPE_SPECS):
            sample = cubes[layer_type]
            vol = sample[key]
            if vlim is not None:
                vmin, vmax = vlim
            else:
                vmin = vmax = None

            ov = SLICE_OVERRIDES.get(layer_type, {})
            z_idx = ov.get('z', z_mid)
            y_idx = ov.get('y', y_mid)
            x_idx = ov.get('x', x_mid)
            for si, slicer in enumerate((slice_xy, slice_xz, slice_yz)):
                ax = fig.add_subplot(sub_gs[ri, si])
                if slicer is slice_xy:
                    img = slicer(vol, z_idx)
                elif slicer is slice_xz:
                    img = slicer(vol, y_idx)
                else:
                    img = slicer(vol, x_idx)
                im = _draw_panel(ax, img, cmap=cmap, vmin=vmin, vmax=vmax,
                                  norm=norm)
                # Restore tick MARKS so the reader sees the grid extent.
                # X-axis is 64 voxels wide for all 3 slices; Y-axis is 64
                # for XY (si=0) and 32 for XZ/YZ (si=1,2). Tick positions
                # depend on the renderer:
                #   imshow (rasterized): pixel centres at int -> ticks at
                #     [-0.5, N/2-0.5, N-0.5] for labels [0, N/2, N].
                #   pcolormesh (vector): cell edges at int -> ticks at
                #     [0, N/2, N] for the same labels.
                if VECTOR:
                    xt_64 = (0, 32, 64)
                    yt_64 = (0, 32, 64)
                    yt_32 = (0, 16, 32)
                else:
                    xt_64 = (-0.5, 31.5, 63.5)
                    yt_64 = (-0.5, 31.5, 63.5)
                    yt_32 = (-0.5, 15.5, 31.5)
                ax.set_xticks(xt_64)
                if si == 0:
                    ax.set_yticks(yt_64)
                    y_labels = ['0', '32', '64']
                else:
                    ax.set_yticks(yt_32)
                    y_labels = ['0', '16', '32']
                # Tick LABELS only on the bottom row (x) and the leftmost
                # cell of each field group (y) -- everywhere else the
                # tick marks are kept but the labels suppressed so the
                # 36-panel grid stays uncluttered.
                if ri == n_rows - 1:
                    ax.set_xticklabels(['0', '32', '64'], fontsize=11)
                    # XY and XZ panels share X for their horizontal axis; YZ
                    # uses Y. Annotate the physical axis so a reader doesn't
                    # have to re-derive it from the slice header.
                    ax_letter = 'Y' if slicer is slice_yz else 'X'
                    ax.set_xlabel(ax_letter, fontsize=11, labelpad=1)
                else:
                    ax.set_xticklabels([])
                # Show y-tick labels for every slice type in every field
                # group, exposing both extents (XY: 0/32/64, XZ/YZ: 0/16/32).
                ax.set_yticklabels(y_labels, fontsize=11)
                ax.tick_params(axis='both', length=2.5, pad=1.5,
                               direction='in')
                cbar_handles[fi] = (im, fg_label, key)

    # --- Row labels: 4 invisible axes aligned 1:1 with the data row sub-gridspec.
    # Horizontal text right-aligned against the data panels so long names
    # ("Shoestring") never collide. ---
    label_sub_gs = master_gs[1, 0].subgridspec(
        nrows=n_rows, ncols=1, hspace=0.20)
    for ri, (pretty_name, _) in enumerate(TYPE_SPECS):
        ax = fig.add_subplot(label_sub_gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.82, 0.5, pretty_name, ha='right', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)

    # --- Colorbars: one per field-group, in the bottom master row ---
    for fi, (fg_label, key, cmap, vlim, norm) in enumerate(field_groups):
        # Use a sub-gridspec to inset the cbar so it doesn't span the full
        # group width (central 60%) AND to push it toward the bottom of
        # the master row, leaving an explicit gap above the colorbar
        # without inflating master hspace (which would also push the
        # XY/XZ/YZ subtitles away from the data panels).
        sub_gs = master_gs[2, fg_master_col(fi)].subgridspec(
            nrows=2, ncols=5,
            width_ratios=[1, 1, 6, 1, 1],
            height_ratios=[1, 1],
            hspace=0.0)
        cax = fig.add_subplot(sub_gs[1, 2])
        im = cbar_handles[fi][0]
        if key == 'facies_alluvsim':
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal',
                                ticks=[-1, 0, 1, 2, 3, 4])
            cbar.set_ticklabels(ALLUVSIM_LABELS)
            cbar.ax.tick_params(labelsize=11)
        elif key == 'facies':
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal',
                                ticks=[0, 1])
            cbar.set_ticklabels(FACIES_BINARY_LABELS)
            cbar.ax.tick_params(labelsize=11)
        else:
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=11)

    fig.savefig(out_path_png, dpi=200, bbox_inches='tight')
    if out_path_pdf is not None:
        fig.savefig(out_path_pdf, bbox_inches='tight')
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
    p.add_argument('--out-name', default='figure1',
                   help='base name for output png/pdf (default: figure1)')
    p.add_argument('--vector', action='store_true',
                   help='render with pcolormesh(rasterized=False) so each '
                        'voxel is a true vector cell (sharp at any zoom; '
                        'much larger PDF). Default uses imshow (rasterized).')
    args = p.parse_args()
    global VECTOR
    VECTOR = args.vector

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cache = data_dir / '_cond_cache' / 'train.npz'
    test_cache = data_dir / '_cond_cache' / 'test.npz'
    if not train_cache.exists():
        sys.exit(f'Train cond cache not found at {train_cache}. Run any '
                 'reservoir training script once to build it (or instantiate '
                 'ReservoirDataset(split="train")).')

    print('Picking medoid sample per type (or override if listed) ...')
    picks = {}
    for pretty, lt in TYPE_SPECS:
        if lt in SAMPLE_OVERRIDES:
            ov = SAMPLE_OVERRIDES[lt]
            split = ov.get('split', 'train')
            cache = test_cache if split == 'test' else train_cache
            if not cache.exists():
                sys.exit(f'{split} cond cache not found at {cache}; '
                         'instantiate ReservoirDataset(split=...) once.')
            picks[lt] = lookup_sample(cache, lt, ov['shard_dir'], ov['sample_idx'])
            tag = f'(override:{split})'
        else:
            picks[lt] = find_medoid(train_cache, lt)
            tag = '(medoid)'
        used = picks[lt]['cond_used']
        mean = picks[lt]['family_mean']
        print(f'  {pretty:>30s} {tag}  shard={picks[lt]["shard_dir"]:<35s} '
              f'idx={picks[lt]["sample_idx"]:>5d}')
        for k in used:
            print(f'      {k:>22s}  sample={used[k]:>8.4f}   mean={mean[k]:>8.4f}')

    print('Loading cubes (auto-fetches poro/perm/facies_alluvsim if missing) ...')
    cubes = {}
    for _, lt in TYPE_SPECS:
        info = picks[lt]
        cubes[lt] = load_sample_arrays(data_dir, info['shard_dir'],
                                       info['sample_idx'])

    out_png = out_dir / f'{args.out_name}.png'
    out_pdf = out_dir / f'{args.out_name}.pdf'
    print(f'Rendering -> {out_png}')
    build_figure(picks, cubes, out_png, out_pdf)
    print(f'Done. {out_png}  /  {out_pdf}')

    # Also drop a small machine-readable record of which samples were chosen.
    log_path = out_dir / 'figure1_picks.txt'
    with log_path.open('w') as f:
        for pretty, lt in TYPE_SPECS:
            info = picks[lt]
            f.write(f'{lt}\n')
            f.write(f'  shard_dir = {info["shard_dir"]}\n')
            f.write(f'  sample_idx = {info["sample_idx"]}\n')
            f.write(f'  azimuth = {info["azimuth"]:.2f}\n')
            for k, v in info['cond_used'].items():
                f.write(f'  {k} = {v:.6f}  (family mean {info["family_mean"][k]:.6f})\n')
            f.write('\n')
    print(f'Picks logged -> {log_path}')


if __name__ == '__main__':
    main()
