"""Presentation Figure 3: ResFlow schematic.

Shows how the model takes a conditioning vector + a sparse well-log
constraint and produces a full XZ reservoir cross-section that honors
both. Uses a high-NTG lobe medoid (a "sandy" lobe so the schematic looks
clean, picked by --ntg-min) with a single well at (x=x_well, y=y_slice,
full z).

Layout (left -> right):
    [ conditions text ]
                       \\__-> [ ResFlow box ] -> [ large XZ output ]
    [ well-only panel ] /                         (with black well outline)

  - The conditions panel lists the lobe cond scalars (layer type, NTG,
    width/depth, aspect, azimuth).
  - The well-only panel shows just the column of facies values supplied
    by the well at x=x_well; voxels outside the well are WHITE (unknown),
    not grey (which would be misleading since grey == "shale").
  - The output panel shows the FM-inpaint cube's XZ slice at y=y_slice
    with the well column outlined in black.

The first invocation searches the train cond cache for a lobe sample
with NTG >= --ntg-min and that is closest to the family mean for
width/depth/asp, then runs FM-inpaint with one well to produce the cube
and caches everything to figs/figure3_lobe_cube.npz. Subsequent runs are
near-instant.

Run:
    python examples/reservoirs/presentation_figures/figure3.py
    # regenerate with a different NTG floor:
    python examples/reservoirs/presentation_figures/figure3.py --regenerate --ntg-min 0.80
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PRES_DIR = Path(__file__).resolve().parent
PAPER_DIR = PRES_DIR.parent / 'paper_figures'
sys.path.insert(0, str(PAPER_DIR))

from resflow.utils.plotting_reservoirs import (  # noqa: E402
    CMAP_FACIES, NORM_FACIES, slice_xz, imshow_slice,
)


SX, SY, SZ = 64, 64, 32
DEFAULT_Y_SLICE = 32
DEFAULT_X_WELL = 32        # 1 well at fraction 0.5 -> x = 32
DEFAULT_NTG_MIN = 0.75
DEFAULT_SEED = 7

DEFAULT_CACHE = PRES_DIR / 'figs' / 'figure3_lobe_cube.npz'

# Same hex colors as CMAP_FACIES, but with `bad` (NaN) cells -> white so the
# unknown voxels in the well-log input panel render as white instead of grey
# (grey is the binary "shale" color, which would be misleading).
CMAP_WELL = ListedColormap(['#999999', '#b85a18'], name='facies_well')
CMAP_WELL.set_bad('white')


# ----------------- pick / sample (run once, cache the result) ----------------

def pick_high_ntg_lobe(data_dir, ntg_min):
    """Search the train cond cache for a lobe sample with NTG >= ntg_min and
    closest to family mean for the remaining cont cols (width/depth/asp).
    Returns dict[shard_dir, sample_idx, azimuth, cont_raw, cond_dict]."""
    from resflow.assembly import LAYER_TYPE_TO_IDX
    from resflow.utils.data_reservoirs import CONT_COLS

    cache_path = Path(data_dir) / '_cond_cache' / 'train.npz'
    if not cache_path.exists():
        sys.exit(f'Train cond cache not found at {cache_path}.')
    d = np.load(cache_path, allow_pickle=True)
    layer_idx = d['layer_idx']
    cont = d['cont_raw']
    shard_keys = d['shard_keys']
    sample_idx_arr = d['sample_idx']
    shard_dirs = list(d['shard_dirs'])
    azimuth = d['azimuth']

    target_layer = LAYER_TYPE_TO_IDX['lobe']
    rows = np.where((layer_idx == target_layer) & (cont[:, 0] >= ntg_min))[0]
    if len(rows) == 0:
        raise RuntimeError(f'No lobe samples with NTG >= {ntg_min}')

    # Family-mean medoid over (width_cells, depth_cells, asp) for the
    # filtered subset.
    sub = cont[rows]
    cols = [1, 2, 3]
    X = sub[:, cols]
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    Xn = np.nan_to_num((X - mu) / sigma, nan=0.0)
    best = int(np.argmin(np.linalg.norm(Xn, axis=1)))
    gi = int(rows[best])

    cond_dict = {col: float(cont[gi, k])
                 for k, col in enumerate(CONT_COLS)
                 if not np.isnan(cont[gi, k])}
    return {
        'shard_dir': shard_dirs[int(shard_keys[gi])],
        'sample_idx': int(sample_idx_arr[gi]),
        'azimuth': float(azimuth[gi]),
        'cont_raw': cont[gi].astype(np.float32),
        'cond_dict': cond_dict,
    }


def generate_lobe_cube(info, data_dir, ckpt, cond_stats, device, *,
                        x_well, y_slice, seed):
    """Run FM-inpaint on the picked lobe sample with a single well at
    (x_well, y_slice, full z). Returns (cube np.float32, cond_vec np.float32).
    """
    import torch
    from resflow.assembly import (
        BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX, build_cond_vector,
    )
    from resflow.methods.flow_matching import FlowMatching
    from resflow.models.unet3d import UNet3D
    from resflow.utils.data_reservoirs import ReservoirDataset, VOLUME_SHAPE
    from resflow.utils.masking import apply_inpaint_output

    stats = np.load(cond_stats, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    train_set = ReservoirDataset(data_dir, split='train',
                                 cont_min=cont_min, cont_max=cont_max)
    target_shard = train_set.shard_dirs.index(info['shard_dir'])
    hits = np.where(
        (train_set.layer_idx == LAYER_TYPE_TO_IDX['lobe'])
        & (train_set.shard_keys == target_shard)
        & (train_set.sample_idx == info['sample_idx']))[0]
    if len(hits) == 0:
        raise RuntimeError('Picked row not found in train split.')
    real, _ = train_set[int(hits[0])]   # (1, X, Y, Z), {-1, +1}

    spec = BlockSpec(
        layer_idx=LAYER_TYPE_TO_IDX['lobe'],
        azimuth_deg=info['azimuth'],
        raw_scalars=info['cond_dict'],
    )
    cond_vec = build_cond_vector(spec, cont_min, cont_max)

    X, Y, Z = VOLUME_SHAPE
    mask = torch.zeros(1, 1, X, Y, Z, device=device)
    mask[0, 0, x_well, y_slice, :] = 1.0
    real_d = real.to(device).unsqueeze(0)            # (1, 1, X, Y, Z)
    known = real_d * mask

    print(f'Loading checkpoint -> {ckpt}', flush=True)
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    method = FlowMatching(model)

    cond_t = torch.from_numpy(cond_vec).float().unsqueeze(0).to(device)
    with torch.no_grad():
        model.set_inpaint_context(mask, known)
        torch.manual_seed(seed)
        s = method.sample((1, 1, X, Y, Z), device,
                          cond=cond_t, cfg_scale=3.0, n_steps=50)
        s = apply_inpaint_output(s, mask, known)
        model.clear_inpaint_context()
    cube = s[0, 0].cpu().numpy().astype(np.float32)
    print(f'  generated NTG = {(cube > 0).mean():.3f}', flush=True)
    return cube, cond_vec.astype(np.float32)


def load_or_generate(cache_path, data_dir, ckpt, cond_stats,
                      *, ntg_min, x_well, y_slice, seed, regenerate):
    """Load from cache, or pick + run inference + cache + return."""
    import torch
    if cache_path.exists() and not regenerate:
        d = np.load(cache_path, allow_pickle=True)
        info = {
            'shard_dir':    str(d['shard_dir']),
            'sample_idx':   int(d['sample_idx']),
            'azimuth':      float(d['azimuth']),
            'ntg':          float(d['ntg']),
            'width_cells':  float(d['width_cells']),
            'depth_cells':  float(d['depth_cells']),
            'asp':          float(d['asp']),
            'x_well':       int(d['x_well']),
            'y_slice':      int(d['y_slice']),
        }
        cube = d['cube']
        print(f'Loaded cached cube -> {cache_path}')
        print(f'  shard_dir={info["shard_dir"]}  sample_idx={info["sample_idx"]}  '
              f'NTG={info["ntg"]:.3f}')
        return cube, info

    print(f'Picking lobe sample with NTG >= {ntg_min} ...')
    pick = pick_high_ntg_lobe(data_dir, ntg_min)
    cd = pick['cond_dict']
    print(f"  picked shard_dir={pick['shard_dir']}  sample_idx={pick['sample_idx']}")
    print(f"  ntg={cd.get('ntg', float('nan')):.3f}  "
          f"width={cd.get('width_cells', float('nan')):.1f}  "
          f"depth={cd.get('depth_cells', float('nan')):.1f}  "
          f"asp={cd.get('asp', float('nan')):.2f}  "
          f"az={pick['azimuth']:.1f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Generating with FM-inpaint on {device} ...')
    cube, _ = generate_lobe_cube(
        pick, data_dir, ckpt, cond_stats, device,
        x_well=x_well, y_slice=y_slice, seed=seed)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             cube=cube,
             shard_dir=pick['shard_dir'],
             sample_idx=pick['sample_idx'],
             azimuth=pick['azimuth'],
             ntg=cd.get('ntg', float('nan')),
             width_cells=cd.get('width_cells', float('nan')),
             depth_cells=cd.get('depth_cells', float('nan')),
             asp=cd.get('asp', float('nan')),
             x_well=x_well, y_slice=y_slice)
    print(f'Cached -> {cache_path}')

    return cube, {
        'shard_dir':    pick['shard_dir'],
        'sample_idx':   pick['sample_idx'],
        'azimuth':      pick['azimuth'],
        'ntg':          cd.get('ntg', float('nan')),
        'width_cells':  cd.get('width_cells', float('nan')),
        'depth_cells':  cd.get('depth_cells', float('nan')),
        'asp':          cd.get('asp', float('nan')),
        'x_well':       x_well,
        'y_slice':      y_slice,
    }


# ------------------------------- rendering -----------------------------------

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


def render_schematic(cube, info, out_png, out_pdf=None, *,
                     vector=False, dpi=400):
    """Compose the schematic on a single figure with manually-placed axes."""
    y_slice = int(info['y_slice'])
    x_well = int(info['x_well'])
    xz_full = slice_xz(cube, y_slice)                     # (Z, X), {-1, +1}
    xz_binary = (xz_full > 0).astype(np.float32)          # {0, 1}

    # Well-only input: NaN everywhere, real binary values along the column
    # at x=x_well. NaN cells render as white via CMAP_WELL.set_bad('white').
    well_only = np.full_like(xz_binary, np.nan)
    well_only[:, x_well] = xz_binary[:, x_well]
    well_masked = np.ma.masked_invalid(well_only)

    fig = plt.figure(figsize=(11, 4))

    # ---- LEFT: conditioning text (compact) ----
    ax_cond = fig.add_axes([0.020, 0.52, 0.235, 0.45])
    ax_cond.set_axis_off()
    ax_cond.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.02,rounding_size=0.04',
        facecolor='#f4f4f4', edgecolor='0.30', linewidth=0.8,
        transform=ax_cond.transAxes))
    ax_cond.text(0.5, 0.91, 'Conditioning', fontsize=11, fontweight='bold',
                 ha='center', va='top', transform=ax_cond.transAxes)
    cond_lines = [
        f'Layer type  =  lobe',
        f'NTG  =  {info["ntg"]:.2f}',
        f'Width  =  {info["width_cells"]:.1f} cells',
        f'Depth  =  {info["depth_cells"]:.1f} cells',
        f'Aspect  =  {info["asp"]:.2f}',
        f'Azimuth  =  {info["azimuth"]:.1f}°',
    ]
    for i, txt in enumerate(cond_lines):
        ax_cond.text(0.08, 0.76 - i * 0.115, txt, fontsize=10,
                     ha='left', va='center', transform=ax_cond.transAxes,
                     family='monospace')

    # ---- LEFT: well-only panel (no outline; NaN cells -> white) ----
    ax_well = fig.add_axes([0.020, 0.10, 0.235, 0.34])
    _draw_panel(ax_well, well_masked, cmap=CMAP_WELL, norm=NORM_FACIES,
                 vector=vector)
    ax_well.set_xticks([]); ax_well.set_yticks([])
    for spine in ax_well.spines.values():
        spine.set_edgecolor('0.30'); spine.set_linewidth(0.8)
    ax_well.set_title('Well log', fontsize=10, pad=4)

    # ---- MIDDLE: ResFlow black box ----
    ax_box = fig.add_axes([0.305, 0.32, 0.135, 0.36])
    ax_box.set_axis_off()
    ax_box.add_patch(FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.90,
        boxstyle='round,pad=0.02,rounding_size=0.06',
        facecolor='black', edgecolor='black',
        transform=ax_box.transAxes))
    ax_box.text(0.5, 0.5, 'ResFlow', color='white', ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax_box.transAxes)

    # ---- RIGHT: full XZ output (with well outline) ----
    ax_out = fig.add_axes([0.485, 0.10, 0.500, 0.80])
    _draw_panel(ax_out, xz_binary, cmap=CMAP_FACIES, norm=NORM_FACIES,
                 vector=vector)
    ax_out.set_xticks([]); ax_out.set_yticks([])
    for spine in ax_out.spines.values():
        spine.set_edgecolor('0.20'); spine.set_linewidth(1.0)
    ax_out.set_title('Generated reservoir (XZ slice)',
                      fontsize=11, pad=6, fontweight='bold')

    # Black well outline on the output (1 voxel wide column at x=x_well).
    x_off = 0.0 if vector else -0.5
    y_off = 0.0 if vector else -0.5
    ax_out.add_patch(Rectangle(
        (x_well + x_off, y_off), 1, SZ, fill=False,
        edgecolor='black', linewidth=1.4, alpha=1.0, zorder=6))

    # ---- ARROWS in figure coordinates ----
    arrow_kw = dict(arrowstyle='-|>,head_length=10,head_width=6',
                    color='black', linewidth=1.6,
                    mutation_scale=1.0,
                    transform=fig.transFigure)
    fig.patches.append(FancyArrowPatch((0.255, 0.74), (0.305, 0.60),
                                        connectionstyle='arc3,rad=-0.12',
                                        **arrow_kw))
    fig.patches.append(FancyArrowPatch((0.255, 0.27), (0.305, 0.41),
                                        connectionstyle='arc3,rad=0.12',
                                        **arrow_kw))
    fig.patches.append(FancyArrowPatch((0.440, 0.50), (0.485, 0.50),
                                        connectionstyle='arc3,rad=0',
                                        **arrow_kw))

    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------- main ------------------------------------

def main():
    SCRATCH = os.environ.get('SCRATCH', '/tmp')
    DEFAULT_CKPT = os.path.join(
        SCRATCH, 'genflows_runs/reservoirs_inpainting/checkpoints/flow_matching.pt')
    DEFAULT_COND_STATS = os.path.join(
        SCRATCH, 'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz')
    DEFAULT_DATA_DIR = os.environ.get(
        'RESERVOIR_DATA_DIR',
        os.path.join(SCRATCH, 'SiliciclasticReservoirs'))

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--cond-stats', default=DEFAULT_COND_STATS)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--cube-cache', default=str(DEFAULT_CACHE),
                    help='cube .npz cache (default: figs/figure3_lobe_cube.npz)')
    ap.add_argument('--regenerate', action='store_true',
                    help='ignore cache and re-pick + re-sample')
    ap.add_argument('--ntg-min', type=float, default=DEFAULT_NTG_MIN,
                    help='minimum NTG for the picked lobe (default: 0.75)')
    ap.add_argument('--y', type=int, default=DEFAULT_Y_SLICE)
    ap.add_argument('--x-well', type=int, default=DEFAULT_X_WELL)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--out-dir', default=str(PRES_DIR / 'figs'))
    ap.add_argument('--out-name', default='figure3')
    ap.add_argument('--vector', action='store_true')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    cube, info = load_or_generate(
        Path(args.cube_cache), Path(args.data_dir),
        args.ckpt, args.cond_stats,
        ntg_min=args.ntg_min, x_well=args.x_well, y_slice=args.y,
        seed=args.seed, regenerate=args.regenerate)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name
    if args.vector and not out_name.endswith('_vector'):
        out_name = f'{out_name}_vector'
    out_png = out_dir / f'{out_name}.png'
    out_pdf = out_dir / f'{out_name}.pdf'
    print(f'Rendering -> {out_png}')
    render_schematic(cube, info, out_png, out_pdf,
                     vector=args.vector, dpi=args.dpi)
    print(f'Done. {out_png}  /  {out_pdf}')


if __name__ == '__main__':
    main()
