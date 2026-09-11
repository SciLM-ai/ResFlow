"""Comparison figure for the best Addendum-G model.

Three questions the figure has to answer at a glance:
  1. what does a MultiDiffusion assembly look like before and after?
  2. is the body-size distribution actually closer to the engine?
  3. by how much, on the frozen benchmark metrics?

Facies maps are a two-level categorical field (sand / mud), so they use
ink-on-surface rather than a colour ramp -- the standard geological
convention and the honest encoding for binary data. The two MODELS are
the categorical series (validated blue/orange); the ENGINE is the
reference, so it is drawn as a muted dashed line, not a third hue.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, '/work/08405/ilgar/vista/codes/ResBench')
from resbench import metrics  # noqa: E402

O = Path('/work/08405/ilgar/vista/resbench_eval')
BENCH = Path('/work/08405/ilgar/vista/codes/ResBench')

# design tokens (validated: see dataviz/references/palette.md)
INK, SECOND, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, SURFACE = '#e1e0d9', '#fcfcfb'
PREV, NEW = '#2a78d6', '#eb6834'
SAND, MUD = '#2f2e2b', '#efeee8'
CMAP = matplotlib.colors.ListedColormap([MUD, SAND])


def load_assembly(d, i=0):
    return np.load(sorted(Path(d).glob('assembly_*.npz'))[i])['binary']


def load_native(d):
    f = sorted(Path(d).glob('lobe/volumes_*.npz')) or sorted(Path(d).glob('volumes_*.npz'))
    return np.load(f[0], allow_pickle=True)['volumes']


def geobody_sizes(vols, cap=None):
    out = []
    for v in (vols if cap is None else vols[:cap]):
        s = metrics.geobody_sizes(v)
        if len(s):
            out.append(s)
    return np.concatenate(out) if out else np.empty(0)


def tiles_from(assembly_dir, n_asm=10):
    """Same 5x5 centred 64-tile cut the scorer uses."""
    tiles = []
    for f in sorted(Path(assembly_dir).glob('assembly_*.npz'))[:n_asm]:
        b = np.load(f)['binary']
        o = (b.shape[0] - 5 * 64) // 2
        for i in range(5):
            for j in range(5):
                tiles.append(b[o + i * 64:o + i * 64 + 64,
                               o + j * 64:o + j * 64 + 64, :])
    return np.stack(tiles)


def main():
    prev_dir = O / 'assembly_lobe_specialist'          # published approach
    new_dir = O / 'ep_065'                              # best Addendum-G model
    eng = load_native(BENCH / 'results/assembly_reference')

    a_prev, a_new = load_assembly(prev_dir), load_assembly(new_dir)
    z = a_new.shape[2] // 2

    fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.22, 1.0],
                          hspace=0.30, wspace=0.20,
                          left=0.05, right=0.975, top=0.885, bottom=0.085)

    # --- row 1: plan views -------------------------------------------
    for col, (arr, name, sub, accent) in enumerate([
            (a_prev, 'Published approach', 'lobe specialist · geobody W1 0.259', PREV),
            (a_new, 'Best new model', 'crops192 + slabs, ep65 · geobody W1 0.070', NEW)]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr[:, :, z].T, cmap=CMAP, origin='lower', interpolation='nearest')
        ax.set_title(name, color=INK, fontsize=12.5, fontweight='600',
                     pad=13, loc='left')
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=SECOND,
                fontsize=9.3, va='bottom')
        for s in ax.spines.values():
            s.set_color(accent); s.set_linewidth(2.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'{arr.shape[0]}×{arr.shape[1]} cells, plan view at z={z}',
                      color=MUTED, fontsize=8.6)

    ax = fig.add_subplot(gs[0, 2])
    grid = np.zeros((64 * 3, 64 * 3), dtype=np.int8)
    for k in range(9):
        grid[(k // 3) * 64:(k // 3) * 64 + 64,
             (k % 3) * 64:(k % 3) * 64 + 64] = eng[k][:, :, z]
    ax.imshow(grid.T, cmap=CMAP, origin='lower', interpolation='nearest')
    for k in (64, 128):
        ax.axhline(k - .5, color=SURFACE, lw=1.6)
        ax.axvline(k - .5, color=SURFACE, lw=1.6)
    ax.set_title('Engine reference', color=INK, fontsize=12.5,
                 fontweight='600', pad=13, loc='left')
    ax.text(0, 1.012, 'ResMill ground truth · 9 of 256 volumes',
            transform=ax.transAxes, color=SECOND, fontsize=9.3, va='bottom')
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(2.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('nine 64×64 volumes, same z', color=MUTED, fontsize=8.6)

    # --- row 2 left: geobody size CDF --------------------------------
    ax = fig.add_subplot(gs[1, :2])
    series = [('Engine reference', geobody_sizes(eng), MUTED, '--', 2.0),
              ('Published approach', geobody_sizes(tiles_from(prev_dir)), PREV, '-', 2.0),
              ('Best new model', geobody_sizes(tiles_from(new_dir)), NEW, '-', 2.0)]
    for label, s, c, ls, lw in series:
        s = np.sort(s[s > 0])
        ax.plot(np.log10(s), np.arange(1, len(s) + 1) / len(s),
                color=c, ls=ls, lw=lw, label=label, solid_capstyle='round')
    ax.set_xlabel('geobody size  (log₁₀ voxels)', color=SECOND, fontsize=10)
    ax.set_ylabel('cumulative fraction', color=SECOND, fontsize=10)
    ax.set_title('Body-size distribution — the metric that was failing',
                 color=INK, fontsize=12, fontweight='600', pad=10, loc='left')
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9.6, loc='lower right')
    for t in leg.get_texts():
        t.set_color(SECOND)

    # --- row 2 right: metric bars ------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    names = ['geobody\nW1', 'extent\nW1', '|ΔNTG|', 'connect.\nMAE']
    prev_v = [0.2592, 0.0981, 0.0175, 0.0175]
    new_v = [0.0701, 0.0222, 0.0041, 0.0126]
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.19, prev_v, height=0.34, color=PREV, label='Published')
    ax.barh(y - 0.19, new_v, height=0.34, color=NEW, label='New model')
    for yy, pv, nv in zip(y, prev_v, new_v):
        ax.text(pv + .006, yy + 0.19, f'{pv:.3f}', va='center',
                color=SECOND, fontsize=8.4)
        ax.text(nv + .006, yy - 0.19, f'{nv:.3f}', va='center',
                color=SECOND, fontsize=8.4, fontweight='600')
    ax.set_yticks(y); ax.set_yticklabels(names, color=SECOND, fontsize=9)
    ax.set_xlabel('distance to engine  (lower is better)', color=SECOND,
                  fontsize=9.4)
    ax.set_title('Assembly metrics', color=INK, fontsize=12,
                 fontweight='600', pad=10, loc='left')
    ax.grid(True, axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(0, max(prev_v) * 1.30)
    leg = ax.legend(frameon=False, fontsize=9.2, loc='lower right')
    for t in leg.get_texts():
        t.set_color(SECOND)

    fig.suptitle('MultiDiffusion assembly of 3D lobe reservoirs — '
                 'body-scale fidelity', color=INK, fontsize=15,
                 fontweight='700', x=0.05, ha='left', y=0.965)
    fig.text(0.05, 0.927,
             'Single seed. Assembly gain replicates across ~10 runs; the '
             'per-seed numbers do not — identical recipes span 0.070–0.166.',
             color=MUTED, fontsize=9.4, ha='left')

    out = Path('/work/08405/ilgar/vista/resbench_eval/best_model_figure.png')
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print('wrote', out)


if __name__ == '__main__':
    main()
