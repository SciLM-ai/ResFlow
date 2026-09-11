"""Block schedulers for tiled assembly beyond the training extent.

Four ways to combine overlapping blocks into one reservoir, sharing a
single model so that the *scheduler* is the only variable when they are
compared.

  multi     -- MultiDiffusion: one shared latent, velocities AVERAGED in
               overlaps. Fully parallel (50 sequential steps) but the
               average of two velocity fields is not the velocity field
               of any distribution, so disagreements become smears that
               binarization resolves into merged bodies.
  raster    -- Outpainting in raster order. Each block is an exact
               conditional sample given its finished left/top neighbours,
               so nothing is averaged. Exact, but the anti-diagonal
               wavefront costs (2n-1) x 50 sequential steps -- O(n).
  stage4    -- 4-colour staging. Blocks are coloured by (row, col)
               parity; same-colour blocks sit 2 strides apart, hence are
               disjoint whenever overlap <= block/2, hence can be
               generated together. Four stages regardless of grid size:
               O(1) depth. The approximation is that stage-1 blocks are
               generated independently of one another.
  coupled   -- Trajectory conditioning. Every block advances together,
               each conditioned on its neighbours' CURRENT partially
               denoised state. For the joint z = (x_A, x_B) the flow
               velocity is E[z1 - z0 | z_t], whose A-component is exactly
               "my velocity given my noisy state and my neighbour's noisy
               state" -- so this is the exact joint flow AND fully
               parallel. Requires a model trained with a context noise
               level (in_channels == 4).

`stage4` and `coupled` need a model trained on the corresponding
neighbour configurations; see resflow.utils.masking_context.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from resflow.utils.masking_context import (
    build_assembly_context_mask, known_sides_4stage, known_sides_raster,
    stage_of,
)


def _layout(grid_shape, block_shape, overlap):
    ny, nx = grid_shape
    Sx, Sy, Sz = block_shape
    stride = Sx - overlap
    return ny, nx, Sx, Sy, Sz, stride, Sx + (nx - 1) * stride, Sy + (ny - 1) * stride


def global_noise(total_shape, device, generator=None):
    """One noise field for the whole reservoir.

    Flow matching is a deterministic map from x0 to x1 given the
    conditioning, so blocks whose initial noise is spatially correlated
    produce correlated geology -- coherence for free, with no
    communication. MultiDiffusion gets this implicitly from its shared
    latent; the sequential samplers must do it explicitly or they throw
    it away by drawing fresh noise per block.
    """
    return torch.randn(*total_shape, device=device, generator=generator)


@torch.no_grad()
def _euler_blocks(model, x, cond, mask, ctx, ctx_level, cfg_scale, n_steps,
                  t_start=0.0, x_is_state=True, solver='euler'):
    """Euler (or Heun, 2 evaluations/step) + CFG over a batch of blocks
    with a fixed inpaint context."""
    dt = 1.0 / n_steps
    n_from = int(round(t_start * n_steps))
    model.set_inpaint_context(mask, ctx, ctx_level=ctx_level)

    def vel(x, t_val):
        temb = torch.full((x.shape[0],), t_val, device=x.device) * 1000
        if cfg_scale > 0:
            v = model(x, temb, cond)
            vu = model(x, temb)
            return vu + cfg_scale * (v - vu)
        return model(x, temb, cond)

    try:
        for step in range(n_from, n_steps):
            t = step * dt
            v = vel(x, t)
            if solver == 'heun':
                v = 0.5 * (v + vel(x + v * dt, t + dt))
            x = x + v * dt
    finally:
        model.clear_inpaint_context()
    return x


@torch.no_grad()
def _multidiffusion_warmup(model, cond_vec, grid_shape, block_shape, overlap,
                           n_steps, md_frac, cfg_scale, device, noise,
                           max_batch=24):
    """Integrate MultiDiffusion on a shared latent over t in [0, md_frac].

    Averaging velocities is destructive once boundaries are sharp, but
    early in the trajectory the field is still low-frequency and the
    average costs little -- and it is the only mechanism that lets
    disjoint same-stage blocks learn about each other. So it is used to
    fix the global layout only, and conditioning takes over for detail.
    """
    ny, nx, Sx, Sy, Sz, stride, Tx, Ty = _layout(grid_shape, block_shape,
                                                 overlap)
    cells = [(i, j) for i in range(ny) for j in range(nx)]
    x = noise.clone() if noise is not None else torch.randn(Tx, Ty, Sz,
                                                            device=device)
    cond = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32)).to(device)
    per_block = cond.ndim == 3           # (ny, nx, COND_DIM) grid of conditions

    # Blend weights: linear ramps on overlapping faces summing to 1.
    w_blocks = {}
    ramp_up = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
    ramp_dn = torch.linspace(1, 0, overlap + 2, device=device)[1:-1]
    for i, j in cells:
        w = torch.ones(Sx, Sy, Sz, device=device)
        if j > 0:
            w[:overlap] *= ramp_up.view(-1, 1, 1)
        if j < nx - 1:
            w[-overlap:] *= ramp_dn.view(-1, 1, 1)
        if i > 0:
            w[:, :overlap] *= ramp_up.view(1, -1, 1)
        if i < ny - 1:
            w[:, -overlap:] *= ramp_dn.view(1, -1, 1)
        w_blocks[(i, j)] = w

    dt = 1.0 / n_steps
    n_warm = int(round(md_frac * n_steps))
    zero = torch.zeros(max_batch, 1, Sx, Sy, Sz, device=device)
    for step in range(n_warm):
        t_val = step * dt
        v_acc = torch.zeros_like(x)
        w_acc = torch.zeros_like(x)
        for lo in range(0, len(cells), max_batch):
            batch = cells[lo:lo + max_batch]
            bs = len(batch)
            blk = torch.empty(bs, 1, Sx, Sy, Sz, device=device)
            for b, (i, j) in enumerate(batch):
                xs, ys = j * stride, i * stride
                blk[b, 0] = x[xs:xs + Sx, ys:ys + Sy, :]
            model.set_inpaint_context(zero[:bs], zero[:bs])
            tt = torch.full((bs,), t_val, device=device) * 1000
            if per_block:
                cb = torch.stack([cond[i, j] for (i, j) in batch]).contiguous()
            else:
                cb = cond.unsqueeze(0).expand(bs, -1).contiguous()
            if cfg_scale > 0:
                v = model(blk, tt, cb)
                vu = model(blk, tt)
                v = vu + cfg_scale * (v - vu)
            else:
                v = model(blk, tt, cb)
            model.clear_inpaint_context()
            for b, (i, j) in enumerate(batch):
                xs, ys = j * stride, i * stride
                w = w_blocks[(i, j)]
                v_acc[xs:xs + Sx, ys:ys + Sy, :] += w * v[b, 0]
                w_acc[xs:xs + Sx, ys:ys + Sy, :] += w
        x = x + (v_acc / w_acc.clamp(min=1e-8)) * dt
    return x


@torch.no_grad()
def generate_staged(model, cond_vec, grid_shape=(10, 10),
                    block_shape=(64, 64, 32), overlap=24, n_steps=50,
                    cfg_scale=3.0, device='cuda', order='stage4',
                    shared_noise=True, md_frac=0.0, method=None,
                    grid_specs=None, cont_min=None, cont_max=None,
                    generator=None, verbose=True, solver='euler',
                    trajectory_path=None):
    """Sequential conditioning: 'stage4' (4 colours) or 'raster'.

    trajectory_path: if set, the partially assembled field is recorded after
    every group (stage or wavefront) as fp16 into this .npz ('traj', 'group').

    Returns (x_global on CPU, per-group elapsed seconds).
    """
    ny, nx, Sx, Sy, Sz, stride, Tx, Ty = _layout(grid_shape, block_shape,
                                                 overlap)
    if order == 'stage4' and overlap > min(Sx, Sy) // 2:
        raise ValueError(
            f'overlap {overlap} > block/2; same-stage blocks would overlap '
            f'and a 4-colour schedule is invalid (needs 9 colours)')

    x_global = torch.zeros(Tx, Ty, Sz, device=device)
    filled = torch.zeros(Tx, Ty, Sz, device=device)
    noise = (global_noise((Tx, Ty, Sz), device, generator) if shared_noise
             else None)

    # Optional MultiDiffusion warm-up. Averaging is destructive where
    # boundaries are sharp, but early in the trajectory structure is still
    # low-frequency and the average is nearly harmless -- while it is the
    # only mechanism that lets same-stage (disjoint) blocks learn about
    # each other at all. So run it for the first md_frac of the schedule
    # to fix the global layout, then let conditioning do the detail.
    t_start = 0.0
    if md_frac > 0:
        # A genuine PARTIAL trajectory: same dt as the main schedule, but
        # stopped at t = md_frac. (Reusing generate_big_reservoir_multi
        # with a reduced step count does NOT do this -- it integrates the
        # full 0 -> 1 range coarsely and hands back an almost-finished
        # volume, which is then inconsistent with the t it is labelled
        # with.)
        noise = _multidiffusion_warmup(
            model, cond_vec, grid_shape, block_shape, overlap, n_steps,
            md_frac, cfg_scale, device, noise)
        t_start = md_frac
    cond = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32)).to(device)
    # (COND_DIM,) = one condition for all blocks; (ny, nx, COND_DIM) = per-block
    # grid, same [i][j] = [y][x] convention as the MultiDiffusion spec grid.
    per_block = cond.ndim == 3
    if per_block:
        assert tuple(cond.shape[:2]) == (ny, nx), (cond.shape, grid_shape)

    if order == 'stage4':
        groups = [[(i, j) for i in range(ny) for j in range(nx)
                   if stage_of(i, j) == st] for st in (1, 2, 3, 4)]
    else:                                   # raster anti-diagonal wavefront
        groups = [[(i, k - i) for i in range(ny) if 0 <= k - i < nx]
                  for k in range(ny + nx - 1)]

    timings = []
    traj = []
    for gi, cells in enumerate(groups):
        if not cells:
            continue
        t0 = time.time()
        bs = len(cells)
        masks = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
        ctxs = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
        x0 = torch.empty(bs, 1, Sx, Sy, Sz, device=device)
        for b, (i, j) in enumerate(cells):
            xs, ys = j * stride, i * stride
            sides = (known_sides_4stage(i, j, ny, nx) if order == 'stage4'
                     else known_sides_raster(i, j))
            m = build_assembly_context_mask(block_shape, sides, overlap).to(device)
            # Only trust what has actually been written; on grid edges the
            # geometric mask can reach outside the filled region.
            m = m * filled[xs:xs + Sx, ys:ys + Sy, :].unsqueeze(0)
            masks[b] = m
            ctxs[b] = m * x_global[xs:xs + Sx, ys:ys + Sy, :].unsqueeze(0)
            x0[b, 0] = (noise[xs:xs + Sx, ys:ys + Sy, :] if shared_noise
                        else torch.randn(Sx, Sy, Sz, device=device,
                                         generator=generator))
        if per_block:
            cond_b = torch.stack([cond[i, j] for (i, j) in cells]).contiguous()
        else:
            cond_b = cond.unsqueeze(0).expand(bs, -1).contiguous()
        lvl = (torch.ones(bs, device=device)
               if getattr(model, 'in_channels', 3) >= 4 else None)
        out = _euler_blocks(model, x0, cond_b, masks, ctxs, lvl, cfg_scale,
                            n_steps, t_start=t_start, solver=solver)
        out = out * (1 - masks) + ctxs * masks
        out = (out > 0).float() * 2 - 1
        for b, (i, j) in enumerate(cells):
            xs, ys = j * stride, i * stride
            x_global[xs:xs + Sx, ys:ys + Sy, :] = out[b, 0]
            filled[xs:xs + Sx, ys:ys + Sy, :] = 1.0
        if trajectory_path is not None:
            traj.append(x_global.half().cpu().numpy())
        timings.append(time.time() - t0)
        if verbose:
            print(f'    group {gi + 1}/{len(groups)}  {bs:3d} blocks  '
                  f'{timings[-1]:.1f}s', flush=True)
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj),
                            group=np.arange(len(traj), dtype=np.int32))
    return x_global.cpu(), timings


@torch.no_grad()
def generate_coupled(model, cond_vec, grid_shape=(10, 10),
                     block_shape=(64, 64, 32), overlap=24, n_steps=50,
                     cfg_scale=3.0, device='cuda', max_batch=25,
                     generator=None, verbose=True, solver='euler',
                     trajectory_path=None, trajectory_every=5):
    """Trajectory conditioning: every block advances together.

    trajectory_path: if set, all block states are recorded every
    `trajectory_every` steps (fp16, 'traj' (S, nb, Sx, Sy, Sz), 't' (S,)).

    Each block keeps its OWN state. At every step a block is conditioned
    on its neighbours' current states (noise level s = t), which is the
    exact joint flow-matching velocity restricted to that block -- no
    averaging anywhere. Requires in_channels >= 4.

    Overlapping blocks hold independent copies of shared voxels, so the
    final volume takes each voxel from the block whose centre is nearest,
    a Voronoi assignment that never blends two values.
    """
    if getattr(model, 'in_channels', 3) < 4:
        raise ValueError('coupled sampling needs a model trained with a '
                         'context noise level (in_channels == 4)')
    ny, nx, Sx, Sy, Sz, stride, Tx, Ty = _layout(grid_shape, block_shape,
                                                 overlap)
    cells = [(i, j) for i in range(ny) for j in range(nx)]
    nb = len(cells)
    idx_of = {c: k for k, c in enumerate(cells)}

    noise = global_noise((Tx, Ty, Sz), device, generator)
    state = torch.empty(nb, 1, Sx, Sy, Sz, device=device)
    for k, (i, j) in enumerate(cells):
        xs, ys = j * stride, i * stride
        state[k, 0] = noise[xs:xs + Sx, ys:ys + Sy, :]

    cond = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32)).to(device)
    if cond.ndim == 3:                       # per-block (ny, nx, COND_DIM) grid
        cond_all = torch.stack([cond[i, j] for (i, j) in cells]).contiguous()
    else:
        cond_all = cond.unsqueeze(0).expand(nb, -1).contiguous()
    # Every interior block knows all four neighbours; this mask is fixed
    # across steps, only the values behind it change.
    masks = torch.zeros(nb, 1, Sx, Sy, Sz, device=device)
    for k, (i, j) in enumerate(cells):
        sides = set()
        if j > 0:
            sides.add('left')
        if j < nx - 1:
            sides.add('right')
        if i > 0:
            sides.add('top')
        if i < ny - 1:
            sides.add('bottom')
        masks[k] = build_assembly_context_mask(block_shape, sides,
                                               overlap).to(device)

    dt = 1.0 / n_steps
    timings = []

    def coupled_velocity(state, t_val):
        """Velocity of every block given its neighbours' CURRENT states."""
        ctx = torch.zeros_like(state)
        for k, (i, j) in enumerate(cells):
            for di, dj, side in ((0, -1, 'left'), (0, 1, 'right'),
                                 (-1, 0, 'top'), (1, 0, 'bottom')):
                nb_cell = (i + di, j + dj)
                if nb_cell not in idx_of:
                    continue
                nk = idx_of[nb_cell]
                # Region of block k that the neighbour also covers, and
                # the same region expressed in the neighbour's frame.
                if side == 'left':
                    sl_k = (slice(0, overlap), slice(None))
                    sl_n = (slice(Sx - overlap, Sx), slice(None))
                elif side == 'right':
                    sl_k = (slice(Sx - overlap, Sx), slice(None))
                    sl_n = (slice(0, overlap), slice(None))
                elif side == 'top':
                    sl_k = (slice(None), slice(0, overlap))
                    sl_n = (slice(None), slice(Sy - overlap, Sy))
                else:
                    sl_k = (slice(None), slice(Sy - overlap, Sy))
                    sl_n = (slice(None), slice(0, overlap))
                ctx[k, 0][sl_k] = state[nk, 0][sl_n]

        lvl = torch.full((nb,), t_val, device=device)
        vel = torch.empty_like(state)
        for lo in range(0, nb, max_batch):
            hi = min(lo + max_batch, nb)
            xb = state[lo:hi]
            model.set_inpaint_context(masks[lo:hi], ctx[lo:hi] * masks[lo:hi],
                                      ctx_level=lvl[lo:hi])
            tt = torch.full((hi - lo,), t_val, device=device) * 1000
            cb = cond_all[lo:hi]
            if cfg_scale > 0:
                v = model(xb, tt, cb)
                vu = model(xb, tt)
                v = vu + cfg_scale * (v - vu)
            else:
                v = model(xb, tt, cb)
            model.clear_inpaint_context()
            vel[lo:hi] = v
        return vel

    traj, traj_t = [], []
    for step in range(n_steps):
        t0 = time.time()
        t_val = step * dt
        v = coupled_velocity(state, t_val)
        if solver == 'heun':
            v = 0.5 * (v + coupled_velocity(state + v * dt, t_val + dt))
        state = state + v * dt
        if trajectory_path is not None and ((step + 1) % trajectory_every == 0
                                            or step == n_steps - 1):
            traj.append(state[:, 0].half().cpu().numpy()); traj_t.append((step + 1) * dt)
        timings.append(time.time() - t0)
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f'    step {step:3d}/{n_steps}  {timings[-1]:.2f}s',
                  flush=True)

    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj), t=np.array(traj_t, dtype=np.float32))
    # Voronoi assembly: nearest block centre wins, so no voxel is ever an
    # average of two disagreeing blocks.
    x_global = torch.zeros(Tx, Ty, Sz, device=device)
    best = torch.full((Tx, Ty), float('inf'), device=device)
    gx = torch.arange(Tx, device=device).view(-1, 1).float()
    gy = torch.arange(Ty, device=device).view(1, -1).float()
    for k, (i, j) in enumerate(cells):
        xs, ys = j * stride, i * stride
        cx, cy = xs + Sx / 2.0, ys + Sy / 2.0
        d = torch.full((Tx, Ty), float('inf'), device=device)
        d[xs:xs + Sx, ys:ys + Sy] = ((gx[xs:xs + Sx] - cx) ** 2
                                     + (gy[:, ys:ys + Sy] - cy) ** 2)
        take = d < best
        best = torch.where(take, d, best)
        blk = torch.zeros(Tx, Ty, Sz, device=device)
        blk[xs:xs + Sx, ys:ys + Sy, :] = state[k, 0]
        x_global = torch.where(take.unsqueeze(-1), blk, x_global)
    return x_global.cpu(), timings


def _halton(k, base):
    """k-th Halton number in the given base, in [0, 1)."""
    f, r = 1.0, 0.0
    while k > 0:
        f /= base
        r += f * (k % base)
        k //= base
    return r


def _spot_intervals(off, S, T):
    """Window origins along one axis for a lattice of stride S displaced by
    ``off`` (clamped into the reservoir) and the disjoint voxel interval
    each window owns: the midpoints between consecutive window centres, so
    every voxel goes to the window whose centre is nearest and nothing is
    ever blended."""
    origins = sorted({min(max(v, 0), T - S) for v in range(off - S, T, S)})
    centres = [o + S / 2 for o in origins]
    bounds = [0] + [int(math.ceil(0.5 * (a + b))) for a, b in
                    zip(centres[:-1], centres[1:])] + [T]
    return [(o, bounds[k], bounds[k + 1]) for k, o in enumerate(origins)]


@torch.no_grad()
def generate_spot(model, cond_vec, grid_shape=(10, 10),
                  block_shape=(64, 64, 32), overlap=24, n_steps=50,
                  cfg_scale=3.0, device='cuda', max_batch=24,
                  generator=None, verbose=True, solver='euler',
                  trajectory_path=None, trajectory_every=1,
                  shifts='halton'):
    """Shifted non-overlapping windows (SpotDiffusion-style), no averaging.

    The reservoir has the extent of the other schedulers' (grid_shape,
    overlap) layout so results are comparable, but the model never runs
    on that lattice. At every ODE step the field is partitioned into
    disjoint block-size windows whose grid is displaced by a fresh 2D
    offset in [0, block)^2; every window's velocity is computed
    independently (batched, fully parallel, no exchange within a step)
    and written back to its own voxels only. The seams of one step lie in
    the interior of the next step's windows, which is what repairs them.
    Windows that would leave the reservoir are clamped to its edge, and a
    voxel covered twice belongs to the window whose centre is nearest.
    Cost per step is one model evaluation per voxel (MultiDiffusion at
    overlap 24 spends 2.6), with the same 3 channels as MultiDiffusion.

    shifts: 'halton' (low-discrepancy 2D sequence, deterministic) or
    'random' (uniform, as in SpotDiffusion). Heun's two evaluations of a
    step share the step's partition.

    cond_vec: (COND_DIM,) for one condition, or a (ny, nx, COND_DIM) grid
    on the block lattice ([i][j] = [y][x]); a window takes the condition
    of the lattice block whose centre is nearest its own centre.

    trajectory_path: if set, the field is recorded every
    `trajectory_every` steps (fp16, 'traj' (S, Tx, Ty, Sz), 't' (S,),
    'offsets' (n_steps, 2)).

    Returns (x on CPU, per-step elapsed seconds).
    """
    ny, nx, Sx, Sy, Sz, stride, Tx, Ty = _layout(grid_shape, block_shape,
                                                 overlap)
    cond = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32)).to(device)
    per_block = cond.ndim == 3
    if per_block:
        assert tuple(cond.shape[:2]) == (ny, nx), (cond.shape, grid_shape)

    x = global_noise((Tx, Ty, Sz), device, generator)
    dt = 1.0 / n_steps
    if shifts == 'halton':
        offsets = [(int(_halton(k + 1, 2) * Sx), int(_halton(k + 1, 3) * Sy))
                   for k in range(n_steps)]
    elif shifts == 'random':
        offsets = [(int(torch.randint(0, Sx, (1,), generator=generator,
                                      device='cpu')),
                    int(torch.randint(0, Sy, (1,), generator=generator,
                                      device='cpu')))
                   for _ in range(n_steps)]
    else:
        raise ValueError(f'unknown shifts={shifts!r}')
    zero = torch.zeros(max_batch, 1, Sx, Sy, Sz, device=device)

    def cond_of(ox, oy):
        if not per_block:
            return cond
        j = min(max(int(round(ox / stride)), 0), nx - 1)
        i = min(max(int(round(oy / stride)), 0), ny - 1)
        return cond[i, j]

    def velocity(x_state, t_val, off):
        wins = [(wx, wy) for wx in _spot_intervals(off[0], Sx, Tx)
                for wy in _spot_intervals(off[1], Sy, Ty)]
        v_field = torch.empty_like(x_state)
        for lo in range(0, len(wins), max_batch):
            batch = wins[lo:lo + max_batch]
            bs = len(batch)
            blk = torch.empty(bs, 1, Sx, Sy, Sz, device=device)
            for b, ((ox, _, _), (oy, _, _)) in enumerate(batch):
                blk[b, 0] = x_state[ox:ox + Sx, oy:oy + Sy, :]
            model.set_inpaint_context(zero[:bs], zero[:bs])
            tt = torch.full((bs,), t_val, device=device) * 1000
            cb = torch.stack([cond_of(ox, oy)
                              for (ox, _, _), (oy, _, _) in batch]).contiguous()
            if cfg_scale > 0:
                v = model(blk, tt, cb)
                vu = model(blk, tt)
                v = vu + cfg_scale * (v - vu)
            else:
                v = model(blk, tt, cb)
            model.clear_inpaint_context()
            for b, ((ox, xa, xb), (oy, ya, yb)) in enumerate(batch):
                v_field[xa:xb, ya:yb, :] = v[b, 0, xa - ox:xb - ox,
                                             ya - oy:yb - oy, :]
        return v_field, len(wins)

    timings, traj, traj_t = [], [], []
    for step in range(n_steps):
        t0 = time.time()
        t_val = step * dt
        v, nwin = velocity(x, t_val, offsets[step])
        if solver == 'heun':
            v2, _ = velocity(x + v * dt, t_val + dt, offsets[step])
            v = 0.5 * (v + v2)
        x = x + v * dt
        timings.append(time.time() - t0)
        if trajectory_path is not None and ((step + 1) % trajectory_every == 0
                                            or step == n_steps - 1):
            traj.append(x.half().cpu().numpy())
            traj_t.append((step + 1) * dt)
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f'    step {step:3d}/{n_steps}  offset={offsets[step]}  '
                  f'{nwin} windows  {timings[-1]:.2f}s', flush=True)
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj),
                            t=np.array(traj_t, dtype=np.float32),
                            offsets=np.array(offsets, dtype=np.int32))
    return x.cpu(), timings
