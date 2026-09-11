"""Sequential outpainting assembly — the alternative to velocity averaging.

MultiDiffusion (``big_reservoir_multi.py``) fuses overlapping blocks by
averaging their predicted velocities. Measured on the lobe tier of
ResBench Addendum E, that triples geobody-size W1 relative to native
generation, and the effect is the same magnitude for the foundation
model and for an 80-epoch lobe specialist — it is a property of the
fusion rule, not the weights. The mechanism is that averaging two fields
which disagree about a boundary's position yields a low-amplitude smear
at both, which binarization resolves into one merged body. At the
deployed block 64 / overlap 24, 84% of an assembly is a blend of two or
more blocks.

Here each block is instead *conditioned* on its already-generated
neighbours: the overlap region enters through the model's existing
3-channel inpainting interface as known data, and the block is sampled
normally. Nothing is averaged, so nothing is smeared.

Ordering: block (i, j) depends only on (i, j-1) and (i-1, j), so all
blocks on an anti-diagonal i + j = k are mutually independent and are
sampled as one batch. A 10x10 grid needs 19 sequential sampling passes
instead of 100, with batches of up to 10.

Requires a model trained with slab-shaped masks (see
``resflow.utils.masking_context``); the wells-only foundation and
specialist checkpoints have never seen a 24-cell-wide known region and
will be out of distribution here.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from resflow.utils.masking_context import (
    build_assembly_context_mask, known_sides_raster)


@torch.no_grad()
def _euler_cfg_block(model, x0, cond, mask, known, cfg_scale, n_steps,
                     solver='euler'):
    """Euler (or Heun) + CFG sampling for a batch of blocks with inpaint
    context.

    'euler' mirrors the Table 6 sampler (Euler, NFE 50, CFG 3.0) used
    everywhere else in the benchmark; the only addition is that the
    inpaint context is set once for the whole trajectory. 'heun' is the
    second-order predictor-corrector at 2 velocity evaluations per step
    (so Heun with n_steps/2 costs the same as Euler with n_steps); DiT3D
    velocity fields are more curved than UNet3D's and need it.
    """
    x = x0
    dt = 1.0 / n_steps
    model.set_inpaint_context(mask, known)

    def vel(x, t_val):
        t_emb = torch.full((x.shape[0],), t_val, device=x.device) * 1000
        if cfg_scale > 0:
            v_cond = model(x, t_emb, cond)
            v_uncond = model(x, t_emb)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        return model(x, t_emb, cond)

    try:
        for step in range(n_steps):
            t = step * dt
            v = vel(x, t)
            if solver == 'heun':
                v2 = vel(x + v * dt, t + dt)
                v = 0.5 * (v + v2)
            x = x + v * dt
    finally:
        model.clear_inpaint_context()
    return x


@torch.no_grad()
def generate_big_reservoir_outpaint(
    model,
    cond_vec,
    grid_shape=(10, 10),
    block_shape=(64, 64, 32),
    overlap=24,
    n_steps=50,
    cfg_scale=3.0,
    device='cuda',
    binarize_each_block=True,
    generator=None,
    verbose=True,
    solver='euler',
    trajectory_path=None,
):
    """Raster-order outpainting assembly.

    trajectory_path: if set, the partially assembled field is recorded after
    every wavefront (fp16) into this .npz ('traj', 'group').

    Args:
        model: UNet3D with in_channels=3, trained with context slabs.
        cond_vec: (COND_DIM,) numpy array — one condition for all blocks —
            or (ny, nx, COND_DIM) for a per-block condition grid (same
            [i][j] = [y][x] convention as the MultiDiffusion spec grid).
        binarize_each_block: binarize a block to {-1, +1} before it is
            used as context for its neighbours. This matches deployment,
            where the assembled volume is binary, and prevents
            continuous-valued drift from accumulating across the grid.

    Returns:
        (x_global, elapsed_by_diagonal) — float32 (Tx, Ty, Sz) on CPU.
    """
    ny, nx = grid_shape
    Sx, Sy, Sz = block_shape
    stride = Sx - overlap
    Tx = Sx + (nx - 1) * stride
    Ty = Sy + (ny - 1) * stride

    x_global = torch.zeros(Tx, Ty, Sz, device=device)
    filled = torch.zeros(Tx, Ty, Sz, device=device)  # 1 where already written
    cond = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32)).to(device)
    per_block = cond.ndim == 3
    if per_block:
        assert tuple(cond.shape[:2]) == (ny, nx), (cond.shape, grid_shape)

    timings = []
    traj = []
    for k in range(ny + nx - 1):
        diag = [(i, k - i) for i in range(ny)
                if 0 <= k - i < nx]
        if not diag:
            continue
        t0 = time.time()
        bs = len(diag)

        masks = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
        knowns = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
        for b, (i, j) in enumerate(diag):
            xs, ys = j * stride, i * stride
            # Raster order: left (low x, j > 0) and top (low y, i > 0) are
            # already final. Same convention as schedulers.generate_staged.
            m = build_assembly_context_mask(
                block_shape, known_sides_raster(i, j), overlap).to(device)
            # Only count as known what has actually been written. On the
            # grid edges the geometric mask can reach outside the filled
            # region; multiplying by `filled` keeps the two consistent.
            m = m * filled[xs:xs + Sx, ys:ys + Sy, :].unsqueeze(0)
            masks[b] = m
            knowns[b] = m * x_global[xs:xs + Sx, ys:ys + Sy, :].unsqueeze(0)

        x0 = torch.randn(bs, 1, Sx, Sy, Sz, device=device, generator=generator)
        if per_block:
            cond_b = torch.stack([cond[i, j] for (i, j) in diag]).contiguous()
        else:
            cond_b = cond.unsqueeze(0).expand(bs, -1).contiguous()
        out = _euler_cfg_block(model, x0, cond_b, masks, knowns,
                               cfg_scale, n_steps, solver=solver)
        # Hard-replace the known voxels, exactly as single-block
        # inpainting does at its final step.
        out = out * (1 - masks) + knowns * masks
        if binarize_each_block:
            out = (out > 0).float() * 2 - 1

        for b, (i, j) in enumerate(diag):
            xs, ys = j * stride, i * stride
            x_global[xs:xs + Sx, ys:ys + Sy, :] = out[b, 0]
            filled[xs:xs + Sx, ys:ys + Sy, :] = 1.0
        if trajectory_path is not None:
            traj.append(x_global.half().cpu().numpy())

        dt_ = time.time() - t0
        timings.append(dt_)
        if verbose:
            print(f"    diagonal {k:2d}/{ny + nx - 2}  {bs:2d} blocks  "
                  f"{dt_:.1f}s", flush=True)

    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj),
                            group=np.arange(len(traj), dtype=np.int32))
    return x_global.cpu(), timings
