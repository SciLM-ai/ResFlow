"""Whole-field generation with a size-agnostic DiT (RoPE + windowed attention).

No tiling and no fusion rule: the model sees the entire reservoir at every
ODE step, so there is nothing to average, stage or outpaint. Requires a
DiT3D built with ``pos_embed='rope'`` (any token grid) and, for fields
much larger than the training crop, ``window`` (linear cost, and no
relative offset outside the trained range). Spatially varying conditions
are passed per token, so the paper's Figure-4 grid of block conditions is
one condition map.

The field extent follows the tiled schedulers' (grid_shape, overlap)
layout so results are directly comparable with 4-stage / MultiDiffusion
runs of the same layout; the condition of a token is that of the block
whose centre is nearest, i.e. the block lattice is honoured without any
overlap semantics.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def generate_wholefield(model, cond_vec, grid_shape=(10, 10),
                        block_shape=(64, 64, 32), overlap=12, n_steps=25,
                        cfg_scale=3.0, device='cuda', solver='heun',
                        generator=None, verbose=True, trajectory_path=None,
                        trajectory_every=1, field_shape=None, amp=True,
                        max_batched_tokens=400_000):
    """One model call per ODE step (two with CFG, batched) over the field.

    cond_vec: (COND_DIM,) or a (ny, nx, COND_DIM) grid ([i][j] = [y][x]).
    field_shape: (Tx, Ty[, Tz]) to override the layout-derived extent.
    trajectory_path: if set, the field is recorded every `trajectory_every`
    steps as fp16 ('traj' (S, Tx, Ty, Tz), 't' (S,)).

    max_batched_tokens: above this many tokens the conditional and
    unconditional passes run one after the other instead of as a batch of
    two, halving peak memory (a 30x30-block field is 1.2M tokens).

    Returns (x on CPU as (Tx, Ty, Tz), per-step elapsed seconds).
    """
    assert getattr(model, 'pos_type', 'learned') == 'rope', \
        'whole-field generation needs a RoPE DiT (any-size token grid)'
    ny, nx = grid_shape
    Sx, Sy, Sz = block_shape
    stride = Sx - overlap
    if field_shape is None:
        Tx, Ty, Tz = Sx + (nx - 1) * stride, Sy + (ny - 1) * stride, Sz
    else:
        Tx, Ty = field_shape[:2]
        Tz = field_shape[2] if len(field_shape) > 2 else Sz
    px, py, pz = model.patch_size
    # The requested extent need not be a multiple of the patch (e.g. the
    # overlap-12 layout gives 532 cells, which an 8-cell patch cannot tile).
    # Generate on the next multiple up and crop back: the model is
    # size-agnostic, so the padding costs a little compute and nothing else.
    Px, Py, Pz = (-(-Tx // px) * px, -(-Ty // py) * py, -(-Tz // pz) * pz)
    gx, gy, gz = Px // px, Py // py, Pz // pz
    N = gx * gy * gz

    cond = torch.as_tensor(np.asarray(cond_vec, dtype=np.float32), device=device)
    if cond.ndim == 1:
        cond_tok = cond.view(1, 1, -1).expand(1, N, -1).contiguous()
    else:
        assert tuple(cond.shape[:2]) == (ny, nx), (cond.shape, grid_shape)
        cx = torch.arange(gx, device=device) * px + px / 2      # token centres (voxels)
        cy = torch.arange(gy, device=device) * py + py / 2
        j = ((cx - Sx / 2) / stride).round().long().clamp(0, nx - 1)
        i = ((cy - Sy / 2) / stride).round().long().clamp(0, ny - 1)
        grid_c = cond[i[None, :], j[:, None]]                    # (gx, gy, C)
        cond_tok = grid_c[:, :, None, :].expand(gx, gy, gz, -1).reshape(1, N, -1).contiguous()

    x = torch.randn(1, 1, Px, Py, Pz, device=device, generator=generator)
    batched = cfg_scale > 0 and N <= max_batched_tokens
    zero = torch.zeros(2 if batched else 1, 1, Px, Py, Pz, device=device)
    model.set_inpaint_context(zero, zero)
    drop = torch.tensor([False, True], device=device)
    cond2 = cond_tok.repeat(2, 1, 1) if batched else None

    def vel(x_state, t_val):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            if batched:
                tt = torch.full((2,), t_val, device=device) * 1000
                v = model(x_state.repeat(2, 1, 1, 1, 1), tt, cond2, drop_mask=drop).float()
                return v[1:2] + cfg_scale * (v[0:1] - v[1:2])
            tt = torch.full((1,), t_val, device=device) * 1000
            if cfg_scale > 0:
                v_c = model(x_state, tt, cond_tok).float()
                v_u = model(x_state, tt).float()
                return v_u + cfg_scale * (v_c - v_u)
            return model(x_state, tt, cond_tok).float()

    dt = 1.0 / n_steps
    timings, traj, traj_t = [], [], []
    for step in range(n_steps):
        t0 = time.time()
        t_val = step * dt
        v = vel(x, t_val)
        if solver == 'heun':
            v = 0.5 * (v + vel(x + v * dt, t_val + dt))
        x = x + v * dt
        timings.append(time.time() - t0)
        if trajectory_path is not None and ((step + 1) % trajectory_every == 0
                                            or step == n_steps - 1):
            traj.append(x[0, 0, :Tx, :Ty, :Tz].half().cpu().numpy())
            traj_t.append((step + 1) * dt)
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f'    step {step:3d}/{n_steps}  {timings[-1]:.2f}s', flush=True)
    model.clear_inpaint_context()
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj),
                            t=np.array(traj_t, dtype=np.float32))
    return x[0, 0, :Tx, :Ty, :Tz].cpu(), timings
