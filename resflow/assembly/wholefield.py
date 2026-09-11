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
                        trajectory_every=1, field_shape=None, amp=True):
    """One model call per ODE step (two with CFG, batched) over the field.

    cond_vec: (COND_DIM,) or a (ny, nx, COND_DIM) grid ([i][j] = [y][x]).
    field_shape: (Tx, Ty[, Tz]) to override the layout-derived extent.
    trajectory_path: if set, the field is recorded every `trajectory_every`
    steps as fp16 ('traj' (S, Tx, Ty, Tz), 't' (S,)).

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
    assert Tx % px == 0 and Ty % py == 0 and Tz % pz == 0, (Tx, Ty, Tz, model.patch_size)
    gx, gy, gz = Tx // px, Ty // py, Tz // pz
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

    x = torch.randn(1, 1, Tx, Ty, Tz, device=device, generator=generator)
    zero = torch.zeros(2 if cfg_scale > 0 else 1, 1, Tx, Ty, Tz, device=device)
    model.set_inpaint_context(zero, zero)
    drop = torch.tensor([False, True], device=device)
    cond2 = cond_tok.repeat(2, 1, 1)

    def vel(x_state, t_val):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            if cfg_scale > 0:
                tt = torch.full((2,), t_val, device=device) * 1000
                v = model(x_state.repeat(2, 1, 1, 1, 1), tt, cond2, drop_mask=drop).float()
                return v[1:2] + cfg_scale * (v[0:1] - v[1:2])
            tt = torch.full((1,), t_val, device=device) * 1000
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
            traj.append(x[0, 0].half().cpu().numpy()); traj_t.append((step + 1) * dt)
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f'    step {step:3d}/{n_steps}  {timings[-1]:.2f}s', flush=True)
    model.clear_inpaint_context()
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj),
                            t=np.array(traj_t, dtype=np.float32))
    return x[0, 0].cpu(), timings
