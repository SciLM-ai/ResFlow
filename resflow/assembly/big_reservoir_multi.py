"""Multi-type big-reservoir generation via parallel denoising.

Generalizes the single-type lobe assembly in ``reservoir.py`` to the
SiliciclasticReservoirs setting where the conditioning vector is 18-D
(layer-type one-hot + universal scalars + sin/cos azimuth + family
scalars) and adjacent blocks may have different layer types.

Three transition strategies are supported via ``transition_mode``:

  - ``"hard"``    : 1 block per slot, pure one-hot, velocity blending in
                    overlap (the existing MultiDiffusion recipe).
  - ``"soft"``    : 2x grid in X — odd X positions are "transition"
                    blocks whose one-hot is a 0.5/0.5 mix of the two
                    neighbours' one-hots (their family scalars are also
                    averaged). Even positions are pure layer types.
  - ``"buffer"``  : 2x grid in X — odd X positions are unconditional
                    (model called without ``cond``, so the network's
                    learned null embedding is used). Even positions are
                    pure layer types.

In all three modes the generation loop is identical; the only difference
is what cond / mode is fed to the model per block.

Data axis convention: tensor is (C, X, Y, Z) where Z is the LAST axis
(depth, surface at z=Z-1). For SiliciclasticReservoirs blocks are
(64, 64, 32). The big volume tiles X and Y; Z stays at block depth.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-block conditioning construction
# ---------------------------------------------------------------------------

# Order must match data_reservoirs.LAYER_TYPES (used as one-hot indices).
LAYER_TYPES = [
    'lobe',
    'channel:PV_SHOESTRING',
    'channel:CB_LABYRINTH',
    'channel:CB_JIGSAW',
    'channel:SH_DISTAL',
    'channel:SH_PROXIMAL',
    'channel:MEANDER_OXBOW',
    'delta',
]
NUM_LAYERS = len(LAYER_TYPES)
LAYER_TYPE_TO_IDX = {n: i for i, n in enumerate(LAYER_TYPES)}

UNIVERSAL_CONT = ['ntg', 'width_cells', 'depth_cells']
FAMILY_CONT = ['asp', 'mCHsinu', 'mFFCHprop', 'probAvulInside',
               'trunk_length_fraction']
CONT_COLS = UNIVERSAL_CONT + FAMILY_CONT
COND_DIM = NUM_LAYERS + len(UNIVERSAL_CONT) + 2 + len(FAMILY_CONT)  # 18


@dataclass
class BlockSpec:
    """Per-block specification before normalization.

    ``layer_idx`` is None for buffer blocks (no cond at all). ``layer_mix``
    overrides ``layer_idx`` and lets the one-hot be a soft mixture; it
    must be a length-NUM_LAYERS vector summing to ~1.
    """
    layer_idx: int | None
    azimuth_deg: float
    raw_scalars: dict     # {col_name: raw_value}
    layer_mix: np.ndarray | None = None  # shape (NUM_LAYERS,) or None


def build_cond_vector(spec: BlockSpec, cont_min, cont_max) -> np.ndarray | None:
    """Return an 18-D cond array for the model, or None for an unconditional
    (buffer) block.

    Universal continuous columns (ntg, width_cells, depth_cells) and family
    columns are min-max normalized using the training-time stats. Missing
    family values are 0 (matching training).
    """
    if spec.layer_idx is None and spec.layer_mix is None:
        return None  # unconditional

    onehot = np.zeros(NUM_LAYERS, dtype=np.float32)
    if spec.layer_mix is not None:
        onehot = spec.layer_mix.astype(np.float32)
    else:
        onehot[spec.layer_idx] = 1.0

    # Continuous columns: build a length-8 raw vector with NaN for any column
    # not provided, then normalize.
    raw = np.full(len(CONT_COLS), np.nan, dtype=np.float32)
    for k, col in enumerate(CONT_COLS):
        if col in spec.raw_scalars:
            raw[k] = spec.raw_scalars[col]
    cont_norm = (raw - cont_min) / (cont_max - cont_min + 1e-8)
    cont_norm = np.where(np.isnan(cont_norm), 0.0, cont_norm).astype(np.float32)

    az = (spec.azimuth_deg % 360.0) / 360.0
    sin_a = np.float32(np.sin(2 * math.pi * az))
    cos_a = np.float32(np.cos(2 * math.pi * az))

    cond = np.concatenate([
        onehot,
        cont_norm[:len(UNIVERSAL_CONT)],
        np.array([sin_a, cos_a], dtype=np.float32),
        cont_norm[len(UNIVERSAL_CONT):],
    ])
    return cond


def expand_blockspecs_for_transition(
    specs_x: list[BlockSpec],
    transition_mode: str,
) -> list[BlockSpec]:
    """For a single Y-row's per-X-block specs, insert transition blocks
    between every pair according to ``transition_mode``.

    Returns the expanded list:
      - "hard"          : N blocks, pure one-hots (no change).
      - "soft_nobuffer" : N blocks, each block's one-hot and scalars are
                          a (0.10, 0.80, 0.10) weighted mix with its
                          immediate neighbours. No extra blocks inserted.
      - "soft"          : 2N-1 blocks, odd positions get a 50/50 mixed
                          one-hot of their two pure neighbours.
      - "buffer"        : 2N-1 blocks, odd positions are unconditional.
    """
    if transition_mode == 'hard':
        return list(specs_x)

    if transition_mode == 'soft_nobuffer':
        weights = (0.10, 0.80, 0.10)
        n = len(specs_x)
        out: list[BlockSpec] = []
        for i, spec in enumerate(specs_x):
            mix = np.zeros(NUM_LAYERS, dtype=np.float32)
            scalars_avg: dict[str, float] = {}
            total_w = 0.0
            for di, w in [(-1, weights[0]), (0, weights[1]), (1, weights[2])]:
                ni = i + di
                if 0 <= ni < n:
                    src = specs_x[ni]
                    if src.layer_idx is not None:
                        mix[src.layer_idx] += w
                    elif src.layer_mix is not None:
                        mix += w * src.layer_mix
                    for k, v in src.raw_scalars.items():
                        scalars_avg[k] = scalars_avg.get(k, 0.0) + w * v
                    total_w += w
            if total_w > 0 and abs(total_w - 1.0) > 1e-6:
                mix /= total_w
                for k in list(scalars_avg.keys()):
                    scalars_avg[k] /= total_w
            out.append(BlockSpec(
                layer_idx=None, azimuth_deg=spec.azimuth_deg,
                raw_scalars=scalars_avg, layer_mix=mix,
            ))
        return out

    expanded: list[BlockSpec] = []
    for i, spec in enumerate(specs_x):
        expanded.append(spec)
        if i == len(specs_x) - 1:
            break
        nxt = specs_x[i + 1]
        if transition_mode == 'buffer':
            # Unconditional. Carry the average azimuth (used only for any
            # downstream display) and empty scalars.
            az_mid = 0.5 * (spec.azimuth_deg + nxt.azimuth_deg)
            expanded.append(BlockSpec(
                layer_idx=None, azimuth_deg=az_mid, raw_scalars={},
                layer_mix=None,
            ))
        elif transition_mode == 'soft':
            mix = np.zeros(NUM_LAYERS, dtype=np.float32)
            if spec.layer_idx is not None:
                mix[spec.layer_idx] += 0.5
            elif spec.layer_mix is not None:
                mix += 0.5 * spec.layer_mix
            if nxt.layer_idx is not None:
                mix[nxt.layer_idx] += 0.5
            elif nxt.layer_mix is not None:
                mix += 0.5 * nxt.layer_mix
            # Average raw scalars where both have them; otherwise carry
            # whichever is non-zero halved (matches "0 outside family"
            # training).
            keys = set(spec.raw_scalars) | set(nxt.raw_scalars)
            scalars = {}
            for k in keys:
                a = spec.raw_scalars.get(k, 0.0)
                b = nxt.raw_scalars.get(k, 0.0)
                scalars[k] = 0.5 * (a + b)
            az_mid = 0.5 * (spec.azimuth_deg + nxt.azimuth_deg)
            expanded.append(BlockSpec(
                layer_idx=None, azimuth_deg=az_mid, raw_scalars=scalars,
                layer_mix=mix,
            ))
        else:
            raise ValueError(f"unknown transition_mode={transition_mode!r}")
    return expanded


# ---------------------------------------------------------------------------
# Blending weights (trilinear ramp on every overlapping face)
# ---------------------------------------------------------------------------

def _compute_blend_weights(grid_shape, block_shape, overlap_xy, device):
    """Per-block 3D weight tensor that ramps to 0 on overlapping faces.

    Z has no overlap (single block in Z). X and Y use the same overlap.
    Neighbouring blocks' ramps sum to 1 in the overlap zone (corners too).
    """
    ny, nx = grid_shape
    Sx, Sy, Sz = block_shape
    weights = {}
    if overlap_xy > 0:
        ramp_up = torch.linspace(0, 1, overlap_xy + 2)[1:-1]
        ramp_down = torch.linspace(1, 0, overlap_xy + 2)[1:-1]
    for i in range(ny):
        for j in range(nx):
            w = torch.ones(Sx, Sy, Sz)
            if overlap_xy > 0:
                if j > 0:
                    w[:overlap_xy, :, :] *= ramp_up.view(-1, 1, 1)
                if j < nx - 1:
                    w[-overlap_xy:, :, :] *= ramp_down.view(-1, 1, 1)
                if i > 0:
                    w[:, :overlap_xy, :] *= ramp_up.view(1, -1, 1)
                if i < ny - 1:
                    w[:, -overlap_xy:, :] *= ramp_down.view(1, -1, 1)
            weights[(i, j)] = w.to(device)
    return weights


# ---------------------------------------------------------------------------
# Parallel denoising loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_big_reservoir_multi(
    method,
    grid_specs: list[list[BlockSpec]],   # (ny, nx) BlockSpec grid
    cont_min: np.ndarray,
    cont_max: np.ndarray,
    block_shape=(64, 64, 32),
    overlap_xy: int = 24,
    n_steps: int = 50,
    cfg_scale: float = 3.0,
    max_batch: int = 24,
    device: str = 'cuda',
    solver: str = 'euler',
    trajectory_path=None,
    trajectory_every: int = 1,
):
    """Run MultiDiffusion-style denoising for an (ny, nx) grid of blocks.

    trajectory_path: if set, the whole field is recorded every
    `trajectory_every` steps (fp16, plus the final state) and written to
    this .npz with keys 'traj' (S, Tx, Ty, Sz) and 't' (S,). The path
    shape is often the result; keep it rather than re-running.

    solver: 'euler' (paper) or 'heun' (2 averaged-velocity evaluations per
    step; the averaged field is re-evaluated at the predictor point).

    The grid is rectangular in (Y, X). Each cell may be conditional (its
    own one-hot or a soft mix) or unconditional (BlockSpec.layer_idx is
    None and layer_mix is None — buffer block). For unconditional blocks
    the model is called without ``cond``, using its trained null embedding.
    For conditional blocks, CFG = uncond + cfg_scale * (cond - uncond).

    Returns:
        (X_total, Y_total, Z_total) float volume in [-1, 1] (continuous,
        not yet binarized) — plus a per-step elapsed-time list.
    """
    ny = len(grid_specs)
    nx = len(grid_specs[0])
    Sx, Sy, Sz = block_shape
    stride_x = Sx - overlap_xy
    stride_y = Sy - overlap_xy
    Tx = Sx + (nx - 1) * stride_x
    Ty = Sy + (ny - 1) * stride_y

    positions = [(i, j) for i in range(ny) for j in range(nx)]

    # Pre-build per-block cond and an "is unconditional" flag.
    cond_per = {}
    uncond_flag = {}
    for i, j in positions:
        spec = grid_specs[i][j]
        v = build_cond_vector(spec, cont_min, cont_max)
        if v is None:
            cond_per[(i, j)] = None
            uncond_flag[(i, j)] = True
        else:
            cond_per[(i, j)] = torch.from_numpy(v).to(device)
            uncond_flag[(i, j)] = False

    blend_w = _compute_blend_weights((ny, nx), block_shape, overlap_xy, device)

    # Initialize with noise
    x_global = torch.randn(Tx, Ty, Sz, device=device)
    dt = 1.0 / n_steps
    step_times = []

    # For unconditional generation the model takes (mask, known) inpaint
    # context with both zero (matches training's empty-mask 30% case).
    zero_mask = torch.zeros(1, 1, Sx, Sy, Sz, device=device)
    zero_known = torch.zeros_like(zero_mask)
    method.model.set_inpaint_context(zero_mask.expand(max_batch, -1, -1, -1, -1).contiguous(),
                                     zero_known.expand(max_batch, -1, -1, -1, -1).contiguous())

    def block_origin(i, j):
        return j * stride_x, i * stride_y

    def avg_velocity(x_state, t_val):
        """Blend-weighted MultiDiffusion velocity of the whole field at t."""
        v_accum = torch.zeros_like(x_state)
        w_accum = torch.zeros_like(x_state)

        # Group blocks by (is_uncond, has_cond+cfg) so we can stack into
        # batches with identical input semantics.
        cond_pos = [p for p in positions if not uncond_flag[p]]
        uncond_pos = [p for p in positions if uncond_flag[p]]

        for group_pos, group_uncond in [(cond_pos, False), (uncond_pos, True)]:
            for b_start in range(0, len(group_pos), max_batch):
                batch_pos = group_pos[b_start:b_start + max_batch]
                bs = len(batch_pos)
                if bs == 0:
                    continue

                # Re-set inpaint context to match the actual batch size
                method.model.set_inpaint_context(
                    zero_mask.expand(bs, -1, -1, -1, -1).contiguous(),
                    zero_known.expand(bs, -1, -1, -1, -1).contiguous(),
                )

                blocks = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
                for idx, (i, j) in enumerate(batch_pos):
                    xs, ys = block_origin(i, j)
                    blocks[idx, 0] = x_state[xs:xs + Sx, ys:ys + Sy, :]

                t_tensor = torch.full((bs,), t_val, device=device)
                t_emb_in = t_tensor * 1000  # the trained model uses t*1000 as time emb input

                if group_uncond:
                    v_pred = method.model(blocks, t_emb_in)
                else:
                    cond_batch = torch.stack(
                        [cond_per[p] for p in batch_pos], dim=0
                    )
                    if cfg_scale > 0:
                        v_cond = method.model(blocks, t_emb_in, cond_batch)
                        v_uncond = method.model(blocks, t_emb_in)
                        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v_pred = method.model(blocks, t_emb_in, cond_batch)

                # Scatter weighted velocities
                for idx, (i, j) in enumerate(batch_pos):
                    xs, ys = block_origin(i, j)
                    w = blend_w[(i, j)]
                    v_accum[xs:xs + Sx, ys:ys + Sy, :] += w * v_pred[idx, 0]
                    w_accum[xs:xs + Sx, ys:ys + Sy, :] += w

        return v_accum / w_accum.clamp(min=1e-8)

    traj, traj_t = [], []
    for step in range(n_steps):
        t_val = step * dt
        t0 = time.time()
        v = avg_velocity(x_global, t_val)
        if solver == 'heun':
            v = 0.5 * (v + avg_velocity(x_global + v * dt, t_val + dt))
        x_global = x_global + v * dt
        if trajectory_path is not None and ((step + 1) % trajectory_every == 0
                                            or step == n_steps - 1):
            traj.append(x_global.half().cpu().numpy()); traj_t.append((step + 1) * dt)

        elapsed = time.time() - t0
        step_times.append(elapsed)
        if step % 10 == 0 or step == n_steps - 1:
            print(f"    step {step:3d}/{n_steps}  {elapsed:.2f}s")

    method.model.clear_inpaint_context()
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, traj=np.stack(traj), t=np.array(traj_t, dtype=np.float32))
    return x_global.cpu(), step_times


def grid_layout_info(grid_specs, block_shape=(64, 64, 32), overlap_xy=24):
    """Helper: return total volume shape and per-block (xs, xe, ys, ye)
    bounding boxes for downstream visualization."""
    ny = len(grid_specs)
    nx = len(grid_specs[0])
    Sx, Sy, Sz = block_shape
    stride_x = Sx - overlap_xy
    stride_y = Sy - overlap_xy
    Tx = Sx + (nx - 1) * stride_x
    Ty = Sy + (ny - 1) * stride_y
    boxes = {}
    for i in range(ny):
        for j in range(nx):
            xs, ys = j * stride_x, i * stride_y
            boxes[(i, j)] = (xs, xs + Sx, ys, ys + Sy)
    return (Tx, Ty, Sz), boxes
