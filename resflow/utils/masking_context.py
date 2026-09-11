"""Neighbour-context masks and trajectory conditioning for tiled assembly.

Background
----------
MultiDiffusion fuses overlapping blocks by *averaging their predicted
velocities*. The average of two velocity fields is not the velocity field
of any distribution: where two blocks disagree about a boundary, the mean
is a low-amplitude smear over both positions, which binarization then
resolves into one merged body. Measured on ResBench Addendum E, that
triples geobody-size W1, identically for the foundation model and for an
80-epoch specialist -- so it is a property of the fusion rule.

The alternative is to *condition* rather than average: give a block its
neighbours' content as known data through the model's inpainting
interface. This module builds the masks that make that trainable, in two
generations of capability.

Neighbour configurations
------------------------
Which sides a block already knows depends on the generation schedule:

  raster order      -- left, top, left+top
  4-colour staging  -- nothing (stage 1), left+right (2), top+bottom (3),
                       all four sides (4)

A model that has only seen L-shaped masks is out of distribution on
left+right, so supporting both schedulers means training on all six.

Trajectory conditioning
-----------------------
Classic outpainting hands over *clean* neighbour values, which forces a
sequential schedule. If instead the context may itself be partially
denoised -- at its own noise level s -- then blocks can advance together:
for the joint variable z = (x_A, x_B), the flow-matching velocity is
E[z1 - z0 | z_t], whose A-component is exactly "my velocity given my
noisy state and my neighbour's noisy state at time t". Conditioning on a
neighbour's trajectory and integrating the coupled ODE is therefore the
*exact* joint flow, not an approximation -- and it is fully parallel.

s = 1 recovers clean context (and hence the sequential schemes), so one
model trained over s in [0, 1] supports every scheduler.

This mirrors Diffusion Forcing (Chen et al., NeurIPS 2024), which trains
a diffusion model over independent per-token noise levels; here the
"tokens" are spatial blocks rather than sequence positions.

Mask convention: 1 = known (context), 0 = to be generated.
"""
from __future__ import annotations

import torch

# All six neighbour configurations, with the relative frequency each one
# occurs at in the schedules we actually run. Raster order on an n x n
# grid yields 1 corner with no context, (n-1) blocks with one arm along
# each of the top row and left column, and (n-1)^2 interior blocks with
# both -- so 'left_top' dominates raster. The 4-colour schedule
# contributes equal numbers of left_right, top_bottom and all_four.
# Sampling uniformly over configurations instead would badly under-train
# the interior case, which is the overwhelming majority of real blocks.
GRID_N = 10
CONFIG_WEIGHTS = {
    'left':       GRID_N - 1,
    'top':        GRID_N - 1,
    'left_top':   (GRID_N - 1) ** 2,
    'left_right': (GRID_N // 2) ** 2,
    'top_bottom': (GRID_N // 2) ** 2,
    'all_four':   (GRID_N // 2) ** 2,
}
CONFIGS = list(CONFIG_WEIGHTS)

# Slab width is randomized rather than pinned to the deployed 24 so the
# generation-time overlap can be swept without retraining. Overlap must
# stay <= block/2 for a 4-colour schedule to keep same-stage blocks
# disjoint (2*stride >= block), which for block 64 means <= 32.
OVERLAP_MIN = 8
OVERLAP_MAX = 32


# Raster order only ever needs three of the six configurations. Training
# on all six spreads the same budget thinner, which measurably cost
# accuracy in Addendum G round 3 (0.0914 vs 0.0811 under raster), so the
# set is selectable rather than fixed.
CONFIG_SETS = {
    'raster3': ['left', 'top', 'left_top'],
    'all6': CONFIGS,
}


def sample_context_config(generator=None, config_set='all6'):
    keys = CONFIG_SETS[config_set]
    w = torch.tensor([float(CONFIG_WEIGHTS[k]) for k in keys])
    return keys[int(torch.multinomial(w, 1, generator=generator).item())]


def generate_context_mask(volume_shape, overlap=None, config=None,
                          generator=None, config_set='all6'):
    """Neighbour-context mask for one of the six configurations.

    Returns (1, X, Y, Z) float tensor, 1 = known.
    """
    X, Y, Z = volume_shape
    if overlap is None:
        overlap = int(torch.randint(OVERLAP_MIN, OVERLAP_MAX + 1, (1,),
                                    generator=generator).item())
    overlap = max(1, min(int(overlap), min(X, Y) // 2))
    if config is None:
        config = sample_context_config(generator, config_set)

    m = torch.zeros(1, X, Y, Z)
    if config in ('left', 'left_top', 'left_right', 'all_four'):
        m[0, :overlap, :, :] = 1.0
    if config in ('top', 'left_top', 'top_bottom', 'all_four'):
        m[0, :, :overlap, :] = 1.0
    if config in ('left_right', 'all_four'):
        m[0, -overlap:, :, :] = 1.0
    if config in ('top_bottom', 'all_four'):
        m[0, :, -overlap:, :] = 1.0
    return m


def build_assembly_context_mask(volume_shape, sides, overlap):
    """Deterministic mask from an explicit set of known sides.

    `sides` is any subset of {'left', 'right', 'top', 'bottom'}; used by
    the samplers, where which neighbours exist is known exactly.
    """
    X, Y, Z = volume_shape
    m = torch.zeros(1, X, Y, Z)
    if 'left' in sides:
        m[0, :overlap, :, :] = 1.0
    if 'right' in sides:
        m[0, -overlap:, :, :] = 1.0
    if 'top' in sides:
        m[0, :, :overlap, :] = 1.0
    if 'bottom' in sides:
        m[0, :, -overlap:, :] = 1.0
    return m


def noisy_context(clean, s, generator=None):
    """Context values at noise level s along the flow-matching path.

    x_s = (1 - s) * eps + s * x_clean, the same linear interpolant the
    method trains on, so no ODE integration is needed to synthesize a
    partially denoised neighbour. s = 1 returns the clean values.
    """
    if isinstance(s, float) or (torch.is_tensor(s) and s.ndim == 0):
        s = torch.full((clean.shape[0],), float(s), device=clean.device)
    s_exp = s.view(-1, *([1] * (clean.ndim - 1))).to(clean.dtype)
    eps = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype,
                      generator=generator)
    return (1 - s_exp) * eps + s_exp * clean


def stage_of(i, j):
    """4-colour class (1..4) of grid cell (i, j).

    Same-class blocks sit 2 strides apart, so with overlap <= block/2
    they are disjoint and can be generated in parallel. Four is the
    minimum number of colours for that: you need k*stride >= block, i.e.
    k >= block/stride, which is 2 per axis for overlap in [8, 32].
    """
    return 1 + (i % 2) * 2 + (j % 2)


def known_sides_4stage(i, j, ny, nx):
    """Which neighbours are already final when cell (i, j) is generated.

    Stage 1 has none; stage 2 sees class-1 blocks left and right; stage 3
    sees class-1 above and below; stage 4 is enclosed by classes 2 and 3
    on all four sides (93.75% of it is known at overlap 24).
    """
    st = stage_of(i, j)
    sides = set()
    if st == 1:
        return sides
    if st == 2:
        if j > 0:
            sides.add('left')
        if j < nx - 1:
            sides.add('right')
    elif st == 3:
        if i > 0:
            sides.add('top')
        if i < ny - 1:
            sides.add('bottom')
    else:
        if j > 0:
            sides.add('left')
        if j < nx - 1:
            sides.add('right')
        if i > 0:
            sides.add('top')
        if i < ny - 1:
            sides.add('bottom')
    return sides


def known_sides_raster(i, j):
    """Neighbours already final in raster order: left and top only."""
    sides = set()
    if j > 0:
        sides.add('left')
    if i > 0:
        sides.add('top')
    return sides
