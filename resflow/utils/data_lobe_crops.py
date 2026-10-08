"""Random 64x64x32 crops of large-domain lobe volumes, plus the
assembly-aware mask distribution that goes with them.

The base dataset mirrors ``data_reservoirs.ReservoirDataset``'s output
contract -- ``(facies, cond)`` with facies ``(1, X, Y, Z)`` in {-1, +1}
and an 18-D cond vector -- so it drops straight into the specialist
training recipe.

Conditioning note: ``ntg``, ``width_cells`` etc. are read from the PARENT
192x192x32 volume, not recomputed on the crop. That is deliberate. The
conditioning vector describes the depositional system, and at generation
time we ask a whole assembly to realize a requested NTG; a model trained
on per-crop NTG would instead learn "make this 64-window have exactly
this proportion", which is a different and less useful conditioning
semantics. The crop's own NTG fluctuates around the parent's, which is
exactly the variability a window of a larger field should show.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .data_reservoirs import (
    CONT_COLS, LAYER_TYPE_TO_IDX, NUM_LAYERS, UNIVERSAL_CONT,
)
from .masking import generate_well_mask
from .masking_context import generate_context_mask

CROP_SHAPE = (64, 64, 32)


def _build_cond(row, cont_min, cont_max, layer_type='lobe'):
    """18-D cond vector: one-hot | universal scalars | sin/cos az | family."""
    onehot = np.zeros(NUM_LAYERS, dtype=np.float32)
    onehot[LAYER_TYPE_TO_IDX[layer_type]] = 1.0

    raw = np.full(len(CONT_COLS), np.nan, dtype=np.float32)
    for k, col in enumerate(CONT_COLS):
        v = row.get(col)
        if v is not None and np.isfinite(v):
            raw[k] = v
    norm = (raw - cont_min) / (cont_max - cont_min + 1e-8)
    norm = np.where(np.isnan(norm), 0.0, norm).astype(np.float32)

    az = (float(row['azimuth']) % 360.0) / 360.0
    sin_a = np.float32(np.sin(2 * np.pi * az))
    cos_a = np.float32(np.cos(2 * np.pi * az))

    return np.concatenate([
        onehot,
        norm[:len(UNIVERSAL_CONT)],
        np.array([sin_a, cos_a], dtype=np.float32),
        norm[len(UNIVERSAL_CONT):],
    ]).astype(np.float32)


class LobeCropDataset(Dataset):
    """Random crops of large-domain lobe volumes.

    One item per parent volume; the crop origin is redrawn every epoch
    (the RNG is seeded per __getitem__ call from a base seed and the
    epoch counter, so DataLoader workers stay reproducible).
    """

    def __init__(self, data_dir, cont_min, cont_max, crop_shape=CROP_SHAPE,
                 layer='lobe', seed=0, indices=None, augment=False):
        self.root = Path(data_dir) / layer
        self.crop_shape = tuple(crop_shape)
        self.cont_min = np.asarray(cont_min, dtype=np.float32)
        self.cont_max = np.asarray(cont_max, dtype=np.float32)
        self.seed = int(seed)
        self.layer = layer
        # 8x dihedral augmentation in the horizontal plane. Lobes carry an
        # explicit azimuth in the conditioning vector, so a rotation is only
        # valid if the azimuth is rotated with it -- otherwise the model is
        # taught that geometry and conditioning disagree. Azimuth is
        # clockwise from +x (ResMill convention), so a k*90 deg array
        # rotation shifts it by -90k and an x-mirror negates it.
        # Motivation: longer training HURTS geostatistics here (ep160 was
        # 2.8x worse than ep65 while its loss kept falling), which is the
        # signature of overfitting on 180k volumes.
        self.augment = bool(augment)

        shards = sorted(glob.glob(str(self.root / 'shard_*')))
        if not shards:
            raise FileNotFoundError(f'no shards under {self.root}')

        # (shard_id, row) index plus the per-row cond, built once.
        self.shard_dirs = shards
        keys, conds = [], []
        for sid, d in enumerate(shards):
            t = pq.read_table(Path(d) / 'params_slim.parquet')
            rows = t.to_pylist()
            for r, row in enumerate(rows):
                keys.append((sid, r))
                conds.append(_build_cond(row, self.cont_min, self.cont_max,
                                         self.layer))
        self.keys = keys
        self.conds = np.stack(conds)
        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            self.keys = [self.keys[i] for i in idx]
            self.conds = self.conds[idx]

        self._mmap = {}
        self.epoch = 0

    def set_epoch(self, epoch):
        """Redraw crop origins each epoch (call from the training loop)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.keys)

    def _facies(self, sid):
        m = self._mmap.get(sid)
        if m is None:
            m = np.load(Path(self.shard_dirs[sid]) / 'facies.npy',
                        mmap_mode='r')
            self._mmap[sid] = m
        return m

    def __getitem__(self, idx):
        sid, row = self.keys[idx]
        arr = self._facies(sid)
        vol = arr[row]                      # (X, Y, Z) int8 {0, 1}
        cx, cy, cz = self.crop_shape

        rng = np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch) * 1_000_003 + idx)
        x0 = int(rng.integers(0, vol.shape[0] - cx + 1))
        y0 = int(rng.integers(0, vol.shape[1] - cy + 1))
        z0 = int(rng.integers(0, vol.shape[2] - cz + 1))
        crop = np.asarray(vol[x0:x0 + cx, y0:y0 + cy, z0:z0 + cz])

        cond = self.conds[idx].copy()
        if self.augment:
            k = int(rng.integers(0, 4))
            mirror = bool(rng.integers(0, 2))
            if k:
                crop = np.rot90(crop, k=k, axes=(0, 1))
            if mirror:
                crop = crop[::-1]
            crop = np.ascontiguousarray(crop)
            # cond layout: [8 one-hot | ntg, width, depth | sin, cos | family]
            si, ci = NUM_LAYERS + len(UNIVERSAL_CONT), NUM_LAYERS + len(UNIVERSAL_CONT) + 1
            ang = np.arctan2(cond[si], cond[ci])          # 2*pi*az_norm
            ang = ang - k * (np.pi / 2.0)
            if mirror:
                ang = -ang
            cond[si], cond[ci] = np.float32(np.sin(ang)), np.float32(np.cos(ang))
        facies = torch.from_numpy(crop.astype(np.float32) * 2.0 - 1.0)
        return facies.unsqueeze(0), torch.from_numpy(cond)


def generate_assembly_training_mask(volume_shape, uncond_prob=0.30,
                                    context_share=0.72, max_wells=5,
                                    p_through=0.5, generator=None,
                                    config_set='all6'):
    """Wells + neighbour-context training mask.

    The unconditional fraction is held at 30% because that is exactly
    what the assembly benchmark's empty-mask native generation
    exercises; changing it would confound comparison with the
    specialist. The remaining 70% is split between neighbour-context
    slabs (all six configurations) and the original well masks.

    context_share is raised from 0.5 to 0.72 relative to the first
    Addendum-G models because six configurations now have to be covered
    instead of three; wells fall to ~20% of samples, which is a
    disclosed cost to well-conditioned performance.
    """
    u = torch.rand(1, generator=generator).item()
    if u < uncond_prob:
        return torch.zeros(1, *volume_shape)
    v = torch.rand(1, generator=generator).item()
    if v < context_share:
        return generate_context_mask(volume_shape, generator=generator,
                                     config_set=config_set)
    return generate_well_mask(volume_shape, max_wells=max_wells,
                              p_through=p_through)


class AssemblyInpaintDataset(Dataset):
    """Wraps a (facies, cond) dataset and adds mask + context noise level.

    Yields (facies, cond, mask, s). `s` is the noise level the context
    will be presented at: s = 1 is clean context (classic outpainting,
    what the sequential schedulers need), s < 1 is a partially denoised
    neighbour, which is what makes the fully parallel coupled sampler
    possible. Training over a mixture of both means one model serves
    every scheduler.

    `traj_prob` is the fraction of samples drawn with s ~ U(0,1); the
    rest use s = 1. Keeping half the mass at s = 1 preserves the clean
    context case that the sequential schemes rely on.
    """

    def __init__(self, base_dataset, volume_shape=CROP_SHAPE,
                 traj_prob=0.0, **mask_kw):
        self.base_dataset = base_dataset
        self.volume_shape = tuple(volume_shape)
        self.traj_prob = float(traj_prob)
        self.mask_kw = mask_kw

    def __len__(self):
        return len(self.base_dataset)

    def set_epoch(self, epoch):
        if hasattr(self.base_dataset, 'set_epoch'):
            self.base_dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        facies, cond = self.base_dataset[idx]
        # the item's own shape: equals volume_shape for fixed-size crops, and
        # follows the crop when a batch sampler varies the size per step
        mask = generate_assembly_training_mask(tuple(facies.shape[1:]),
                                               **self.mask_kw)
        if self.traj_prob > 0 and torch.rand(1).item() < self.traj_prob:
            s = torch.rand(1)
        else:
            s = torch.ones(1)
        return facies, cond, mask, s
