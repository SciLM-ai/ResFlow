"""Training crops LARGER than the released 64x64x32 windows, cut from the raw
128 x 128 x 64 ResMill volumes the windows were taken from.

The released dataset's train split names each window's parent volume
(`source_shard`, `source_row` in params.parquet), so the split is inherited
exactly: a volume whose window is in train is used here, nothing else.

Each item is a crop of `crop_shape` (default the full 128 x 128 plan area,
32 cells tall) whose vertical origin is redrawn every epoch in 1..31 (the
dataset's own rule, which keeps the engine's floor/roof fingerprint out).
The condition's `ntg` is the crop's REALISED sand fraction, computed here;
every other column (width, depth, family, azimuth) is the volume's own and
is read from the released params.parquet.
"""
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .data_reservoirs import CONT_COLS, LAYER_TYPE_TO_IDX
from .data_lobe_crops import _build_cond

RAW_DIR = {'lobe': 'lobes', 'delta': 'delta'}


def raw_env_dir(layer_type):
    return RAW_DIR.get(layer_type, layer_type.split(':')[-1].lower())


class RawCropDataset(Dataset):
    def __init__(self, data_dir, raw_root, cont_min, cont_max, envs,
                 crop_shape=(128, 128, 32), split='train', seed=42, cache=None):
        self.raw_root = Path(raw_root)
        self.crop_shape = tuple(crop_shape)
        self.cont_min, self.cont_max = cont_min, cont_max
        self.seed, self.epoch = seed, 0
        cache = Path(cache or Path(data_dir) / '_raw_index' / f'{split}.npz')
        if not cache.exists():
            self._build(Path(data_dir), split, cache)
        z = np.load(cache, allow_pickle=True)
        keep = np.isin(z['layer'], [LAYER_TYPE_TO_IDX[e] for e in envs])
        self.layer = z['layer'][keep]
        self.path_id = z['path_id'][keep]
        self.row = z['row'][keep]
        self.params = z['params'][keep]          # (n, len(CONT_COLS)+1): CONT_COLS..., azimuth
        self.paths = list(z['paths'])
        self._mmap = {}

    @staticmethod
    def _build(data_dir, split, cache):
        from collections import defaultdict
        sp = pq.read_table(data_dir / 'splits' / f"{'validation' if split == 'val' else split}.parquet").to_pandas()
        by = defaultdict(list)
        for lt, sd, si in zip(sp['layer_type'], sp['shard_dir'], sp['sample_idx']):
            by[(lt, sd)].append(int(si))
        paths, pid, layer, row, params = {}, [], [], [], []
        for (lt, sd), idx in sorted(by.items()):
            t = pq.read_table(data_dir / sd / 'params.parquet').to_pandas()
            for i in idx:
                r = t.iloc[i]
                p = str(Path(raw_env_dir(lt)) / r['source_shard'] / 'facies.npy')
                pid.append(paths.setdefault(p, len(paths)))
                layer.append(LAYER_TYPE_TO_IDX[lt]); row.append(int(r['source_row']))
                params.append([float(r[c]) if c in t.columns and r[c] is not None else np.nan
                               for c in CONT_COLS] + [float(r['azimuth'])])
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + f'.{os.getpid()}.tmp.npz')
        np.savez(tmp, layer=np.array(layer, np.int8), path_id=np.array(pid, np.int32),
                 row=np.array(row, np.int16), params=np.array(params, np.float32),
                 paths=np.array(list(paths), dtype=object))
        os.replace(tmp, cache)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def restrict(self, indices):
        """Keep only `indices` (in that order). Used instead of torch Subset so
        (index, size) items from SizedBatchSampler still work."""
        ix = np.asarray(indices)
        self.layer, self.path_id = self.layer[ix], self.path_id[ix]
        self.row, self.params = self.row[ix], self.params[ix]
        return self

    def __len__(self):
        return len(self.row)

    def __getitem__(self, idx):
        size = None
        if isinstance(idx, (tuple, list)):
            idx, size = int(idx[0]), int(idx[1])
        pid = int(self.path_id[idx])
        m = self._mmap.get(pid)
        if m is None:
            if len(self._mmap) >= 256:
                self._mmap.pop(next(iter(self._mmap)))
            m = np.load(self.raw_root / self.paths[pid], mmap_mode='r')
            self._mmap[pid] = m
        vol = m[int(self.row[idx])]                               # (128, 128, 64)
        cx, cy, cz = self.crop_shape
        if size is not None:
            cx = cy = size
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch) * 1_000_003 + int(idx))
        x0 = int(rng.integers(0, vol.shape[0] - cx + 1))
        y0 = int(rng.integers(0, vol.shape[1] - cy + 1))
        z0 = int(rng.integers(1, min(31, vol.shape[2] - cz) + 1))  # dataset rule: z0 in 1..31
        crop = np.asarray(vol[x0:x0 + cx, y0:y0 + cy, z0:z0 + cz], dtype=np.float32)
        p = self.params[idx]
        rowd = {c: (None if np.isnan(p[k]) else float(p[k])) for k, c in enumerate(CONT_COLS)}
        rowd['ntg'] = float(crop.mean())                          # realised, of THIS crop
        rowd['azimuth'] = float(p[-1])
        cond = _build_cond(rowd, self.cont_min, self.cont_max,
                           layer_type=list(LAYER_TYPE_TO_IDX)[int(self.layer[idx])])
        return torch.from_numpy(crop * 2.0 - 1.0).unsqueeze(0), torch.from_numpy(cond)


class SizedBatchSampler:
    """Batches of (index, crop size): one plan-view crop size per optimizer
    step, drawn from `sizes` with rng(seed, epoch, step) -- the SAME on every
    rank (so no rank waits on a bigger crop) and different from step to step.
    Crops smaller than the volume sit at random positions, so their sides are
    mostly inside the volume rather than on the engine's domain boundary."""
    def __init__(self, sampler, batch_size, sizes, seed=42):
        self.sampler, self.batch_size = sampler, int(batch_size)
        self.sizes, self.seed, self.epoch = [int(s) for s in sizes], int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        batch, step = [], 0
        size = int(rng.choice(self.sizes))
        for i in self.sampler:
            batch.append((int(i), size))
            if len(batch) == self.batch_size:
                yield batch
                batch, step = [], step + 1
                size = int(rng.choice(self.sizes))
