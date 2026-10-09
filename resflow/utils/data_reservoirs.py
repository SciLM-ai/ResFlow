"""DataLoader for SciLM/SiliciclasticReservoirs (binary facies + slim params).

Mirrors the structure of ``data_lobes.py`` but works against a sharded HuggingFace
dataset with 1M samples across 8 reservoir architectures. Per dataset README:

    <layer_type>/shard_XXXX/facies.npy           # (N, 64, 64, 32) int8 {0, 1}
    <layer_type>/shard_XXXX/params_slim.parquet  # per-row conditioning
    splits/{train, validation, test}.parquet     # (layer_type, shard_dir, sample_idx)

Conditioning excludes caption / perm_ave / poro_ave (user instruction: caption
requires an LLM, perm/poro make no sense for binary-facies training). The cube
is converted to float and re-mapped {0, 1} -> {-1, 1} like the lobe loader.
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Canonical layer-type ordering -> int index for one-hot.
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
LAYER_TYPE_TO_IDX = {lt: i for i, lt in enumerate(LAYER_TYPES)}
NUM_LAYERS = len(LAYER_TYPES)

# Universal continuous columns from params_slim (caption/perm_ave/poro_ave excluded).
UNIVERSAL_CONT = ['ntg', 'width_cells', 'depth_cells']
ANGLE_COL = 'azimuth'  # degrees in [0, 360)
FAMILY_CONT = ['asp', 'mCHsinu', 'mFFCHprop', 'probAvulInside', 'trunk_length_fraction']
CONT_COLS = UNIVERSAL_CONT + FAMILY_CONT  # 8 continuous, NaN if not in shard schema

# Cond layout sent to the model (one-hot + normalized continuous + sin/cos azimuth).
# Order matters for any downstream sample.py.
COND_DIM = NUM_LAYERS + len(UNIVERSAL_CONT) + 2 + len(FAMILY_CONT)  # 8 + 3 + 2 + 5 = 18


# Map from internal split name -> filename (HF release uses 'validation').
_SPLIT_FILENAMES = {'train': 'train', 'val': 'validation', 'test': 'test'}


def _read_parquet(path):
    """pq.read_table that goes through Python's open() instead of pyarrow's
    file handling. Works around an `OSError: [Errno 14] Bad address` from
    pyarrow when reading parquet directly off some BeeGFS-backed $SCRATCH filesystems. The
    parquet files we touch (splits + per-shard params_slim) are at most a
    few MB, so the buffer copy is free."""
    with open(path, 'rb') as f:
        return pq.read_table(f)


class ReservoirDataset(Dataset):
    """Map-style dataset over the official splits.

    Caches a flat (n, 8) conditioning array + per-row shard pointers to
    ``data_dir/_cond_cache/<split>.npz`` so subsequent runs skip the per-shard
    parquet sweep.
    """

    def __init__(self, data_dir, split='train', cont_min=None, cont_max=None,
                 download=True):
        self.data_dir = Path(data_dir)
        self.split = split

        # Auto-fetch from HuggingFace if data is missing. Mirrors torchvision's
        # MNIST(download=True) idiom. snapshot_download is a no-op when files
        # already exist (verifies hash, takes ~1 s) so this is safe to run on
        # every job. Only the binary-facies subset is fetched (~123 GB);
        # poro / perm / facies_alluvsim / full params / captions are skipped.
        if download:
            self._ensure_dataset_local()

        cache_dir = self.data_dir / '_cond_cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_path = cache_dir / f'{split}.npz'

        if not cache_path.exists():
            self._build_cache(cache_path)

        data = np.load(cache_path, allow_pickle=True)
        self.shard_keys = data['shard_keys']            # (n,) int32
        self.shard_dirs = list(data['shard_dirs'])      # list[str], shard pool
        self.sample_idx = data['sample_idx']            # (n,) int32
        self.layer_idx = data['layer_idx']              # (n,) int32
        self.cont_raw = data['cont_raw']                # (n, 8) float32, NaN for missing
        self.azimuth = data['azimuth']                  # (n,) float32 in [0, 360)

        # Per-column [0, 1] normalization stats. Pass training stats explicitly to
        # val/test so they share the same scaling.
        if cont_min is None or cont_max is None:
            assert split == 'train', \
                "Pass cont_min/cont_max from the training split for val/test."
            self.cont_min = np.nanmin(self.cont_raw, axis=0).astype(np.float32)
            self.cont_max = np.nanmax(self.cont_raw, axis=0).astype(np.float32)
        else:
            self.cont_min = cont_min.astype(np.float32)
            self.cont_max = cont_max.astype(np.float32)

        self._mmap_cache = {}  # per-worker shard_dir -> mmap

    # -- dataset auto-download --------------------------------------------
    def _ensure_dataset_local(self):
        """Fetch the binary-facies subset of SciLM/SiliciclasticReservoirs
        into self.data_dir if it isn't already there.

        Skipped silently if every file we need is present (saves the HF API
        round-trip on warm runs)."""
        # Fast path: if the splits parquet AND at least one shard's files
        # exist, assume the dataset is fully present and skip snapshot_download.
        splits_ok = all(
            (self.data_dir / 'splits' / f'{f}.parquet').exists()
            for f in ('train', 'validation', 'test')
        )
        sentinel_shard = self.data_dir / 'lobe' / 'shard_0000'
        sentinel_ok = ((sentinel_shard / 'facies.npy').exists()
                       and (sentinel_shard / 'params_slim.parquet').exists())
        if splits_ok and sentinel_ok:
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to auto-fetch SiliciclasticReservoirs. "
                "Install with `pip install huggingface_hub`."
            ) from e

        self.data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="SciLM/SiliciclasticReservoirs",
            repo_type="dataset",
            local_dir=str(self.data_dir),
            allow_patterns=[
                "splits/*.parquet",
                "*/shard_*/facies.npy",
                "*/shard_*/params_slim.parquet",
                "README.md",
                "DATASHEET.md",
            ],
            max_workers=8,
        )

    # -- cache build ------------------------------------------------------
    def _build_cache(self, cache_path):
        split_file = _SPLIT_FILENAMES[self.split]
        split_table = _read_parquet(self.data_dir / 'splits' / f'{split_file}.parquet')
        ltype = split_table['layer_type'].to_pylist()
        shard_dir_list = split_table['shard_dir'].to_pylist()
        sample_idx = np.asarray(split_table['sample_idx'].to_pylist(), dtype=np.int32)

        n = len(ltype)
        unique_shards = sorted(set(shard_dir_list))
        shard_to_id = {s: i for i, s in enumerate(unique_shards)}
        shard_keys = np.fromiter((shard_to_id[s] for s in shard_dir_list),
                                 dtype=np.int32, count=n)
        layer_idx = np.fromiter((LAYER_TYPE_TO_IDX[lt] for lt in ltype),
                                dtype=np.int32, count=n)

        cont_raw = np.full((n, len(CONT_COLS)), np.nan, dtype=np.float32)
        azimuth = np.zeros(n, dtype=np.float32)

        # Group split rows by shard so we read each params_slim once.
        by_shard = defaultdict(list)
        for i, s in enumerate(shard_dir_list):
            by_shard[s].append(i)

        for s, row_ids in tqdm(by_shard.items(),
                               desc=f'Building {self.split} cond cache'):
            t = _read_parquet(self.data_dir / s / 'params_slim.parquet')
            cols = set(t.column_names)
            cached = {c: t[c].to_numpy(zero_copy_only=False)
                      for c in (CONT_COLS + [ANGLE_COL]) if c in cols}
            for ri in row_ids:
                src = int(sample_idx[ri])
                if ANGLE_COL in cached:
                    azimuth[ri] = cached[ANGLE_COL][src]
                for k, col in enumerate(CONT_COLS):
                    if col in cached:
                        cont_raw[ri, k] = cached[col][src]

        np.savez(
            cache_path,
            shard_keys=shard_keys,
            shard_dirs=np.array(unique_shards, dtype=object),
            sample_idx=sample_idx,
            layer_idx=layer_idx,
            cont_raw=cont_raw,
            azimuth=azimuth,
        )

    # -- access -----------------------------------------------------------
    def __len__(self):
        return len(self.layer_idx)

    def _get_facies_mmap(self, shard_dir):
        m = self._mmap_cache.get(shard_dir)
        if m is None:
            # Bound the per-worker mmap pool; FIFO eviction is fine here since
            # access is shuffled and memory is the OS page cache (cheap).
            if len(self._mmap_cache) >= 64:
                self._mmap_cache.pop(next(iter(self._mmap_cache)))
            m = np.load(self.data_dir / shard_dir / 'facies.npy', mmap_mode='r')
            self._mmap_cache[shard_dir] = m
        return m

    def __getitem__(self, idx):
        shard = self.shard_dirs[int(self.shard_keys[idx])]
        i = int(self.sample_idx[idx])
        cube = np.array(self._get_facies_mmap(shard)[i], dtype=np.float32)
        cube = cube * 2.0 - 1.0  # {0, 1} -> {-1, 1}
        # HF stores cubes as (x, y, z) per README — z (depth) is the LAST
        # spatial axis. Tensor is (C, X, Y, Z); masking and Conv3d treat all
        # spatial axes symmetrically, but consumers that need depth (well-mask
        # generators, plots) MUST use axis 2 (last) for z.
        facies = torch.from_numpy(cube).unsqueeze(0)  # (1, 64, 64, 32)

        # Build conditioning vector (length COND_DIM).
        lt = int(self.layer_idx[idx])
        onehot = np.zeros(NUM_LAYERS, dtype=np.float32)
        onehot[lt] = 1.0

        cont = self.cont_raw[idx]
        cont_norm = (cont - self.cont_min) / (self.cont_max - self.cont_min + 1e-8)
        cont_norm = np.where(np.isnan(cont_norm), 0.0, cont_norm).astype(np.float32)

        az_norm = float(self.azimuth[idx]) / 360.0
        sin_a = np.float32(np.sin(2.0 * np.pi * az_norm))
        cos_a = np.float32(np.cos(2.0 * np.pi * az_norm))

        cond = np.concatenate([
            onehot,                              # 8: layer-type one-hot
            cont_norm[:len(UNIVERSAL_CONT)],     # 3: ntg, width_cells, depth_cells
            np.array([sin_a, cos_a], dtype=np.float32),
            cont_norm[len(UNIVERSAL_CONT):],     # 5: family-specific (0 if missing)
        ])
        return facies, torch.from_numpy(cond)


VOLUME_SHAPE = (64, 64, 32)  # (X, Y, Z); z is the LAST axis (depth/elevation)


def get_reservoir_loaders(data_dir, batch_size=32, num_workers=4, seed=42):
    """Train / val / test DataLoaders sharing one normalization scheme."""
    train_set = ReservoirDataset(data_dir, split='train')
    val_set = ReservoirDataset(data_dir, split='val',
                               cont_min=train_set.cont_min,
                               cont_max=train_set.cont_max)
    test_set = ReservoirDataset(data_dir, split='test',
                                cont_min=train_set.cont_min,
                                cont_max=train_set.cont_max)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, generator=g,
                              persistent_workers=num_workers > 0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=num_workers > 0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader, test_loader, train_set


def get_reservoir_inpaint_loaders(data_dir, batch_size=32, num_workers=4, seed=42):
    """Train / val / test inpaint DataLoaders. Each yields (facies, cond, mask)."""
    from resflow.utils.masking import InpaintDataset

    train_set = ReservoirDataset(data_dir, split='train')
    val_set = ReservoirDataset(data_dir, split='val',
                               cont_min=train_set.cont_min, cont_max=train_set.cont_max)
    test_set = ReservoirDataset(data_dir, split='test',
                                cont_min=train_set.cont_min, cont_max=train_set.cont_max)

    train_inp = InpaintDataset(train_set, volume_shape=VOLUME_SHAPE)
    val_inp = InpaintDataset(val_set, volume_shape=VOLUME_SHAPE)
    test_inp = InpaintDataset(test_set, volume_shape=VOLUME_SHAPE)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_inp, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, generator=g,
                              persistent_workers=num_workers > 0, pin_memory=True)
    val_loader = DataLoader(val_inp, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=num_workers > 0, pin_memory=True)
    test_loader = DataLoader(test_inp, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader, test_loader, train_set
