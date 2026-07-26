"""Phase 3: extract the reference ensemble from the dataset by instance id.

Reads manifest.csv and copies the raw int8 binary-facies cubes (values {0,1},
no transform) out of the per-shard facies.npy mmaps into ResBench's volume-
directory format: reference/<env_slug>/volumes_r0000-r0511.npz with keys
`ids` and `volumes` (N, 64, 64, 32) int8.
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs'))


def env_slug(layer_type):
    return layer_type.replace(':', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    mf = pd.read_csv(args.manifest, keep_default_na=False)
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)

    for lt, grp in mf.groupby('environment', sort=False):
        grp = grp.sort_values('row_index')
        vols = np.empty((len(grp), 64, 64, 32), dtype=np.int8)
        # one mmap per shard, sequential reads grouped by shard
        by_shard = defaultdict(list)
        for pos, (_, r) in enumerate(grp.iterrows()):
            by_shard[r['shard_dir']].append((pos, int(r['sample_idx'])))
        for shard, items in by_shard.items():
            m = np.load(data_dir / shard / 'facies.npy', mmap_mode='r')
            for pos, si in items:
                vols[pos] = m[si]
        assert set(np.unique(vols)) <= {0, 1}, 'facies must be binary {0,1}'

        d = out / env_slug(lt)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f'volumes_r0000-r{len(grp) - 1:04d}.npz'
        np.savez_compressed(path,
                            ids=grp['row_id'].to_numpy(),
                            volumes=vols)
        print(f'{lt}: {vols.shape} ntg_mean={vols.mean():.4f} -> {path}')


if __name__ == '__main__':
    main()
