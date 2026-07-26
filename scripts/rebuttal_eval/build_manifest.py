"""Phase 2 of the ResBench rebuttal evaluation: build manifest.csv.

Samples 512 test-split instances per environment (MANIFEST_SEED), recovers the
ResMill seed + full parameter vector from the per-shard params.parquet, draws
one fresh noise seed per row, assigns ensemble-(b) well configs (row mod 3 for
the first 256 rows/env), and exports the exact conditioning vectors the model
sees (built by resflow's own ReservoirDataset, normalized with the training
cond_stats.npz stored beside the checkpoint).

Outputs in --out-dir:
  manifest.csv        one row per sampled instance (4096 rows)
  conds.npz           ids + cond (4096, 18) float32, aligned with manifest
  run_manifest.json   seeds, versions, checkpoint/cond-stats md5, git SHAs

Protocol: ResBench EVAL.md (frozen before any metric was computed).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from resflow.utils.data_reservoirs import (  # noqa: E402
    LAYER_TYPES, ReservoirDataset, _read_parquet,
)

MANIFEST_SEED = 20260726          # EVAL.md §6
N_PER_ENV = 512
N_WELL_ROWS = 256                 # first N rows/env belong to ensemble (b)
WELL_CONFIGS = ['1well', '2wells', '3wells']   # assigned by row_index % 3

DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs'))
DEFAULT_CKPT_DIR = os.path.join(
    os.environ.get('WORK', '.'),
    'genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def git_sha(repo):
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--cond-stats',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'cond_stats.npz'))
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    stats = np.load(args.cond_stats, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    assert list(stats['layer_types']) == LAYER_TYPES, \
        'cond_stats layer-type order != resflow canonical order'

    # Test split, sorted for sampling determinism independent of parquet row
    # order; the dataset row index (for cond lookup) is the pre-sort position.
    split = _read_parquet(data_dir / 'splits' / 'test.parquet').to_pandas()
    split['ds_index'] = np.arange(len(split))
    split = split.sort_values(['layer_type', 'shard_dir', 'sample_idx'],
                              kind='mergesort').reset_index(drop=True)

    rng = np.random.default_rng(MANIFEST_SEED)
    rows = []
    for lt in LAYER_TYPES:                       # canonical env order
        env = split[split['layer_type'] == lt].reset_index(drop=True)
        pick = np.sort(rng.choice(len(env), size=N_PER_ENV, replace=False))
        sel = env.iloc[pick].reset_index(drop=True)
        for i, r in sel.iterrows():
            rows.append({
                'environment': lt,
                'row_index': i,                  # 0..511 within env
                'shard_dir': r['shard_dir'],
                'sample_idx': int(r['sample_idx']),
                'ds_index': int(r['ds_index']),  # row in dataset test cache
                'row_id': f"{lt}|{r['shard_dir']}|{int(r['sample_idx'])}",
                'well_config': (WELL_CONFIGS[i % 3] if i < N_WELL_ROWS else ''),
            })
    mf = pd.DataFrame(rows)
    mf['fresh_noise_seed'] = rng.integers(0, 2**31 - 1, size=len(mf))

    # Recover ResMill seed + slim params from each touched shard's full
    # params.parquet (the resolvable pointer for the rest of the vector).
    slim_cols = ['ntg', 'requested_ntg', 'azimuth', 'width_cells',
                 'depth_cells', 'asp', 'mCHsinu', 'mFFCHprop',
                 'probAvulInside', 'trunk_length_fraction']
    for c in ['resmill_seed'] + slim_cols:
        mf[c] = np.nan
    for shard, grp in mf.groupby('shard_dir'):
        t = pq.read_table(data_dir / shard / 'params.parquet').to_pandas()
        mf.loc[grp.index, 'resmill_seed'] = \
            t['seed'].iloc[grp['sample_idx']].to_numpy()
        for c in slim_cols:
            if c in t.columns:
                mf.loc[grp.index, c] = t[c].iloc[grp['sample_idx']].to_numpy()
    mf['resmill_seed'] = mf['resmill_seed'].astype('int64')
    mf['params_pointer'] = mf['shard_dir'] + '/params.parquet#row=' + \
        mf['sample_idx'].astype(str)

    # Conditioning vectors exactly as the model sees them: built by resflow's
    # own dataset code with the training normalization stats.
    test_set = ReservoirDataset(str(data_dir), split='test',
                                cont_min=cont_min, cont_max=cont_max,
                                download=False)
    conds = np.stack([test_set[int(i)][1].numpy() for i in mf['ds_index']])
    np.savez_compressed(out / 'conds.npz',
                        ids=mf['row_id'].to_numpy(),
                        cond=conds.astype(np.float32))

    mf.to_csv(out / 'manifest.csv', index=False)

    import torch  # noqa: F401  (version logging only)
    run_manifest = {
        'manifest_seed': MANIFEST_SEED,
        'n_per_env': N_PER_ENV,
        'n_well_rows_per_env': N_WELL_ROWS,
        'well_configs_cycle': WELL_CONFIGS,
        'checkpoint': {'path': args.ckpt, 'md5': md5(args.ckpt)},
        'cond_stats': {'path': args.cond_stats, 'md5': md5(args.cond_stats)},
        'data_dir': str(data_dir),
        'git': {
            'resflow': git_sha(Path(__file__).resolve().parents[2]),
            'resbench': git_sha(Path(__file__).resolve().parents[3] / 'ResBench'),
        },
        'versions': {
            'python': sys.version.split()[0],
            'torch': __import__('torch').__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'gpu': (__import__('torch').cuda.get_device_name(0)
                if __import__('torch').cuda.is_available() else None),
    }
    (out / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2))

    print(f'manifest: {len(mf)} rows -> {out / "manifest.csv"}')
    print(mf.groupby("environment").size())
    print(f'conds: {conds.shape} -> {out / "conds.npz"}')


if __name__ == '__main__':
    main()
