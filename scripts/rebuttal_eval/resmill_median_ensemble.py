"""Engine reference for the assembly-consistency benchmark (EVAL.md Addendum E.3 C).

Selects the training-split PV_SHOESTRING instance nearest the per-type slim
medians (E.2), then generates 256 unconditional engine realizations at that
instance's full engine parameters with azimuth overridden to 95 degrees.
Engine source untouched; machinery reused from resmill_reference.py.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resmill_reference import _gen_one  # noqa: E402

ENV_CFG = {
    'pv': {'env': 'channel:PV_SHOESTRING', 'slug': 'channel_PV_SHOESTRING',
           'medians': {'ntg': 0.226, 'width_cells': 6.97, 'depth_cells': 6.5,
                       'mCHsinu': 1.575, 'mFFCHprop': 0.15,
                       'probAvulInside': 0.06},
           'index': '/scratch/08405/ilgar/ResBaselines/train_index.npz',
           'seed_base': 2026081000},  # < 2**32, see EVAL.md E.3 amendment
    'lobe': {'env': 'lobe', 'slug': 'lobe',
             'median_cols': ['ntg', 'width_cells', 'depth_cells', 'asp'],
             'index': '/scratch/08405/ilgar/ResBaselines/lobe_train_index.npz',
             'seed_base': 2026082000},  # E.7
}
AZIMUTH = 95.0
N = 256
DATA_DIR = '/scratch/08405/ilgar/SiliciclasticReservoirs'
COND_STATS = '/work/08405/ilgar/vista/genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints/cond_stats.npz'
CONT_COLS = ['ntg', 'width_cells', 'depth_cells', 'asp', 'mCHsinu',
             'mFFCHprop', 'probAvulInside', 'trunk_length_fraction']


def load_indexed(cfg):
    idx = np.load(cfg['index'], allow_pickle=True)
    return idx['shard_dir'], idx['sample_idx']


def compute_medians(cfg):
    if 'medians' in cfg:
        return cfg['medians']
    shard_dirs, sample_idxs = load_indexed(cfg)
    parts = []
    for sd in sorted(set(shard_dirs)):
        mask = shard_dirs == sd
        sl = pd.read_parquet(Path(DATA_DIR) / sd / 'params_slim.parquet')
        parts.append(sl.iloc[sample_idxs[mask]][cfg['median_cols']])
    med = pd.concat(parts).median()
    return {c: float(med[c]) for c in cfg['median_cols']}


def select_row(cfg, medians):
    stats = np.load(COND_STATS, allow_pickle=True)
    lo = dict(zip(CONT_COLS, stats['cont_min']))
    hi = dict(zip(CONT_COLS, stats['cont_max']))
    shard_dirs, sample_idxs = load_indexed(cfg)

    best = (np.inf, None)
    for sd in sorted(set(shard_dirs)):
        mask = shard_dirs == sd
        sl = pd.read_parquet(Path(DATA_DIR) / sd / 'params_slim.parquet')
        rows = sl.iloc[sample_idxs[mask]]
        d = np.zeros(len(rows))
        for c, m in medians.items():
            scale = hi[c] - lo[c]
            d += ((rows[c].to_numpy() - m) / scale) ** 2
        j = int(np.argmin(d))
        if d[j] < best[0]:
            best = (float(d[j]), (sd, int(sample_idxs[mask][j])))
    return best


def main():
    out_dir = Path(sys.argv[1])
    cfg = ENV_CFG[sys.argv[2] if len(sys.argv) > 2 else 'pv']
    SEED_BASE = cfg['seed_base']
    out_dir.mkdir(parents=True, exist_ok=True)

    medians = compute_medians(cfg)
    print('medians:', {k: round(v, 4) for k, v in medians.items()})
    dist, (sd, si) = select_row(cfg, medians)
    slim = pd.read_parquet(Path(DATA_DIR) / sd / 'params_slim.parquet').iloc[si]
    t = pq.read_table(Path(DATA_DIR) / sd / 'params.parquet')
    row = {c: t[c][si].as_py() for c in t.column_names}
    row['azimuth'] = AZIMUTH
    print(f'selected {sd}|{si} dist={dist:.5f}')
    print('slim:', {c: round(float(slim[c]), 4) for c in medians})

    t0 = time.time()
    with Pool(min(64, os.cpu_count())) as pool:
        results = pool.map(_gen_one, [(row, SEED_BASE + k) for k in range(N)])
    vols = np.stack([r[0] for r in results])
    assert vols.shape == (N, 64, 64, 32) and set(np.unique(vols)) <= {0, 1}

    env_dir = out_dir / cfg['slug']
    env_dir.mkdir(exist_ok=True)
    ids = np.array([f'assemblyref|{sd}|{si}|{k}' for k in range(N)], dtype=object)
    np.savez_compressed(env_dir / f'volumes_r0000-r{N - 1:04d}.npz',
                        ids=ids, volumes=vols)

    manifest = {
        'addendum': 'E.3 ensemble C', 'selected_instance': f'{sd}|{si}',
        'selection_distance': dist,
        'slim_params': {c: float(slim[c]) for c in medians},
        'medians_used': medians, 'env': cfg['env'],
        'azimuth_override': AZIMUTH, 'n': N, 'seed_base': SEED_BASE,
        'mean_ntg': float(vols.mean()), 'wall_s': round(time.time() - t0, 1),
        'full_row': {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                     for k, v in row.items()},
    }
    (out_dir / 'engine_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in
                      ('selected_instance', 'slim_params', 'mean_ntg', 'wall_s')}, indent=2))


if __name__ == '__main__':
    main()
