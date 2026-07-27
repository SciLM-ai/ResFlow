"""Addendum C.6: assemble mixed-tier conditional ensembles by dictionary-
mining ALL stored unconditional pools for each frozen mixed pattern.

Pools per condition (matched on environment + row_index):
  entropy_ref/          Addendum A (N = 512, rows 0-3)
  entropy_ref_uncond/   Addendum B (N = 256, rows 4-11)
  entropy_topup_draws/  C.6 stored top-up draws

Writes well_conditional_topup/<slug>/<slug>_mixed.npz (volumes = matches
from the top-up pools ONLY; the pre-existing matches are already in
well_conditional/<slug>_mixed_existing.npz, keeping the analysis merge
disjoint).
"""
import argparse
import json
from pathlib import Path

import numpy as np

WELL_X, WELL_Y = 32, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True)
    args = ap.parse_args()
    root = Path(args.eval_dir)
    conds = json.loads((root / 'mixed_conditions.json').read_text())

    for rec in conds:
        slug = rec['environment'].replace(':', '_')
        pat = np.array(rec['pattern'], np.int8)
        matches = []
        n_pool = 0
        for f in sorted((root / 'entropy_topup_draws' / slug).glob('cond_*.npz')):
            ridx = int(f.stem.split('_')[1][1:])
            if ridx != int(rec['row_index']):
                continue
            v = np.load(f, allow_pickle=True)['volumes']
            n_pool += len(v)
            keep = v[(v[:, WELL_X, WELL_Y, :] == pat).all(axis=1)]
            if len(keep):
                matches.append(keep)
        vols = (np.concatenate(matches) if matches
                else np.empty((0, 64, 64, 32), np.int8))
        d = root / 'well_conditional_topup' / slug
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f'{slug}_mixed.npz', volumes=vols,
                            n_pool_draws=n_pool)
        print(f"{rec['environment']:26s} pool={n_pool:6d} matches={len(vols):4d} "
              f"(+{rec['n_existing']} existing)", flush=True)


if __name__ == '__main__':
    main()
