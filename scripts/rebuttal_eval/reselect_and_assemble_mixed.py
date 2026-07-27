"""Addendum C.7: final mixed-tier assembly with most-matches re-selection.

For every environment: mine ALL stored pools for its mixed condition's row
(Addendum A ensembles, Addendum B ensembles, C.6 top-up pools). If the
frozen pattern's total match count is below the 50-realization floor, the
pattern is RE-SELECTED as the pool-wide most-matched mixed pattern
(>= 4 voxels of each facies; ties broken by total count then lexicographic
byte order — deterministic). Original frozen pattern kept as provenance.

Writes:
  mixed_conditions.json        updated (reselected flag, final patterns)
  well_conditional_topup/<slug>/<slug>_mixed.npz   ALL matches, merged
  well_conditional/<slug>_mixed_existing.npz       emptied (avoids double
                                                   count in the analysis merge)
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

WELL_X, WELL_Y = 32, 32
FLOOR = 50


def pools_for(root, slug, ridx):
    for sub in ('entropy_ref', 'entropy_ref_uncond', 'entropy_topup_draws'):
        d = root / sub / slug
        if d.is_dir():
            for f in sorted(d.glob(f'cond_r{ridx:04d}_*.npz')):
                yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True)
    args = ap.parse_args()
    root = Path(args.eval_dir)
    conds = json.loads((root / 'mixed_conditions.json').read_text())

    for rec in conds:
        slug = rec['environment'].replace(':', '_')
        ridx = int(rec['row_index'])
        cols_all, vols_all = [], []
        for f in pools_for(root, slug, ridx):
            v = np.load(f, allow_pickle=True)['volumes']
            vols_all.append(v)
            cols_all.append(v[:, WELL_X, WELL_Y, :])
        vols = np.concatenate(vols_all)
        cols = np.concatenate(cols_all)

        frozen = np.array(rec['pattern'], np.int8)
        n_frozen = int((cols == frozen).all(1).sum())
        if n_frozen >= FLOOR:
            pat, n, reselected = frozen, n_frozen, False
        else:
            mixed = cols[(cols.sum(1) >= 4) & (cols.sum(1) <= 28)]
            cnt = collections.Counter(map(bytes, mixed))
            top = max(cnt.values()) if cnt else 0
            cands = sorted(p for p, c in cnt.items() if c == top)
            pat = np.frombuffer(cands[0], np.int8)
            n, reselected = int(top), True
            rec['frozen_pattern'] = rec['pattern']
            rec['pattern'] = [int(b) for b in pat]
        rec['reselected'] = reselected
        rec['n_final'] = n
        rec['n_pool_total'] = int(len(cols))
        rec['n_existing'] = 0   # everything merged below; existing emptied

        keep = vols[(cols == pat).all(1)]
        assert len(keep) == n
        d = root / 'well_conditional_topup' / slug
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f'{slug}_mixed.npz', volumes=keep,
                            n_pool_draws=len(cols))
        np.savez_compressed(root / 'well_conditional' / f'{slug}_mixed_existing.npz',
                            volumes=np.empty((0, 64, 64, 32), np.int8))
        print(f"{rec['environment']:26s} pool={len(cols):6d} "
              f"{'RESELECTED' if reselected else 'frozen ok '} "
              f"n={n:4d} well-NTG={pat.mean():.2f}"
              f"{'  BELOW FLOOR' if n < FLOOR else ''}", flush=True)

    (root / 'mixed_conditions.json').write_text(json.dumps(conds, indent=2))
    print('-> mixed_conditions.json updated')


if __name__ == '__main__':
    main()
