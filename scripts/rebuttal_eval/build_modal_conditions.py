"""Addendum C: select modal well-pattern conditions from existing draws.

Scans all pre-existing unconditional engine ensembles (Addendum A: 512 x 4
conds/env; Addendum B: 256 x 8 conds/env), carves the 1-well column
(x, y) = (32, 32), and per environment picks the condition with the largest
modal-pattern count. Writes:

  <out>/modal_conditions.json          selection + patterns + counts
  <out>/well_conditional/<slug>_existing.npz   already-accepted volumes
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

WELL_X, WELL_Y = 32, 32

LAYER_TYPES = ['lobe', 'channel:PV_SHOESTRING', 'channel:CB_LABYRINTH',
               'channel:CB_JIGSAW', 'channel:SH_DISTAL',
               'channel:SH_PROXIMAL', 'channel:MEANDER_OXBOW', 'delta']
SLUG = {lt.replace(':', '_'): lt for lt in LAYER_TYPES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True,
                    help='resbench_eval root (entropy_ref*, manifest.csv)')
    ap.add_argument('--mixed', action='store_true',
                    help='informative-pattern tier: most frequent pattern '
                         'with >= 4 voxels of EACH facies (Addendum C.5)')
    args = ap.parse_args()
    root = Path(args.eval_dir)
    mf = pd.read_csv(root / 'manifest.csv', keep_default_na=False)
    tag = 'mixed' if args.mixed else 'modal'

    best = {}
    for sub in ('entropy_ref', 'entropy_ref_uncond'):
        for f in sorted((root / sub).glob('*/cond_*.npz')):
            v = np.load(f, allow_pickle=True)['volumes']
            cols = v[:, WELL_X, WELL_Y, :]
            if args.mixed:
                cols = cols[(cols.sum(1) >= 4) & (cols.sum(1) <= 28)]
                if not len(cols):
                    continue
            (pat, cnt), = [collections.Counter(map(bytes, cols)).most_common(1)[0]]
            lt = SLUG[f.parent.name]
            if cnt > best.get(lt, {}).get('n_existing', 0):
                ridx = int(f.stem.split('_')[1][1:])
                best[lt] = {'environment': lt, 'row_index': ridx, 'tag': tag,
                            'source': f'{sub}/{f.parent.name}/{f.name}',
                            'pattern': [int(b) for b in np.frombuffer(pat, np.int8)],
                            'n_existing': int(cnt), 'n_draws_scanned': len(v)}

    out_wc = root / 'well_conditional'
    out_wc.mkdir(exist_ok=True)
    for lt, rec in best.items():
        m = mf[(mf['environment'] == lt)
               & (mf['row_index'] == rec['row_index'])].iloc[0]
        rec['row_id'] = m['row_id']
        rec['fresh_noise_seed'] = int(m['fresh_noise_seed'])
        f = root / rec['source']
        v = np.load(f, allow_pickle=True)['volumes']
        pat = np.array(rec['pattern'], np.int8)
        keep = v[(v[:, WELL_X, WELL_Y, :] == pat).all(axis=1)]
        assert len(keep) == rec['n_existing']
        np.savez_compressed(
            out_wc / f"{lt.replace(':', '_')}_{tag}_existing.npz",
            volumes=keep)
        print(f"{lt:26s} r{rec['row_index']} {tag} count={rec['n_existing']} "
              f"well-NTG={pat.mean():.2f}")

    (root / f'{tag}_conditions.json').write_text(
        json.dumps([best[lt] for lt in LAYER_TYPES], indent=2))
    print('->', root / f'{tag}_conditions.json')


if __name__ == '__main__':
    main()
