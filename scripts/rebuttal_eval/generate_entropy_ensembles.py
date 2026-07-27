"""Entropy-calibration model ensembles (EVAL.md addendum A, part a).

For each environment, take the ensemble-(b) manifest rows with
row_index 0..3 (well configs cycle 1well/2wells/3wells/1well) and generate
--samples-per-condition (default 128) well-conditioned samples per condition
with Table 6 settings. Per-sample noise follows the frozen rule:
torch CPU seed = fresh_noise_seed * 1000 + k, k = 0..K-1.

Output: one npz per condition with keys `volumes` (K, X, Y, Z) int8 (hard
replacement applied), `mask` (X, Y, Z) uint8, `ref_id`, `well_config`.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_ensembles import (
    DEFAULT_CKPT_DIR, N_STEPS, CFG, VOLUME_SHAPE, UNet3D,
    apply_inpaint_output, build_well_mask, euler_cfg_sample, load_reference,
    seeded_noise,
)

N_CONDS_PER_ENV = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--conds', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--samples-per-condition', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--rows', default=f'0:{N_CONDS_PER_ENV}',
                    help='manifest row_index range LO:HI per environment')
    ap.add_argument('--empty-mask', action='store_true',
                    help='Addendum B: unconditional-on-wells ensembles '
                         '(no mask, no hard replacement)')
    args = ap.parse_args()
    row_lo, row_hi = (int(x) for x in args.rows.split(':'))

    device = 'cuda'
    model = UNet3D(in_channels=3, out_channels=1, num_cond=18,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=True))
    model.eval()

    mf = pd.read_csv(args.manifest, keep_default_na=False)
    cz = np.load(args.conds, allow_pickle=True)
    cond_by_id = {i: c for i, c in zip(cz['ids'], cz['cond'])}

    out = Path(args.out_dir)
    K = args.samples_per_condition
    t0 = time.time()
    n_conds = 0
    for lt in mf['environment'].unique():
        slug = lt.replace(':', '_')
        refs = None if args.empty_mask else load_reference(args.ref_dir, slug)
        grp = mf[(mf['environment'] == lt)
                 & (mf['row_index'] >= row_lo)
                 & (mf['row_index'] < row_hi)].sort_values('row_index')
        for _, r in grp.iterrows():
            if args.empty_mask:
                mask1 = torch.zeros(1, 1, *VOLUME_SHAPE)[0]
                known1 = torch.zeros_like(mask1)
            else:
                mask1 = build_well_mask(r['well_config'])        # (1, X, Y, Z)
                ref = refs[r['row_id']].astype(np.float32) * 2.0 - 1.0
                known1 = torch.from_numpy(ref).unsqueeze(0) * mask1
            cond1 = torch.from_numpy(cond_by_id[r['row_id']])

            vols = []
            for lo in range(0, K, args.batch_size):
                B = min(args.batch_size, K - lo)
                x0 = seeded_noise(
                    [int(r['fresh_noise_seed']) * 1000 + k
                     for k in range(lo, lo + B)],
                    (1, *VOLUME_SHAPE)).to(device)
                mask = mask1.expand(B, -1, -1, -1, -1).to(device)
                known = known1.expand(B, -1, -1, -1, -1).to(device)
                cond = cond1.expand(B, -1).to(device)
                with torch.no_grad():
                    model.set_inpaint_context(mask, known)
                    x = euler_cfg_sample(model, x0, cond, cfg_scale=CFG,
                                         n_steps=N_STEPS)
                    if not args.empty_mask:
                        x = apply_inpaint_output(x, mask, known)
                model.clear_inpaint_context()
                vols.append((x[:, 0] > 0).to(torch.int8).cpu().numpy())

            cfg_name = 'nowells' if args.empty_mask else r['well_config']
            d = out / slug
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / f"cond_r{int(r['row_index']):04d}_{cfg_name}.npz",
                volumes=np.concatenate(vols),
                mask=mask1[0].numpy().astype(np.uint8),
                ref_id=r['row_id'],
                well_config=cfg_name)
            n_conds += 1
            print(f"{lt} r{int(r['row_index'])} {cfg_name}: {K} samples "
                  f"({time.time() - t0:.0f}s)", flush=True)

    (out / 'entropy_gen_manifest.json').write_text(json.dumps({
        'n_conditions': n_conds, 'samples_per_condition': K,
        'n_steps': N_STEPS, 'cfg_scale': CFG, 'ckpt': args.ckpt,
        'noise_rule': 'torch CPU seed = fresh_noise_seed*1000 + k',
        'wall_clock_s': round(time.time() - t0, 1)}, indent=2))
    print('entropy ensembles done')


if __name__ == '__main__':
    main()
