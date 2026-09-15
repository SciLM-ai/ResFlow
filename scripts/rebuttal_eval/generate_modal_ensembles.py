"""Addendum C model ensembles: K samples conditioned on (parameters, the
modal well pattern) per environment. Noise seed = fresh_noise_seed*1000 +
500 + k (offset avoids the A/B sample streams). Well = 1-well column at
(32, 32), well data = the modal pattern from modal_conditions.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from generate_ensembles import (
    DEFAULT_CKPT_DIR, N_STEPS, CFG, VOLUME_SHAPE, UNet3D,
    apply_inpaint_output, euler_cfg_sample, seeded_noise,
)
from arch_loader import load_any                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modal-json', required=True)
    ap.add_argument('--conds', required=True)
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--samples-per-condition', type=int, default=128)
    args = ap.parse_args()

    device = 'cuda'
    # Architecture-agnostic: see arch_loader for why this must not be
    # a hardcoded UNet3D.
    model, _arch = load_any(args.ckpt, device)

    cz = np.load(args.conds, allow_pickle=True)
    cond_by_id = {i: c for i, c in zip(cz['ids'], cz['cond'])}
    conds = json.loads(Path(args.modal_json).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    K = args.samples_per_condition
    t0 = time.time()
    for rec in conds:
        slug = rec['environment'].replace(':', '_')
        tag = rec.get('tag', 'modal')
        seed_off = 500 if tag == 'modal' else 700   # disjoint noise streams
        pat = np.array(rec['pattern'], dtype=np.float32)
        mask1 = torch.zeros(1, *VOLUME_SHAPE)
        mask1[0, 32, 32, :] = 1.0
        known1 = torch.zeros(1, *VOLUME_SHAPE)
        known1[0, 32, 32, :] = torch.from_numpy(pat * 2.0 - 1.0)
        cond1 = torch.from_numpy(cond_by_id[rec['row_id']])

        B = K
        x0 = seeded_noise([rec['fresh_noise_seed'] * 1000 + seed_off + k
                           for k in range(K)], (1, *VOLUME_SHAPE)).to(device)
        mask = mask1.expand(B, -1, -1, -1, -1).to(device)
        known = known1.expand(B, -1, -1, -1, -1).to(device)
        cond = cond1.expand(B, -1).to(device)
        with torch.no_grad():
            model.set_inpaint_context(mask, known)
            x = euler_cfg_sample(model, x0, cond, cfg_scale=CFG,
                                 n_steps=N_STEPS)
            x = apply_inpaint_output(x, mask, known)
        model.clear_inpaint_context()
        np.savez_compressed(
            out / f'{slug}_{tag}.npz',
            volumes=(x[:, 0] > 0).to(torch.int8).cpu().numpy(),
            mask=mask1[0].numpy().astype(np.uint8),
            pattern=np.array(rec['pattern'], np.int8),
            ref_id=rec['row_id'])
        print(f"{rec['environment']}: {K} samples ({time.time() - t0:.0f}s)",
              flush=True)
    print('done')


if __name__ == '__main__':
    main()
