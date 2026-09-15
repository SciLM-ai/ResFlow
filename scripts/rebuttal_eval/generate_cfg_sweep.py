"""Post-hoc CFG sensitivity sweep (RESULTS.md item 4; NOT master-table scoring).

For lobe and channel:PV_SHOESTRING, regenerate the first 128 manifest rows
parameter-conditioned (empty well mask) at CFG scales {1.0, 1.5, 2.0, 3.0},
all other Table 6 settings fixed.

Noise rule (logged): torch CPU seed = fresh_noise_seed * 100 + int(cfg * 10)
— distinct from the ensemble-(a) rule, so these are fresh draws at every
scale including 3.0.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_ensembles import (
    DEFAULT_CKPT_DIR, N_STEPS, VOLUME_SHAPE, UNet3D,
    euler_cfg_sample, seeded_noise,
)
import os
from arch_loader import load_any                            # noqa: E402

SWEEP_ENVS = ['lobe', 'channel:PV_SHOESTRING']
CFG_SCALES = [1.0, 1.5, 2.0, 3.0]
N_ROWS = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--conds', required=True)
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    device = 'cuda'
    # Architecture-agnostic: see arch_loader for why this must not be
    # a hardcoded UNet3D.
    model, _arch = load_any(args.ckpt, device)

    mf = pd.read_csv(args.manifest, keep_default_na=False)
    cz = np.load(args.conds, allow_pickle=True)
    cond_by_id = {i: c for i, c in zip(cz['ids'], cz['cond'])}

    out = Path(args.out_dir)
    t0 = time.time()
    for cfg in CFG_SCALES:
        for lt in SWEEP_ENVS:
            grp = mf[(mf['environment'] == lt)
                     & (mf['row_index'] < N_ROWS)].sort_values('row_index')
            shape = (len(grp), 1, *VOLUME_SHAPE)
            cond = torch.from_numpy(
                np.stack([cond_by_id[i] for i in grp['row_id']])).to(device)
            x0 = seeded_noise(
                grp['fresh_noise_seed'].to_numpy() * 100 + int(cfg * 10),
                (1, *VOLUME_SHAPE)).to(device)
            with torch.no_grad():
                model.set_inpaint_context(torch.zeros(shape, device=device),
                                          torch.zeros(shape, device=device))
                x = euler_cfg_sample(model, x0, cond, cfg_scale=cfg,
                                     n_steps=N_STEPS)
            model.clear_inpaint_context()
            d = out / f'cfg_{cfg:.1f}' / lt.replace(':', '_')
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / f'volumes_r0000-r{len(grp) - 1:04d}.npz',
                ids=grp['row_id'].to_numpy(),
                volumes=(x[:, 0] > 0).to(torch.int8).cpu().numpy())
            print(f'cfg={cfg} {lt}: {len(grp)} vols '
                  f'({time.time() - t0:.0f}s)', flush=True)

    (out / 'sweep_manifest.json').write_text(json.dumps({
        'envs': SWEEP_ENVS, 'cfg_scales': CFG_SCALES, 'n_rows': N_ROWS,
        'n_steps': N_STEPS, 'ckpt': args.ckpt,
        'noise_rule': 'torch CPU seed = fresh_noise_seed*100 + int(cfg*10)',
        'wall_clock_s': round(time.time() - t0, 1)}, indent=2))
    print('sweep done')


if __name__ == '__main__':
    main()
