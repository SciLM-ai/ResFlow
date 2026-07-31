"""Addendum F.4: memorization/novelty check generations.

64 training-split rows per environment (rng [20260812, env_index] over the
env's rows sorted by (shard_dir, sample_idx)); one Table 6 generation per
row (empty mask), noise seed drawn from the same rng. Saves per env:
ids, train_volumes (stored dataset facies), gen_volumes, noise_seeds.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from resflow.utils.data_reservoirs import (            # noqa: E402
    LAYER_TYPES, ReservoirDataset, _read_parquet,
)
from generate_ensembles import (                       # noqa: E402
    DEFAULT_CKPT_DIR, N_STEPS, CFG, VOLUME_SHAPE, UNet3D,
    euler_cfg_sample, seeded_noise,
)

SEED_BASE = 20260812
N_PER_ENV = 64

DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--cond-stats',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'cond_stats.npz'))
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    stats = np.load(args.cond_stats, allow_pickle=True)
    data_dir = Path(args.data_dir)
    split = _read_parquet(data_dir / 'splits' / 'train.parquet').to_pandas()
    split['ds_index'] = np.arange(len(split))
    split = split.sort_values(['layer_type', 'shard_dir', 'sample_idx'],
                              kind='mergesort').reset_index(drop=True)
    train_set = ReservoirDataset(str(data_dir), split='train',
                                 cont_min=stats['cont_min'],
                                 cont_max=stats['cont_max'], download=False)

    device = 'cuda'
    model = UNet3D(in_channels=3, out_channels=1, num_cond=18,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=True))
    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ei, lt in enumerate(LAYER_TYPES):
        rng = np.random.default_rng([SEED_BASE, ei])
        env = split[split['layer_type'] == lt].reset_index(drop=True)
        pick = np.sort(rng.choice(len(env), size=N_PER_ENV, replace=False))
        sel = env.iloc[pick].reset_index(drop=True)
        noise_seeds = rng.integers(0, 2**31 - 1, size=N_PER_ENV)

        train_vols = np.empty((N_PER_ENV, *VOLUME_SHAPE), np.int8)
        conds = []
        for i, r in sel.iterrows():
            m = np.load(data_dir / r['shard_dir'] / 'facies.npy', mmap_mode='r')
            train_vols[i] = m[int(r['sample_idx'])]
            conds.append(train_set[int(r['ds_index'])][1].numpy())
        cond = torch.from_numpy(np.stack(conds)).to(device)
        x0 = seeded_noise(noise_seeds, (1, *VOLUME_SHAPE)).to(device)
        shape = (N_PER_ENV, 1, *VOLUME_SHAPE)
        with torch.no_grad():
            model.set_inpaint_context(torch.zeros(shape, device=device),
                                      torch.zeros(shape, device=device))
            x = euler_cfg_sample(model, x0, cond, cfg_scale=CFG,
                                 n_steps=N_STEPS)
        model.clear_inpaint_context()
        ids = [f"{lt}|{r['shard_dir']}|{int(r['sample_idx'])}"
               for _, r in sel.iterrows()]
        np.savez_compressed(
            out / f"{lt.replace(':', '_')}.npz",
            ids=np.array(ids), train_volumes=train_vols,
            gen_volumes=(x[:, 0] > 0).to(torch.int8).cpu().numpy(),
            noise_seeds=noise_seeds)
        print(f'{lt}: 64 train rows + 64 generations '
              f'({time.time() - t0:.0f}s)', flush=True)
    print('done')


if __name__ == '__main__':
    main()
