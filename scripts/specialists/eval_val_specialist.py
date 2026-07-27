"""Per-checkpoint validation loss for one specialist (Addendum E.3).

Replicates examples/reservoirs/inpainting/eval_losses.py restricted to one
environment's VALIDATION split only, run over every inference_epoch*.pt in
a checkpoint dir. Same loss (FM velocity MSE, K random (t, noise) draws
per cube averaged, drop_prob 0.1, on-the-fly masks). One fixed seed
(default 20260901) is re-applied identically before each checkpoint's
pass, with num_workers=0 and shuffle=False, so every checkpoint sees the
same paired (t, noise, mask, drop) draws.

Selection rule (printed at the end): argmin val loss over checkpoints;
the anti-undertraining branch fires if the argmin is the final epoch.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from torch.utils.data import DataLoader, Subset             # noqa: E402

from resflow.models.unet3d import UNet3D                    # noqa: E402
from resflow.utils.data_reservoirs import (                 # noqa: E402
    COND_DIM, LAYER_TYPES, LAYER_TYPE_TO_IDX, VOLUME_SHAPE, ReservoirDataset,
)
from resflow.utils.masking import InpaintDataset            # noqa: E402

FOUNDATION_CKPT_DIR = os.path.join(
    os.environ.get('WORK', '.'),
    'genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')


@torch.no_grad()
def eval_ckpt(model, loader, device, k_passes, drop_prob, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model.eval()
    sum_loss, n_seen = 0.0, 0
    for x, cond, mask in loader:
        x = x.to(device); cond = cond.to(device); mask = mask.to(device)
        model.set_inpaint_context(mask, x * mask)
        B = x.shape[0]
        per_cube = torch.zeros(B, device=device, dtype=torch.float64)
        for _ in range(k_passes):
            x0 = torch.randn_like(x)
            t = torch.rand((B,), device=device)
            t_expand = t.view(-1, *([1] * (x.ndim - 1)))
            xt = (1 - t_expand) * x0 + t_expand * x
            drop_mask = torch.rand(B, device=device) < drop_prob
            v_pred = model(xt, t * 1000, cond, drop_mask=drop_mask)
            l = F.mse_loss(v_pred, x - x0, reduction='none')
            per_cube += l.flatten(1).mean(dim=1).to(torch.float64)
        sum_loss += (per_cube / k_passes).sum().item()
        n_seen += B
    model.clear_inpaint_context()
    return sum_loss / n_seen, n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, choices=LAYER_TYPES)
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--data-dir', default=os.environ.get(
        'RESERVOIR_DATA_DIR',
        os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs')))
    ap.add_argument('--foundation-stats',
                    default=os.path.join(FOUNDATION_CKPT_DIR, 'cond_stats.npz'))
    ap.add_argument('--k-passes', type=int, default=4)
    ap.add_argument('--drop-prob', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda'
    stats = np.load(args.foundation_stats, allow_pickle=True)
    base = ReservoirDataset(args.data_dir, split='val',
                            cont_min=stats['cont_min'],
                            cont_max=stats['cont_max'], download=False)
    indices = np.where(base.layer_idx == LAYER_TYPE_TO_IDX[args.env])[0]
    val_set = InpaintDataset(Subset(base, indices.tolist()),
                             volume_shape=VOLUME_SHAPE)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f'env={args.env}  val subset={len(val_set)}  K={args.k_passes}  '
          f'seed={args.seed}', flush=True)

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'inference_epoch*.pt')))
    assert ckpts, f'no inference checkpoints in {args.ckpt_dir}'
    results = {}
    for path in ckpts:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        loss, n = eval_ckpt(model, loader, device, args.k_passes,
                            args.drop_prob, args.seed)
        epoch = int(Path(path).stem.replace('inference_epoch', ''))
        results[epoch] = loss
        print(f'epoch {epoch:3d}: val loss {loss:.6f}  (n={n})', flush=True)

    best = min(results, key=results.get)
    last = max(results)
    print(f'\nBest: epoch {best} ({results[best]:.6f})   '
          f'last: epoch {last} ({results[last]:.6f})')
    print('ANTI-UNDERTRAINING: ' +
          ('FIRES (best == final epoch)' if best == last else 'does not fire'))

    out = args.out or os.path.join(args.ckpt_dir, 'val_losses.json')
    Path(out).write_text(json.dumps({
        'env': args.env, 'seed': args.seed, 'k_passes': args.k_passes,
        'n_val': len(val_set), 'val_loss_by_epoch': results,
        'best_epoch': best, 'last_epoch': last,
        'anti_undertraining_fires': best == last,
    }, indent=2))
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
