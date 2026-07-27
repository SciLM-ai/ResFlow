"""Single-environment specialist training (ResBench EVAL.md Addendum E).

Thin wrapper around the foundation recipe in
examples/reservoirs/inpainting/train.py — identical in every respect
except that the training set is restricted to ONE environment's training
split (Addendum E.2 single-difference principle):

  - same UNet3D / FlowMatching / mask distribution / CFG dropout,
  - foundation cond_stats.npz reused verbatim (bit-identical
    conditioning surface; subset stats would differ),
  - global batch 384 matched via gradient accumulation (micro-batch x
    accum, loss weighted 1/accum so the gradient equals the mean over
    all 384 samples — identical to the foundation's 12-rank DDP mean;
    GroupNorm-only model, so no batch-statistics coupling),
  - peak LR passed explicitly as 1e-3*sqrt(12) (the foundation's
    world-size-scaled value; world size 1 here would skip the scaling),
  - epoch-parametrized schedule (linear warmup max(1, epochs//20) ->
    cosine T_max = epochs - warmup, stepped per epoch) via the library's
    own _make_scheduler,
  - EMA decay 0.9999 updated once per OPTIMIZER step,
  - checkpointing/resume semantics copied from
    resflow.utils.training.train_model_inpaint: training_state.pt
    (auto-resume) + EMA-applied inference_epoch{NNN}.pt every
    --save-every epochs.

Resumable: rerun the same command; it picks up from
<run-dir>/checkpoints/training_state.pt. To reset, delete that file.

No file under resflow/ is modified.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from torch.utils.data import DataLoader, Subset             # noqa: E402

from resflow.models.unet3d import UNet3D                    # noqa: E402
from resflow.methods.flow_matching import FlowMatching      # noqa: E402
from resflow.utils.data_reservoirs import (                 # noqa: E402
    COND_DIM, LAYER_TYPES, LAYER_TYPE_TO_IDX, VOLUME_SHAPE, ReservoirDataset,
)
from resflow.utils.masking import InpaintDataset            # noqa: E402
from resflow.utils.training import EMA, _make_scheduler     # noqa: E402

FOUNDATION_CKPT_DIR = os.path.join(
    os.environ.get('WORK', '.'),
    'genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_head(path):
    try:
        return subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save_training_state(path, epoch, model, optimizer, scheduler, ema,
                        epoch_losses):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_shadow': ema.shadow,
        'epoch_losses': epoch_losses,
    }, path)


def save_inference_checkpoint(model, ema, path):
    """EMA-applied weights for inference (backup/restore live weights)."""
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    torch.save(model.state_dict(), path)
    model.load_state_dict(backup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, choices=LAYER_TYPES)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--seed', type=int, required=True,
                    help='global torch seed (model init); Addendum E.2')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--total-epochs', type=int, default=None,
                    help='LR-schedule horizon (default: --epochs); pass the '
                         'original total when resuming a partial segment')
    ap.add_argument('--micro-batch', type=int, default=96)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--global-batch', type=int, default=384,
                    help='must equal micro-batch * accum (foundation value)')
    ap.add_argument('--lr', type=float, default=1e-3 * math.sqrt(12),
                    help='peak LR; default = foundation effective '
                         '1e-3*sqrt(world_size=12)')
    ap.add_argument('--ema-decay', type=float, default=0.9999)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--data-dir', default=os.environ.get(
        'RESERVOIR_DATA_DIR',
        os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs')))
    ap.add_argument('--foundation-stats',
                    default=os.path.join(FOUNDATION_CKPT_DIR, 'cond_stats.npz'))
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--loader-seed', type=int, default=42,
                    help='shuffle generator seed (foundation value)')
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--smoke-subset', type=int, default=None,
                    help='SMOKE TESTS ONLY: truncate the environment subset '
                         'to the first N samples. Never used for real runs; '
                         'recorded in the run manifest.')
    args = ap.parse_args()
    assert args.micro_batch * args.accum == args.global_batch, \
        'micro-batch * accum must equal the foundation global batch'
    total_epochs = args.total_epochs or args.epochs

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda'

    torch.manual_seed(args.seed)

    # -- data: one environment's training split, foundation normalization --
    stats = np.load(args.foundation_stats, allow_pickle=True)
    base = ReservoirDataset(args.data_dir, split='train',
                            cont_min=stats['cont_min'],
                            cont_max=stats['cont_max'], download=False)
    env_idx = LAYER_TYPE_TO_IDX[args.env]
    indices = np.where(base.layer_idx == env_idx)[0]
    if args.smoke_subset:
        indices = indices[:args.smoke_subset]
    subset = Subset(base, indices.tolist())
    train_set = InpaintDataset(subset, volume_shape=VOLUME_SHAPE)
    g = torch.Generator().manual_seed(args.loader_seed)
    loader = DataLoader(train_set, batch_size=args.global_batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        generator=g, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader)
    print(f'env={args.env}  subset={len(subset)}  global_batch='
          f'{args.global_batch} ({args.micro_batch}x{args.accum})  '
          f'steps/epoch={steps_per_epoch}  peak_lr={args.lr:.6e}  '
          f'epochs={args.epochs}/{total_epochs}', flush=True)

    # -- model / optimizer / schedule: foundation recipe -------------------
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    method = FlowMatching(model)          # drop_prob=0.1 default, as foundation
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = _make_scheduler(optimizer, total_epochs, world_size=1)

    # -- run manifest ------------------------------------------------------
    manifest_path = run_dir / 'run_manifest.json'
    manifest = {
        'protocol': 'ResBench EVAL.md Addendum E',
        'argv': sys.argv,
        'resolved': {
            'env': args.env, 'subset_size': len(subset),
            'steps_per_epoch': steps_per_epoch,
            'global_batch': args.global_batch,
            'micro_batch': args.micro_batch, 'accum': args.accum,
            'peak_lr': args.lr, 'ema_decay': args.ema_decay,
            'epochs': args.epochs, 'total_epochs': total_epochs,
            'warmup_epochs': max(1, total_epochs // 20),
            'cosine_T_max': total_epochs - max(1, total_epochs // 20),
            'seed': args.seed, 'loader_seed': args.loader_seed,
            'save_every': args.save_every,
            'smoke_subset': args.smoke_subset,
            'drop_prob': method.drop_prob,
            'foundation_stats': args.foundation_stats,
            'foundation_stats_md5': md5(args.foundation_stats),
        },
        'versions': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'cuda_device': torch.cuda.get_device_name(0),
        },
        'git': {
            'ResFlow': git_head(REPO),
        },
        'node': os.uname().nodename,
        'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'checkpoints': {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- auto-resume (mirrors train_model_inpaint) -------------------------
    start_epoch = 0
    epoch_losses = []
    resume_path = ckpt_dir / 'training_state.pt'
    ema = None
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        epoch_losses = list(ckpt.get('epoch_losses', []))
        ema = EMA(model, decay=args.ema_decay)
        for name in ema.shadow:
            if name in ckpt['ema_shadow']:
                ema.shadow[name] = ckpt['ema_shadow'][name].to(device)
        del ckpt
        print(f'Resumed from epoch {start_epoch}, prior loss '
              f'{epoch_losses[-1]:.4f}', flush=True)
    else:
        ema = EMA(model, decay=args.ema_decay)

    # -- training loop: foundation loop + gradient accumulation -----------
    model.train()
    mb = args.micro_batch
    for epoch in range(args.epochs - start_epoch):
        global_epoch = start_epoch + epoch + 1
        t0 = time.time()
        total_loss, num_steps = 0.0, 0
        for x, cond, mask in loader:
            optimizer.zero_grad()
            step_loss = 0.0
            for j in range(args.accum):
                xs = x[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                cs = cond[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                ms = mask[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                model.set_inpaint_context(ms, xs * ms)
                loss = method.compute_loss(xs, cs) / args.accum
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            total_loss += step_loss
            num_steps += 1
            if num_steps % args.log_every == 0:
                sps = num_steps * args.global_batch / (time.time() - t0)
                print(f'epoch {global_epoch} step {num_steps}/'
                      f'{steps_per_epoch} loss {step_loss:.4f} '
                      f'lr {optimizer.param_groups[0]["lr"]:.3e} '
                      f'{sps:.0f} samples/s '
                      f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G',
                      flush=True)
        scheduler.step()
        epoch_losses.append(total_loss / num_steps)
        print(f'=== epoch {global_epoch}/{args.epochs} mean loss '
              f'{epoch_losses[-1]:.4f}  {time.time()-t0:.0f}s ===', flush=True)

        if global_epoch % args.save_every == 0 or global_epoch == args.epochs:
            save_training_state(resume_path, global_epoch, model, optimizer,
                                scheduler, ema, epoch_losses)
            inf_path = ckpt_dir / f'inference_epoch{global_epoch:03d}.pt'
            save_inference_checkpoint(model, ema, inf_path)
            manifest = json.loads(manifest_path.read_text())
            manifest['checkpoints'][inf_path.name] = md5(inf_path)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f'Saved checkpoints at epoch {global_epoch}', flush=True)

    np.save(ckpt_dir / 'loss_history_fm.npy', np.array(epoch_losses))
    manifest = json.loads(manifest_path.read_text())
    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
