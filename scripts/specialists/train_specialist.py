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
import torch.distributed as dist                            # noqa: E402
from torch.nn.parallel import DistributedDataParallel       # noqa: E402
from torch.utils.data import DataLoader, Subset             # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

from resflow.models.unet3d import UNet3D                    # noqa: E402
from resflow.methods.flow_matching import FlowMatching      # noqa: E402
from resflow.utils.data_reservoirs import (                 # noqa: E402
    COND_DIM, LAYER_TYPES, LAYER_TYPE_TO_IDX, VOLUME_SHAPE, ReservoirDataset,
)
from resflow.utils.masking import InpaintDataset            # noqa: E402
from resflow.utils.training import (                        # noqa: E402
    EMA, _make_scheduler, _strip_module_prefix,
)

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
        'model_state_dict': _strip_module_prefix(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_shadow': _strip_module_prefix(ema.shadow),
        'epoch_losses': epoch_losses,
    }, path)


def save_inference_checkpoint(model, ema, path):
    """EMA-applied weights for inference (backup/restore live weights)."""
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    torch.save(_strip_module_prefix(model.state_dict()), path)
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
    total_epochs = args.total_epochs or args.epochs

    # Optional multi-node DDP (activated by torchrun's RANK env). The global
    # batch 384 stays fixed: per-rank batch = 384/world, accumulated in
    # micro-batches. DDP's cross-rank gradient mean of per-rank means equals
    # the single-GPU accumulated mean (GroupNorm-only model), so the
    # gradient math is identical to the pre-registered configuration.
    rank = int(os.environ.get('RANK', -1))
    ddp = rank >= 0
    if ddp:
        dist.init_process_group(backend='nccl')
        world = dist.get_world_size()
    else:
        rank, world = 0, 1
    assert args.global_batch % world == 0
    per_rank_batch = args.global_batch // world
    assert per_rank_batch % args.micro_batch == 0, \
        'per-rank batch must be a multiple of micro-batch'
    accum = per_rank_batch // args.micro_batch
    if not ddp:
        assert args.micro_batch * args.accum == args.global_batch, \
            'micro-batch * accum must equal the foundation global batch'

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda'

    # Per-rank seed so FM noise/t/CFG draws are independent across ranks
    # (foundation DDP semantics); on resume the model state comes from the
    # checkpoint, and on fresh DDP starts DistributedDataParallel broadcasts
    # rank 0's weights, so init still matches the logged seed.
    torch.manual_seed(args.seed + rank)

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
    if ddp:
        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank,
                                     shuffle=True, seed=args.loader_seed,
                                     drop_last=True)
        loader = DataLoader(train_set, batch_size=per_rank_batch,
                            sampler=sampler, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    else:
        sampler = None
        g = torch.Generator().manual_seed(args.loader_seed)
        loader = DataLoader(train_set, batch_size=args.global_batch,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True, generator=g, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader)
    if rank == 0:
        print(f'env={args.env}  subset={len(subset)}  global_batch='
              f'{args.global_batch} (world={world} x {args.micro_batch}x{accum})  '
              f'steps/epoch={steps_per_epoch}  peak_lr={args.lr:.6e}  '
              f'epochs={args.epochs}/{total_epochs}', flush=True)

    # -- model / optimizer / schedule: foundation recipe -------------------
    raw_model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                       num_time_embs=1, expand_angle_idx=None).to(device)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    # world_size=1 on purpose: args.lr is already the foundation's fully
    # scaled peak LR; _make_scheduler must not sqrt-scale it again.
    scheduler = _make_scheduler(optimizer, total_epochs, world_size=1)
    method = FlowMatching(raw_model)      # drop_prob=0.1 default, as foundation

    # -- run manifest ------------------------------------------------------
    manifest_path = run_dir / 'run_manifest.json'
    manifest = {
        'protocol': 'ResBench EVAL.md Addendum E',
        'argv': sys.argv,
        'resolved': {
            'env': args.env, 'subset_size': len(subset),
            'steps_per_epoch': steps_per_epoch,
            'global_batch': args.global_batch,
            'world_size': world, 'per_rank_batch': per_rank_batch,
            'micro_batch': args.micro_batch, 'accum': accum,
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
    if rank == 0:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- auto-resume (mirrors train_model_inpaint); load BEFORE DDP wrap ---
    start_epoch = 0
    epoch_losses = []
    resume_path = ckpt_dir / 'training_state.pt'
    resume_shadow = None
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        epoch_losses = list(ckpt.get('epoch_losses', []))
        resume_shadow = ckpt['ema_shadow']
        del ckpt
        if rank == 0:
            print(f'Resumed from epoch {start_epoch}, prior loss '
                  f'{epoch_losses[-1]:.4f}', flush=True)

    model = DistributedDataParallel(raw_model) if ddp else raw_model
    method.model = model                  # forward through DDP for grad sync
    ema = EMA(model, decay=args.ema_decay)
    if resume_shadow is not None:
        for name in ema.shadow:
            clean = name[7:] if name.startswith('module.') else name
            if clean in resume_shadow:
                ema.shadow[name] = resume_shadow[clean].to(device)
        del resume_shadow

    # -- training loop: foundation loop + gradient accumulation -----------
    model.train()
    mb = args.micro_batch
    for epoch in range(args.epochs - start_epoch):
        global_epoch = start_epoch + epoch + 1
        if ddp:
            sampler.set_epoch(global_epoch)
        t0 = time.time()
        total_loss, num_steps = 0.0, 0
        for x, cond, mask in loader:
            optimizer.zero_grad()
            step_loss = 0.0
            for j in range(accum):
                xs = x[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                cs = cond[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                ms = mask[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                raw_model.set_inpaint_context(ms, xs * ms)
                loss = method.compute_loss(xs, cs) / accum
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            total_loss += step_loss
            num_steps += 1
            if rank == 0 and num_steps % args.log_every == 0:
                sps = num_steps * args.global_batch / (time.time() - t0)
                print(f'epoch {global_epoch} step {num_steps}/'
                      f'{steps_per_epoch} loss {step_loss:.4f} '
                      f'lr {optimizer.param_groups[0]["lr"]:.3e} '
                      f'{sps:.0f} samples/s '
                      f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G',
                      flush=True)
        scheduler.step()
        epoch_losses.append(total_loss / num_steps)
        if rank == 0:
            print(f'=== epoch {global_epoch}/{args.epochs} mean loss '
                  f'{epoch_losses[-1]:.4f}  {time.time()-t0:.0f}s ===',
                  flush=True)

        if global_epoch % args.save_every == 0 or global_epoch == args.epochs:
            if rank == 0:
                save_training_state(resume_path, global_epoch, model,
                                    optimizer, scheduler, ema, epoch_losses)
                inf_path = ckpt_dir / f'inference_epoch{global_epoch:03d}.pt'
                save_inference_checkpoint(model, ema, inf_path)
                manifest = json.loads(manifest_path.read_text())
                manifest['checkpoints'][inf_path.name] = md5(inf_path)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f'Saved checkpoints at epoch {global_epoch}', flush=True)
            if ddp:
                dist.barrier()

    if rank == 0:
        np.save(ckpt_dir / 'loss_history_fm.npy', np.array(epoch_losses))
        manifest = json.loads(manifest_path.read_text())
        manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print('Done.', flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
