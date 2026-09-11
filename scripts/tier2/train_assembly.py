"""Assembly-aware lobe training (Tier-2 MultiDiffusion fix).

Fork of scripts/specialists/train_specialist.py. That file is left
untouched so the published specialist runs stay reproducible; every
knob not listed below is copied from it verbatim (UNet3D, FlowMatching
with drop_prob 0.1, global batch 384 via accumulation, peak LR
1e-3*sqrt(12), warmup max(1, epochs//20) -> cosine, EMA 0.9999,
foundation cond_stats.npz, checkpoint/resume semantics).

Two variables under test, selected by --data-mode, against the
80-epoch lobe specialist as the control:

  native64  -- ORIGINAL 64x64x32 lobe training split, but with the
               assembly mask distribution (30% empty / 35% wells /
               35% neighbour-context slabs) instead of wells-only.
               Isolates the MASK change.
  crops192  -- same masks, but each sample is a random 64x64x32 crop of
               a 192x192x32 large-domain volume. Isolates the DATA
               change on top of native64.

So specialist -> native64 measures what context-slab training buys, and
native64 -> crops192 measures what the large-domain dataset buys. This
matters because a paired measurement (scripts/tier2/paired_marginal_check.py)
found the crop marginal differs from the native marginal by only ~1-4%
on body statistics, so the data change may contribute little -- an open
question the ablation settles rather than assumes.

Matched exposure: both modes use a 180,000-volume subset, so
steps/epoch = 468 at global batch 384, identical to the specialist's
468 -- same optimizer-step count at the same batch and schedule.
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
from resflow.models.dit3d import DiT3D                      # noqa: E402
from resflow.methods.flow_matching import FlowMatching      # noqa: E402
from resflow.utils.data_reservoirs import (                 # noqa: E402
    COND_DIM, LAYER_TYPE_TO_IDX, VOLUME_SHAPE, ReservoirDataset,
)
from resflow.utils.data_lobe_crops import (                 # noqa: E402
    AssemblyInpaintDataset, LobeCropDataset,
)
from resflow.utils.masking_context import noisy_context       # noqa: E402
from resflow.utils.training import (                        # noqa: E402
    EMA, _make_scheduler, _strip_module_prefix,
)

FOUNDATION_CKPT_DIR = os.path.join(
    os.environ.get('WORK', '.'),
    'genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')

# Matched to the lobe specialist's subset_size so steps/epoch agree.
SUBSET_SIZE = 180000
SUBSET_SEED = 20260801


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
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_training_state(path, epoch, model, optimizer, scheduler, ema,
                        epoch_losses):
    torch.save({
        'epoch': epoch,
        'model_state_dict': _strip_module_prefix(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_shadow': {(k[7:] if k.startswith('module.') else k): v.cpu()
                       for k, v in ema.shadow.items()},
        'epoch_losses': epoch_losses,
    }, path)


def save_inference_checkpoint(model, ema, path):
    """EMA-applied weights for inference (backup/restore live weights).

    Copied verbatim from train_specialist.py: resflow's EMA exposes
    apply() and has no restore(), so the live weights are snapshotted and
    reloaded around it.
    """
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    torch.save(_strip_module_prefix(model.state_dict()), path)
    model.load_state_dict(backup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-mode', required=True,
                    choices=['native64', 'crops192'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--total-epochs', type=int, default=None)
    ap.add_argument('--micro-batch', type=int, default=96)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--global-batch', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-3 * math.sqrt(12))
    ap.add_argument('--ema-decay', type=float, default=0.9999)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--data-dir', default=os.environ.get(
        'RESERVOIR_DATA_DIR',
        os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs')))
    ap.add_argument('--data-dir-192', default=os.path.join(
        os.environ.get('SCRATCH', '.'), 'resmill_lobes_192'))
    ap.add_argument('--foundation-stats',
                    default=os.path.join(FOUNDATION_CKPT_DIR, 'cond_stats.npz'))
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--loader-seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--smoke-subset', type=int, default=None)
    ap.add_argument('--arch', default='unet', choices=['unet', 'unet_attn',
                                                       'dit'])
    ap.add_argument('--masked-loss', action='store_true',
                    help='weight the FM loss to UNKNOWN voxels only. At '
                         'overlap 24 a context slab covers 61%% of a block, '
                         'so an unweighted loss is dominated by copying '
                         'known values.')
    ap.add_argument('--amp', default='off', choices=['off', 'bf16'],
                    help='bf16 autocast for conv/matmul. Master weights, '
                         'optimizer state, EMA and the loss reduction stay '
                         'fp32, so no GradScaler is needed. Measured 2.1x '
                         'on UNet3D and 3.1x on DiT3D with the loss '
                         'unchanged to 4 decimals.')
    ap.add_argument('--augment', action='store_true',
                    help='8x dihedral augmentation of the horizontal plane '
                         'with the azimuth conditioning rotated to match')
    ap.add_argument('--config-set', default='all6',
                    choices=['raster3', 'all6'],
                    help="which neighbour configurations to train on. "
                         "'raster3' covers raster order only and spends the "
                         "budget on three configs instead of six.")
    ap.add_argument('--context-share', type=float, default=0.5,
                    help='share of the CONDITIONED half that is context '
                         'slabs; the rest is wells')
    ap.add_argument('--traj-prob', type=float, default=0.0,
                    help='fraction of samples whose context is presented at '
                         'a random noise level s~U(0,1) instead of clean. '
                         '>0 switches the model to 4 input channels and '
                         'enables the fully parallel coupled sampler; s=1 '
                         '(clean) still covers the sequential schedulers.')
    ap.add_argument('--unet-dims', type=int, nargs='+', default=None,
                    help='UNet3D hidden_dims (default [64,64,128,128], 5.4M). '
                         'Widening tests whether the remaining assembly gap '
                         'is a capacity limit rather than a receptive-field '
                         'one.')
    ap.add_argument('--attn-levels', type=int, default=0,
                    help='extra encoder/decoder stages (deepest first) that '
                         'get attention, on top of the mid block')
    ap.add_argument('--dit-patch', type=int, nargs=3, default=[8, 8, 4])
    ap.add_argument('--dit-hidden', type=int, default=384)
    ap.add_argument('--dit-depth', type=int, default=12)
    ap.add_argument('--dit-heads', type=int, default=6)
    ap.add_argument('--dit-conv-io', nargs='?', const='refine', default='off',
                    choices=['off', 'up', 'refine'],
                    help="conv stem plus: 'up' = conv upsampling head in "
                         "place of the linear unpatchify (learned far too "
                         "slowly), 'refine' = linear unpatchify followed by "
                         "a residual full-resolution conv refinement that "
                         "also sees the input channels. Either removes the "
                         "patch-tile artefact of the plain linear decoder.")
    ap.add_argument('--dit-no-qk-norm', action='store_true',
                    help='disable per-head RMSNorm on q and k (the '
                         'first-generation DiT3D behaviour, which NaN\'d '
                         'or diverged in every 2026-08-02 run)')
    ap.add_argument('--weight-decay', type=float, default=1e-2,
                    help='AdamW decoupled weight decay (1e-2 is the '
                         'torch default every earlier run used)')
    ap.add_argument('--beta2', type=float, default=0.999,
                    help='AdamW beta2. 0.95 damps the loss spikes a slow '
                         'second-moment estimate produces after a rare '
                         'large gradient (Wortsman et al. 2023).')
    ap.add_argument('--ema-warmup', action='store_true',
                    help='ramp the EMA decay as min(ema_decay, (1+k)/(10+k)) '
                         'over optimizer steps k. At a fixed 0.9999 the '
                         'inference checkpoint of a 37k-step run still holds '
                         '2.4%% of the INITIAL weights at epoch 80 and 31%% '
                         'at epoch 25, which is what made every DiT val '
                         'sweep look far worse than its training loss.')
    ap.add_argument('--save-raw', action='store_true',
                    help='also save the non-EMA weights at every checkpoint '
                         'as raw_epochNNN.pt')
    ap.add_argument('--no-skip-nonfinite', action='store_true',
                    help='by default an optimizer step whose clipped '
                         'gradient norm is not finite is skipped, so one '
                         'bad micro-batch cannot poison the weights for '
                         'the rest of the run')
    args = ap.parse_args()
    total_epochs = args.total_epochs or args.epochs

    rank = int(os.environ.get('RANK', -1))
    ddp = rank >= 0
    if ddp:
        dist.init_process_group(backend='nccl')
        world = dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    else:
        rank, world = 0, 1
    assert args.global_batch % world == 0
    per_rank_batch = args.global_batch // world
    assert per_rank_batch % args.micro_batch == 0
    accum = per_rank_batch // args.micro_batch

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed + rank)

    stats = np.load(args.foundation_stats, allow_pickle=True)
    subset_n = args.smoke_subset or SUBSET_SIZE

    if args.data_mode == 'native64':
        base = ReservoirDataset(args.data_dir, split='train',
                                cont_min=stats['cont_min'],
                                cont_max=stats['cont_max'], download=False)
        idx_all = np.where(base.layer_idx == LAYER_TYPE_TO_IDX['lobe'])[0]
        indices = idx_all[:subset_n]
        source = Subset(base, indices.tolist())
        n_source = len(source)
    else:
        full = LobeCropDataset(args.data_dir_192, stats['cont_min'],
                               stats['cont_max'], seed=args.loader_seed)
        # Deterministic held-out split: the tail 20,000 volumes are never
        # trained on, so a like-for-like validation set exists for the
        # crops arm.
        perm = np.random.default_rng(SUBSET_SEED).permutation(len(full))
        source = LobeCropDataset(args.data_dir_192, stats['cont_min'],
                                 stats['cont_max'], seed=args.loader_seed,
                                 indices=perm[:subset_n],
                                 augment=args.augment)
        n_source = len(source)
        del full

    train_set = AssemblyInpaintDataset(source, volume_shape=VOLUME_SHAPE,
                                      traj_prob=args.traj_prob,
                                      context_share=args.context_share,
                                      config_set=args.config_set)

    # Crops redraw their origin per epoch via set_epoch on the dataset
    # object. Persistent workers would keep a stale copy from epoch 0 and
    # silently freeze every crop for the whole run, so they are disabled
    # in that mode; workers re-fork each epoch from the updated parent.
    persistent = args.num_workers > 0 and args.data_mode == 'native64'
    if ddp:
        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank,
                                     shuffle=True, seed=args.loader_seed,
                                     drop_last=True)
        loader = DataLoader(train_set, batch_size=per_rank_batch,
                            sampler=sampler, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True,
                            persistent_workers=persistent)
    else:
        sampler = None
        g = torch.Generator().manual_seed(args.loader_seed)
        loader = DataLoader(train_set, batch_size=args.global_batch,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True, generator=g, pin_memory=True,
                            persistent_workers=persistent)
    steps_per_epoch = len(loader)
    if rank == 0:
        print(f'mode={args.data_mode}  subset={n_source}  '
              f'global_batch={args.global_batch} '
              f'(world={world} x {args.micro_batch}x{accum})  '
              f'steps/epoch={steps_per_epoch}  peak_lr={args.lr:.6e}  '
              f'epochs={args.epochs}/{total_epochs}', flush=True)

    in_ch = 4 if args.traj_prob > 0 else 3
    if args.arch == 'dit':
        raw_model = DiT3D(in_channels=in_ch, out_channels=1,
                          volume_shape=VOLUME_SHAPE, num_cond=COND_DIM,
                          patch_size=tuple(args.dit_patch),
                          hidden=args.dit_hidden, depth=args.dit_depth,
                          num_heads=args.dit_heads,
                          num_time_embs=1, expand_angle_idx=None,
                          qk_norm=not args.dit_no_qk_norm,
                          conv_io=(False if args.dit_conv_io == 'off'
                                   else args.dit_conv_io)).to(device)
    else:
        raw_model = UNet3D(in_channels=in_ch, out_channels=1,
                           num_cond=COND_DIM,
                           hidden_dims=args.unet_dims,
                           num_time_embs=1, expand_angle_idx=None,
                           attention=(args.arch == 'unet_attn'),
                           attn_levels=args.attn_levels).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if rank == 0:
        print(f'arch={args.arch}  in_ch={in_ch}  '
              f'params={n_params/1e6:.2f}M  masked_loss={args.masked_loss}  '
              f'traj_prob={args.traj_prob}', flush=True)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                  betas=(0.9, args.beta2),
                                  weight_decay=args.weight_decay)
    scheduler = _make_scheduler(optimizer, total_epochs, world_size=1)
    method = FlowMatching(raw_model)

    manifest_path = run_dir / 'run_manifest.json'
    manifest = {
        'protocol': 'ResBench EVAL.md Addendum G (assembly-aware training)',
        'argv': sys.argv,
        'resolved': {
            'data_mode': args.data_mode, 'subset_size': n_source,
            'steps_per_epoch': steps_per_epoch,
            'global_batch': args.global_batch, 'world_size': world,
            'per_rank_batch': per_rank_batch, 'micro_batch': args.micro_batch,
            'accum': accum, 'peak_lr': args.lr, 'ema_decay': args.ema_decay,
            'epochs': args.epochs, 'total_epochs': total_epochs,
            'warmup_epochs': max(1, total_epochs // 20),
            'cosine_T_max': total_epochs - max(1, total_epochs // 20),
            'seed': args.seed, 'loader_seed': args.loader_seed,
            'subset_seed': SUBSET_SEED,
            'save_every': args.save_every, 'smoke_subset': args.smoke_subset,
            'drop_prob': method.drop_prob,
            'mask_mix': '30% empty / 35% wells / 35% context slab',
            'arch': args.arch, 'n_params': n_params,
            'masked_loss': args.masked_loss, 'amp': args.amp,
            'attn_levels': args.attn_levels, 'traj_prob': args.traj_prob,
            'config_set': args.config_set, 'augment': args.augment,
            'context_share': args.context_share,
            'in_channels': in_ch,
            'unet_dims': args.unet_dims,
            'dit_patch': list(args.dit_patch), 'dit_hidden': args.dit_hidden,
            'dit_depth': args.dit_depth, 'lr_arg': args.lr,
            'dit_heads': args.dit_heads,
            'dit_qk_norm': not args.dit_no_qk_norm,
            'dit_conv_io': args.dit_conv_io,
            'weight_decay': args.weight_decay, 'beta2': args.beta2,
            'skip_nonfinite': not args.no_skip_nonfinite,
            'ema_warmup': args.ema_warmup, 'save_raw': args.save_raw,
            'context_overlap_range': [8, 32],
            'data_dir': (args.data_dir if args.data_mode == 'native64'
                         else args.data_dir_192),
            'foundation_stats': args.foundation_stats,
            'foundation_stats_md5': md5(args.foundation_stats),
        },
        'versions': {'python': sys.version.split()[0],
                     'torch': torch.__version__,
                     'cuda_device': torch.cuda.get_device_name(0)},
        'git': {'ResFlow': git_head(REPO)},
        'node': os.uname().nodename,
        'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'checkpoints': {},
    }
    if rank == 0:
        manifest_path.write_text(json.dumps(manifest, indent=2))

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
            print(f'Resumed from epoch {start_epoch}', flush=True)

    model = DistributedDataParallel(raw_model) if ddp else raw_model
    method.model = model
    class EMAWarmup(EMA):
        """EMA whose decay ramps from 0 toward `decay` over steps, so the
        shadow never carries a memory of the initialisation."""
        def __init__(self, model, decay, step=0):
            super().__init__(model, decay)
            self.max_decay, self.step = decay, step
        @torch.no_grad()
        def update(self, model):
            self.step += 1
            self.decay = min(self.max_decay, (1 + self.step) / (10 + self.step))
            super().update(model)

    if args.ema_warmup:
        ema = EMAWarmup(model, args.ema_decay, step=start_epoch * steps_per_epoch)
    else:
        ema = EMA(model, decay=args.ema_decay)
    if resume_shadow is not None:
        for name in ema.shadow:
            clean = name[7:] if name.startswith('module.') else name
            if clean in resume_shadow:
                ema.shadow[name] = resume_shadow[clean].to(device)
        del resume_shadow

    model.train()
    mb = args.micro_batch
    skip_nonfinite = not args.no_skip_nonfinite
    n_skipped = 0
    for epoch in range(args.epochs - start_epoch):
        global_epoch = start_epoch + epoch + 1
        if ddp:
            sampler.set_epoch(global_epoch)
        train_set.set_epoch(global_epoch)
        t0 = time.time()
        total_loss, num_steps = 0.0, 0
        for x, cond, mask, s_ctx in loader:
            optimizer.zero_grad()
            step_loss = 0.0
            for j in range(accum):
                xs = x[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                cs = cond[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                ms = mask[j * mb:(j + 1) * mb].to(device, non_blocking=True)
                if in_ch >= 4:
                    ss = s_ctx[j * mb:(j + 1) * mb].to(device).view(-1)
                    # Context at its own noise level along the same linear
                    # flow-matching path, so no ODE integration is needed
                    # to synthesize a partially denoised neighbour.
                    ctx = noisy_context(xs, ss)
                    raw_model.set_inpaint_context(ms, ctx * ms, ctx_level=ss)
                else:
                    raw_model.set_inpaint_context(ms, xs * ms)
                lw = (1.0 - ms) if args.masked_loss else None
                with torch.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(args.amp == 'bf16')):
                    loss = method.compute_loss(xs, cs, loss_weight=lw) / accum
                loss.backward()
                step_loss += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=1.0)
            # DDP has already all-reduced the gradients, so every rank
            # sees the same norm and takes the same branch.
            if skip_nonfinite and not torch.isfinite(gnorm):
                optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                if rank == 0:
                    print(f'epoch {global_epoch} step {num_steps + 1}: '
                          f'non-finite grad norm, step skipped '
                          f'(total skipped {n_skipped})', flush=True)
            else:
                optimizer.step()
                ema.update(model)
                total_loss += step_loss
            num_steps += 1
            if rank == 0 and num_steps % args.log_every == 0:
                sps = num_steps * args.global_batch / (time.time() - t0)
                print(f'epoch {global_epoch} step {num_steps}/'
                      f'{steps_per_epoch} loss {step_loss:.4f} '
                      f'gnorm {float(gnorm):.3f} '
                      f'lr {optimizer.param_groups[0]["lr"]:.3e} '
                      f'{sps:.0f} samples/s '
                      f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G',
                      flush=True)
        scheduler.step()
        epoch_losses.append(total_loss / max(num_steps, 1))
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
                if args.save_raw:
                    torch.save(_strip_module_prefix(model.state_dict()),
                               ckpt_dir / f'raw_epoch{global_epoch:03d}.pt')
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
        manifest['steps_skipped_nonfinite'] = n_skipped
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print('Done.', flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
