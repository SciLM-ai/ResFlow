"""Single place that maps an --arch string to a model instance.

Training, validation and generation all need to rebuild the same
architecture from a checkpoint. Before this existed each script
hardcoded UNet3D, so a DiT or attention checkpoint failed to load with a
state_dict mismatch -- silently, in the case of the post-training chain,
which then waited forever for a val_losses.json that was never written.
"""
from __future__ import annotations

import torch

from resflow.models.unet3d import UNet3D
from resflow.models.dit3d import DiT3D

ARCHES = ['unet', 'unet_attn', 'dit']


def build_model(arch, cond_dim, volume_shape=(64, 64, 32), device='cuda',
                dit_hidden=384, dit_depth=12, dit_heads=6,
                dit_patch=(8, 8, 4), dit_qk_norm=True, dit_conv_io=False,
                attn_heads=4, in_channels=3, attn_levels=0, unet_dims=None):
    if arch == 'dit':
        m = DiT3D(in_channels=in_channels, out_channels=1,
                  volume_shape=volume_shape,
                  patch_size=dit_patch, hidden=dit_hidden, depth=dit_depth,
                  num_heads=dit_heads, num_cond=cond_dim, num_time_embs=1,
                  expand_angle_idx=None, qk_norm=dit_qk_norm,
                  conv_io=dit_conv_io)
    elif arch in ('unet', 'unet_attn'):
        m = UNet3D(in_channels=in_channels, out_channels=1,
                   num_cond=cond_dim, hidden_dims=unet_dims,
                   num_time_embs=1, expand_angle_idx=None,
                   attention=(arch == 'unet_attn'), attn_heads=attn_heads,
                   attn_levels=attn_levels)
    else:
        raise ValueError(f'unknown arch {arch!r}; expected one of {ARCHES}')
    return m.to(device)


def infer_shape_from_state(state, arch):
    """Recover in_channels, attention placement and width from a checkpoint.

    Addendum-G models vary in input channels (3 = clean context only,
    4 = context carries its own noise level), in how many stages have
    attention, and in width, so tooling must read all of it back rather
    than assume the defaults.
    """
    if arch == 'dit':
        # linear patchify: patch_embed.weight; conv stem: patch_embed.conv.weight
        w = state.get('patch_embed.weight', state.get('patch_embed.conv.weight'))
        return {'in_channels': int(w.shape[1])}
    w = state['init_conv.weight']
    dims = [int(w.shape[0])]
    k = 0
    while f'downs.{k}.conv1.weight' in state:
        dims.append(int(state[f'downs.{k}.conv2.weight'].shape[0]))
        k += 1
    attn_levels = sum(1 for i in range(k)
                      if f'downs.{i}.attn.qkv.weight' in state)
    return {'in_channels': int(w.shape[1]), 'unet_dims': dims,
            'attn_levels': attn_levels}


def infer_arch_from_state(state):
    """Best-effort architecture detection from a checkpoint's keys.

    Lets tooling load a checkpoint without being told which arch it is,
    and lets a mismatch fail loudly instead of hanging downstream.
    """
    keys = set(state)
    if any(k.startswith('blocks.') and '.ada.' in k for k in keys):
        return 'dit'
    if any(k.startswith('mid_attn.') for k in keys):
        return 'unet_attn'
    if 'init_conv.weight' in keys:
        return 'unet'
    raise ValueError('could not infer architecture from checkpoint keys')


def remap_legacy_dit_state(state):
    """Translate a first-generation DiT3D checkpoint to the current keys.

    The original block used ``nn.MultiheadAttention`` (``in_proj_weight``,
    ``in_proj_bias``, ``out_proj.*``). The replacement ``Attention`` keeps
    the same stacked-qkv layout, so the tensors carry over unchanged
    under new names; such a model has no QK-norm gains and no head-count
    buffer, and is rebuilt with ``qk_norm=False``.
    """
    if not any(k.endswith('attn.in_proj_weight') for k in state):
        return state, False
    out = {}
    for k, v in state.items():
        k2 = (k.replace('attn.in_proj_weight', 'attn.qkv.weight')
               .replace('attn.in_proj_bias', 'attn.qkv.bias')
               .replace('attn.out_proj.', 'attn.proj.'))
        out[k2] = v
    return out, True


def dit_dims_from_state(state):
    """Recover (hidden, depth, patch, heads_or_None, qk_norm, conv_io) from a DiT
    checkpoint. ``heads`` is None for legacy checkpoints, which never
    stored it."""
    hidden = state['pos_embed'].shape[-1]
    depth = 1 + max(int(k.split('.')[1]) for k in state
                    if k.startswith('blocks.'))
    stem = 'patch_embed.proj.weight' in state
    conv_io = ('refine' if any(k.startswith('refine.') for k in state)
               else 'up' if any(k.startswith('head.') for k in state)
               else False)
    pw = state['patch_embed.proj.weight' if stem else 'patch_embed.weight']
    patch = tuple(int(s) for s in pw.shape[2:])   # (hidden, c, px, py, pz)
    heads = int(state['num_heads_buf']) if 'num_heads_buf' in state else None
    qk_norm = any(k.endswith('attn.q_norm.weight') for k in state)
    return hidden, depth, patch, heads, qk_norm, conv_io


def load_checkpoint(path, cond_dim, volume_shape=(64, 64, 32), device='cuda',
                    arch=None, dit_heads=6):
    """Build the right architecture for `path` and load it.

    ``dit_heads`` is only a fallback for legacy DiT checkpoints that did
    not record their head count; current ones are self-describing.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    arch = arch or infer_arch_from_state(state)
    kw = dict(infer_shape_from_state(state, arch))
    if arch == 'dit':
        state, legacy = remap_legacy_dit_state(state)
        hidden, depth, patch, heads, qk_norm, conv_io = \
            dit_dims_from_state(state)
        kw.update(dit_hidden=hidden, dit_depth=depth, dit_patch=patch,
                  dit_heads=heads if heads is not None else dit_heads,
                  dit_qk_norm=qk_norm, dit_conv_io=conv_io)
        if legacy:
            # No buffer in the file: keep it out of the strict load.
            model = build_model(arch, cond_dim, volume_shape, device, **kw)
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected and set(missing) <= {'num_heads_buf'}, \
                (missing, unexpected)
            model.eval()
            return model, arch
    model = build_model(arch, cond_dim, volume_shape, device, **kw)
    model.load_state_dict(state)
    model.eval()
    return model, arch
