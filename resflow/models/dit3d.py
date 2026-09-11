"""3D Diffusion Transformer (DiT) for volumetric flow matching.

Peebles & Xie's DiT adapted to (X, Y, Z) volumes, exposing exactly the
interface the methods expect — ``model(x, t)``, ``model(x, t, cond)``,
``model(x, t, cond, drop_mask=...)`` — plus the stateful inpainting
context (``set_inpaint_context`` / ``clear_inpaint_context``) that
UNet3D provides, so it is a drop-in replacement.

Motivation: UNet3D is pure convolution with no attention anywhere; its
theoretical receptive field at the bottleneck (~70 voxels) barely covers
a 64-cell block and its effective field is much smaller, so there is no
mechanism to coordinate structure at body scale. Every token here
attends to every other from the first layer.

Conditioning is adaLN-Zero: time and condition embeddings are summed and
mapped to per-block scale/shift/gate parameters, with the gates
zero-initialised so each block starts as an identity and the residual
stream is clean at init. Unconditional behaviour uses a learned null
embedding, matching UNet3D's CFG representation, so the methods' CFG
strategy is unchanged.

At 64x64x32 with patch (8, 8, 4) the token grid is 8x8x8 = 512.

Stability (2026-09-08). The first generation of this module used
``nn.MultiheadAttention`` with raw q.k logits. All five Addendum-G DiT
runs trained with it (2026-08-02) either NaN'd or diverged: at peak LR
3.46e-3 within 4 epochs, at 1e-4 after 45 epochs with the LR already
decayed to ~4e-5, and the patch-4 / 77M variants after reaching the
best training losses of any run. Divergence at 4e-5 is the signature of
attention-logit growth (Dehghani et al. 2023; Wortsman et al. 2023),
which gradient clipping cannot stop because the drift is slow and
happens in the weights, not the gradients. Fix: per-head RMSNorm on q
and k with a learnable gain (QK-norm), which caps the initial logit
scale at sqrt(head_dim) and lets the model sharpen attention through
the gain instead of through unbounded weight growth. Attention runs via
``F.scaled_dot_product_attention`` (fused kernels in bf16), and the head
count is stored in the state dict so a checkpoint is self-describing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SinusoidalPosEmb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMSNorm over the last dim, computed in fp32 regardless of autocast."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Fused kernel; under bf16 autocast it accumulates in fp32 and
        # returns bf16, bit-identical to the unfused fp32 version but ~30%
        # cheaper, which matters at 4096 tokens x 12 layers x (q, k).
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head self-attention with optional QK-norm.

    Parameter layout matches ``nn.MultiheadAttention`` (a single stacked
    qkv projection and an output projection) so legacy checkpoints can
    be remapped key-for-key; see ``scripts/tier2/model_factory.py``.
    """

    def __init__(self, hidden, num_heads, qk_norm=True):
        super().__init__()
        assert hidden % num_heads == 0, \
            f'hidden {hidden} not divisible by num_heads {num_heads}'
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # (B, H, N, d)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden, num_heads, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, num_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        # adaLN-Zero: every block is an identity at initialisation.
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift2, scale2))
        x = x + gate2.unsqueeze(1) * h
        return x


class ConvStem(nn.Module):
    """Full-resolution 3x3x3 conv mixing before the strided patch projection,
    so a token sees sub-patch structure at its borders instead of a raw
    linear fold of its own voxels."""

    def __init__(self, in_channels, hidden, patch_size, width=32):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, width, 3, padding=1)
        self.act = nn.SiLU()
        self.proj = nn.Conv3d(width, hidden, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(self.act(self.conv(x)))


class ConvHead(nn.Module):
    """Token grid -> voxels by x2 upsampling stages with 3x3x3 convs.

    Replaces the per-token linear projection (patch_vol outputs per token),
    which tiles the output by patch: with 8x8x4 patches the raw samples
    are a checkerboard of patch tiles carrying the right NTG and no body
    geometry (measured 2026-09-08: largest body 99.8% of sand, median body
    1.2 voxels, vs 41% / 731 voxels for the engine). Each stage doubles
    only the axes whose patch factor is not yet exhausted, so anisotropic
    patches such as (8, 8, 4) are handled. The last conv is zero-initialised
    to keep the adaLN-Zero identity-at-init property of the output.
    """

    def __init__(self, hidden, patch_size, out_channels=1, widths=(128, 64, 32)):
        super().__init__()
        remaining = list(patch_size)
        stages, c_in, k = [], hidden, 0
        while max(remaining) > 1:
            scale = tuple(2 if r > 1 else 1 for r in remaining)
            remaining = [r // s for r, s in zip(remaining, scale)]
            c_out = widths[min(k, len(widths) - 1)]
            stages.append(nn.ModuleDict({
                'up': nn.Upsample(scale_factor=scale, mode='trilinear',
                                  align_corners=False),
                'conv': nn.Conv3d(c_in, c_out, 3, padding=1),
                'norm': nn.GroupNorm(8, c_out),
            }))
            c_in, k = c_out, k + 1
        self.stages = nn.ModuleList(stages)
        self.act = nn.SiLU()
        self.out = nn.Conv3d(c_in, out_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        for st in self.stages:
            h = self.act(st['norm'](st['conv'](st['up'](h))))
        return self.out(h)


class RefineHead(nn.Module):
    """Residual full-resolution conv refinement AFTER the linear unpatchify.

    The first conv head (``ConvHead``: trilinear upsample + 3x3x3 convs)
    learned far too slowly (loss ~1.1 at epoch 8 vs 0.41 for the linear
    decoder): upsample+conv is translation-equivariant and carries no
    sub-patch position, so it can only emit smooth fields and has to
    fight to make voxel-sharp facies. This head keeps the linear
    per-token decoder, which is voxel-sharp by construction, and adds a
    small conv stack over [linear output, model input] whose zero-init
    last layer produces a residual correction. Neighbouring patches thus
    interact at full resolution (removing the patch-tile artefact) while
    the decoder's expressiveness is untouched.
    """

    def __init__(self, in_channels, out_channels=1, width=32, depth=3):
        super().__init__()
        self.inp = nn.Conv3d(out_channels + in_channels, width, 3, padding=1)
        self.mid = nn.ModuleList([nn.Conv3d(width, width, 3, padding=1)
                                  for _ in range(depth - 1)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, width)
                                    for _ in range(depth)])
        self.act = nn.SiLU()
        self.out = nn.Conv3d(width, out_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, y, x_in):
        h = self.act(self.norms[0](self.inp(torch.cat([y, x_in], dim=1))))
        for conv, norm in zip(self.mid, self.norms[1:]):
            h = self.act(norm(conv(h)))
        return y + self.out(h)


class DiT3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, volume_shape=(64, 64, 32),
                 patch_size=(8, 8, 4), hidden=384, depth=12, num_heads=6,
                 num_cond=18, time_dim=256, mlp_ratio=4.0,
                 num_time_embs=1, expand_angle_idx=None, qk_norm=True,
                 conv_io=False):
        super().__init__()
        if conv_io is True:
            conv_io = 'up'
        assert conv_io in (False, None, 'up', 'refine'), conv_io
        self.conv_io = conv_io or False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.volume_shape = tuple(volume_shape)
        self.patch_size = tuple(patch_size)
        self.num_cond = num_cond
        self.num_time_embs = num_time_embs
        self.expand_angle_idx = expand_angle_idx
        self.hidden = hidden
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        # Persisted so a checkpoint carries its own head count; the old
        # loader silently assumed 6, which mis-splits a heads=8 model.
        self.register_buffer('num_heads_buf', torch.tensor(int(num_heads)))

        gx, gy, gz = (volume_shape[i] // patch_size[i] for i in range(3))
        self.grid = (gx, gy, gz)
        self.num_patches = gx * gy * gz

        self._inpaint_mask = None
        self._inpaint_data = None
        self._ctx_level = None

        if self.conv_io:
            self.patch_embed = ConvStem(in_channels, hidden, patch_size)
        else:
            self.patch_embed = nn.Conv3d(in_channels, hidden,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        sub = time_dim // num_time_embs
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(sub),
                                      nn.Linear(sub, sub), nn.SiLU())
        self.joint_time_mlp = nn.Sequential(nn.Linear(time_dim, hidden),
                                            nn.SiLU())
        cond_in = num_cond + (1 if expand_angle_idx is not None else 0)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_in, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.null_cond_emb = nn.Parameter(torch.randn(hidden) * 0.02)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, num_heads, mlp_ratio, qk_norm=qk_norm)
             for _ in range(depth)])

        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False,
                                       eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        if self.conv_io == 'up':
            self.head = ConvHead(hidden, patch_size, out_channels)
        else:
            self.final_linear = nn.Linear(hidden, patch_vol * out_channels)
        if self.conv_io == 'refine':
            self.refine = RefineHead(in_channels, out_channels)

        # DiT init: xavier-uniform on the transformer linears (the adaLN
        # and output heads are re-zeroed below), as in Peebles & Xie.
        for blk in self.blocks:
            for m in (blk.attn.qkv, blk.attn.proj, blk.mlp[0], blk.mlp[2]):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            nn.init.zeros_(blk.ada[1].weight)
            nn.init.zeros_(blk.ada[1].bias)
        if not self.conv_io:
            w = self.patch_embed.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.zeros_(self.patch_embed.bias)
        if self.conv_io != 'up':
            nn.init.zeros_(self.final_linear.weight)
            nn.init.zeros_(self.final_linear.bias)
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)

    # -- inpainting context: same contract as UNet3D --------------------
    def set_inpaint_context(self, mask, data, ctx_level=None):
        self._inpaint_mask = mask.detach()
        self._inpaint_data = data.detach()
        if ctx_level is not None and self.in_channels >= 4:
            s = ctx_level
            if not torch.is_tensor(s):
                s = torch.full((mask.shape[0],), float(s), device=mask.device)
            if s.dim() == 1:
                s = s.view(-1, 1, 1, 1, 1)
            self._ctx_level = (s.to(mask.dtype) * mask).detach()
        else:
            self._ctx_level = None

    def clear_inpaint_context(self):
        self._inpaint_mask = None
        self._inpaint_data = None
        self._ctx_level = None

    def _process_conditioning(self, cond):
        i = self.expand_angle_idx
        if i is None:
            return cond
        a = cond[:, i:i + 1]
        return torch.cat([cond[:, :i], torch.sin(2 * math.pi * a),
                          torch.cos(2 * math.pi * a), cond[:, i + 1:]], dim=1)

    def unpatchify(self, x):
        gx, gy, gz = self.grid
        px, py, pz = self.patch_size
        B = x.shape[0]
        x = x.reshape(B, gx, gy, gz, px, py, pz, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, self.out_channels, gx * px, gy * py, gz * pz)

    def forward(self, x, *args, drop_mask=None):
        if len(args) > 1 and args[-1].dim() == 2:
            times, cond = args[:-1], args[-1]
        else:
            times, cond = args, None

        t_embs = [self.time_mlp(t) for t in times]
        if len(t_embs) < self.num_time_embs:
            t_embs.extend([torch.zeros_like(t_embs[0])]
                          * (self.num_time_embs - len(t_embs)))
        c = self.joint_time_mlp(torch.cat(t_embs, dim=-1))

        if cond is not None:
            c_emb = self.cond_mlp(self._process_conditioning(cond))
            if drop_mask is not None:
                c_emb = c_emb.clone()
                # .to(dtype) for autocast: c_emb is bf16, the Parameter fp32.
                c_emb[drop_mask] = self.null_cond_emb.to(c_emb.dtype)
        else:
            c_emb = self.null_cond_emb.unsqueeze(0).expand(x.shape[0], -1)
        c = c + c_emb

        if self.in_channels > 1:
            if self._inpaint_mask is not None:
                extra = [x, self._inpaint_data, self._inpaint_mask]
                if self.in_channels >= 4:
                    lvl = getattr(self, '_ctx_level', None)
                    if lvl is None:
                        lvl = self._inpaint_mask      # clean context, s = 1
                    extra.append(lvl)
                x = torch.cat(extra, dim=1)
            else:
                z = torch.zeros(x.shape[0], self.in_channels - 1, *x.shape[2:],
                                device=x.device, dtype=x.dtype)
                x = torch.cat([x, z], dim=1)

        h = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, hidden)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        if self.conv_io == 'up':
            gx, gy, gz = self.grid
            h = h.transpose(1, 2).reshape(h.shape[0], self.hidden, gx, gy, gz)
            return self.head(h)
        y = self.unpatchify(self.final_linear(h))
        if self.conv_io == 'refine':
            y = self.refine(y, x)
        return y
