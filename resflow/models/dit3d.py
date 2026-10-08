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

Whole-field generation (2026-09-11). With ``pos_embed='rope'`` the
learned absolute position table is replaced by 3D axial rotary
embeddings (RoPE), so attention depends on relative token offsets only
and the same weights run on a token grid of any size. With ``window``
set, attention is restricted to non-overlapping windows of that many
tokens, and every odd block shifts the window grid by half a window in
x and y (Swin-style) so information crosses window borders. Windows are
cut where the grid ends instead of being wrapped or padded, so border
windows are simply smaller: no attention mask is ever needed, flash
kernels apply everywhere, and a field whose size is not a multiple of the
window costs nothing extra. Trained on crops larger than one window, the
model can then generate a whole reservoir in one pass — no tiling, no
fusion rule, cost linear in field size. Conditioning may be given per
token (a condition map) as well as per sample, so a field with spatially
varying parameters is one forward call.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:                                        # PyTorch >= 2.5
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _HAS_FLEX = True
except Exception:                           # pragma: no cover
    _flex_attention = _create_block_mask = None
    _HAS_FLEX = False
import torch.nn.functional as F

from .unet import SinusoidalPosEmb


def modulate(x, shift, scale):
    """adaLN modulation; shift/scale are (B, C) per sample or (B, N, C)
    per token."""
    if shift.dim() == 2:
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
    return x * (1 + scale) + shift


def _gate(g):
    return g.unsqueeze(1) if g.dim() == 2 else g


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


# ---------------------------------------------------------------------------
# 3D rotary position embedding and window partitions
# ---------------------------------------------------------------------------

def rope_axis_dims(head_dim):
    """Split head_dim into three even chunks (x, y, z) for axial RoPE."""
    dx = 2 * int(round(head_dim / 6.0))
    dz = head_dim - 2 * dx
    assert dz > 0 and dz % 2 == 0, (head_dim, dx, dz)
    return dx, dx, dz


# Inference-time RoPE extrapolation (off when None). Set by a sampling script:
#   ROPE_EXTRAP = {'mode': 'pi'|'ntk'|'yarn', 'train': (tx, ty, tz) token extent
#                  of the largest training crop, 'alpha': 1.0, 'beta': 2.0 (yarn
#                  ramp, in rotations within the training extent), 'temp': bool
#                  (yarn attention temperature), 'kappa': 1.0}
# kappa in [0, 1] scales the extrapolation strength (DyPE: the sampler sets it
# from the timestep, full strength at pure noise, none at the data end).
# Per axis s = max(1, grid / train): axes no longer than training are untouched.
#   pi   : positions / s^kappa                          (Chen et al. 2023)
#   ntk  : freq_i / s^(kappa * 2i / (d - 2))            (NTK-aware; DyPE Dy-NTK)
#   yarn : per frequency, r = train / wavelength rotations; gamma = ramp(r;
#          alpha*kappa, beta*kappa); freq' = (1 - gamma) freq / s + gamma freq
#          (Peng et al. 2023 "NTK-by-parts"; DyPE Dy-YaRN scales the ramp)
ROPE_EXTRAP = None


def _axis_freqs(theta, d, g, axis, device):
    """Frequencies (d/2,) and a position divisor for one axis of length g."""
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    freq = theta ** (-i / d)
    ext = ROPE_EXTRAP
    if not ext:
        return freq, 1.0
    s = max(1.0, g / float(ext['train'][axis]))
    kap = float(ext.get('kappa', 1.0))
    if s <= 1.0 or kap <= 0.0:
        return freq, 1.0
    mode = ext['mode']
    if mode == 'pi':
        return freq, s ** kap
    if mode == 'ntk':
        k = i / 2.0                                   # pair index 0 .. d/2-1
        return freq / s ** (kap * 2.0 * k / (d - 2)), 1.0
    if mode == 'yarn':
        r = float(ext['train'][axis]) * freq / (2 * math.pi)   # rotations inside training extent
        a, b = ext.get('alpha', 1.0) * kap, ext.get('beta', 2.0) * kap
        gam = torch.clamp((r - a) / max(b - a, 1e-6), 0.0, 1.0)
        return (1 - gam) * freq / s + gam * freq, 1.0
    raise ValueError(mode)


def rope_attn_scale(grid, head_dim):
    """YaRN attention temperature (logit multiplier) or None."""
    ext = ROPE_EXTRAP
    if not ext or not ext.get('temp') or ext.get('mode') != 'yarn':
        return None
    s = max([1.0] + [g / float(t) for g, t in zip(grid, ext['train'])])
    if s <= 1.0:
        return None
    m = (0.1 * math.log(s) + 1.0) ** 2
    kap = float(ext.get('kappa', 1.0))
    return (1.0 + kap * (m - 1.0)) / math.sqrt(head_dim)


def rope_tables(grid, head_dim, theta, device):
    """cos / sin tables of shape (N, head_dim // 2) for a token grid.

    Token n = ix * gy * gz + iy * gz + iz (the flatten order of a
    (B, C, gx, gy, gz) tensor). Each axis gets its own frequency ladder
    theta^(-2i/d_axis); the angle of channel pair i is coordinate * freq.
    With ROPE_EXTRAP set, long axes are rescaled (see above).
    """
    gx, gy, gz = grid
    dims = rope_axis_dims(head_dim)
    ix, iy, iz = torch.meshgrid(torch.arange(gx, device=device),
                                torch.arange(gy, device=device),
                                torch.arange(gz, device=device), indexing='ij')
    angles = []
    for axis, (coord, d, g) in enumerate(zip((ix, iy, iz), dims, grid)):
        freq, div = _axis_freqs(theta, d, g, axis, device)
        angles.append((coord.reshape(-1, 1).float() / div) * freq)
    ang = torch.cat(angles, dim=1)                     # (N, head_dim/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """Rotate interleaved channel pairs of x (..., N, d) by the table angles
    (broadcastable to (..., N, d/2))."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                       dim=-1).flatten(-2)


def axis_bounds(g, w, s):
    """Window intervals along one axis of length g for window w and shift
    s: [0, s), [s, s+w), ... clipped at g. w <= 0 means one window."""
    if w <= 0:
        return [(0, g)]
    cuts = [0]
    p = s if 0 < s < g else w
    while p < g:
        cuts.append(p)
        p += w
    cuts.append(g)
    return list(zip(cuts[:-1], cuts[1:]))


def window_groups(grid, window, shift, device):
    """Token-index tensors for a window partition of the grid, grouped by
    window size so each group is one batched attention call.

    Returns a list of LongTensors (n_windows, tokens_per_window); every
    token appears in exactly one of them. Border windows are the
    remainder of the grid, never padded or wrapped.
    """
    gx, gy, gz = grid
    bx = axis_bounds(gx, window[0], shift[0])
    by = axis_bounds(gy, window[1], shift[1])
    bz = axis_bounds(gz, window[2], shift[2])
    groups = {}
    for x0, x1 in bx:
        for y0, y1 in by:
            for z0, z1 in bz:
                ix = torch.arange(x0, x1).view(-1, 1, 1)
                iy = torch.arange(y0, y1).view(1, -1, 1)
                iz = torch.arange(z0, z1).view(1, 1, -1)
                idx = (ix * gy * gz + iy * gz + iz).reshape(-1)
                groups.setdefault((x1 - x0, y1 - y0, z1 - z0), []).append(idx)
    return [torch.stack(v).to(device) for v in groups.values()]


def _logn_scale(L, head_dim):
    """Attention temperature for sequences longer than training (off by default).

    RESFLOW_ATTN_LOGN_TRAIN=<N_train tokens>: when L > N_train the logits are
    multiplied by log(L)/log(N_train) (the log-n scaling used for LLM length
    extrapolation), so attention over many more keys than training stays as
    sharp as in training instead of averaging features away. Returns None
    (SDPA's default 1/sqrt(d)) when unset or L <= N_train."""
    import os
    n = os.environ.get('RESFLOW_ATTN_LOGN_TRAIN')
    if not n or L <= int(n):
        return None
    return math.log(L) / math.log(int(n)) / math.sqrt(head_dim)


def neighbourhood_mask_mod(grid, radius):
    """mask_mod for FlexAttention: a token attends to every token within
    `radius` cells of it on each axis of the token grid.

    Unlike a window partition this is translation-equivariant -- there are no
    block boundaries, so the receptive field is identical for every token and
    a grid can be extended along ANY axis (z included) without the model
    meeting a configuration it never saw in training.
    """
    gy, gz = grid[1], grid[2]
    rx, ry, rz = radius

    def mod(b, h, q, kv):
        qx, qy, qz = q // (gy * gz), (q // gz) % gy, q % gz
        kx, ky, kz = kv // (gy * gz), (kv // gz) % gy, kv % gz
        return (((qx - kx).abs() <= rx) & ((qy - ky).abs() <= ry)
                & ((qz - kz).abs() <= rz))
    return mod


def _flex_tile():
    """Optional 3D tiling of the token order inside sliding attention, from
    RESFLOW_FLEX_TILE="tx,ty,tz" (off by default). Attention is permutation-
    equivariant, so reordering q, k, v and un-permuting the output is the SAME
    computation; it only makes each 128-token kernel block spatially compact,
    so fewer blocks are touched (measured 1.9x faster forward on a 32x32x16
    grid at radius 8). RESFLOW_FLEX_KOPTS="BLOCK_M,BLOCK_N" sets the forward
    kernel tile (64,64 measured best on GH200)."""
    import os
    t = os.environ.get('RESFLOW_FLEX_TILE')
    return tuple(int(v) for v in t.split(',')) if t else None


def flex_kernel_options():
    import os
    k = os.environ.get('RESFLOW_FLEX_KOPTS')
    if not k:
        return None
    m, n = (int(v) for v in k.split(','))
    return {'BLOCK_M': m, 'BLOCK_N': n}


def build_neighbourhood_mask(grid, radius, device):
    """Compiled block-sparse mask for `neighbourhood_mask_mod`. Built once per
    (grid, radius) and cached by the caller -- construction is seconds, the
    attention call itself is milliseconds. With RESFLOW_FLEX_TILE set (and the
    grid divisible by the tile) the mask is built for the tiled token order
    and carries `_tile_perm` / `_tile_inv` for the attention call."""
    n = grid[0] * grid[1] * grid[2]
    tile = _flex_tile()
    if tile is None or any(g % t for g, t in zip(grid, tile)):
        return _create_block_mask(neighbourhood_mask_mod(grid, radius),
                                  None, None, n, n, device=device, _compile=True)
    gx, gy, gz = grid; tx, ty, tz = tile; rx, ry, rz = radius
    i = torch.arange(n, device=device)
    cx, cy, cz = i // (gy * gz), (i // gz) % gy, i % gz
    key = ((((cx // tx) * (gy // ty) + cy // ty) * (gz // tz) + cz // tz) * (tx * ty * tz)
           + ((cx % tx) * ty + cy % ty) * tz + cz % tz)
    perm = torch.argsort(key)                  # tiled position j holds token perm[j]
    inv = torch.argsort(perm)
    X, Y, Z = cx[perm], cy[perm], cz[perm]

    def mod(b, h, q, kv):
        return (((X[q] - X[kv]).abs() <= rx) & ((Y[q] - Y[kv]).abs() <= ry)
                & ((Z[q] - Z[kv]).abs() <= rz))
    bm = _create_block_mask(mod, None, None, n, n, device=device, _compile=True)
    bm._tile_perm, bm._tile_inv = perm, inv
    return bm


_FLEX_COMPILED = None


def flex_call(q, k, v, block_mask):
    """Run FlexAttention through its compiled form.

    Eager ``flex_attention`` materialises the full (B, H, N, N) score matrix and
    only then applies the mask, so the block sparsity buys nothing: a whole
    532x532x32 field is 283k tokens, i.e. 1.8 TB of scores. Only the compiled
    kernel skips masked-out blocks. Compile once and reuse across calls.
    """
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(_flex_attention, dynamic=False)
    return _FLEX_COMPILED(q, k, v, block_mask=block_mask,
                          kernel_options=flex_kernel_options())


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm, optional RoPE and optional
    window partition.

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

    def _attend(self, x, cos=None, sin=None):
        """x: (B', L, C); cos/sin: (B' or 1, 1, L, d/2) or None."""
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # (B, H, L, d)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        sc = _logn_scale(L, self.head_dim)
        if sc is None and ROPE_EXTRAP and getattr(self, '_grid', None) is not None:
            sc = rope_attn_scale(self._grid, self.head_dim)
        o = F.scaled_dot_product_attention(q, k, v, scale=sc)
        return o.transpose(1, 2).reshape(B, L, C)

    def _attend_flex(self, x, block_mask, cos=None, sin=None):
        """Sliding-neighbourhood attention over the whole token grid."""
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        perm = getattr(block_mask, '_tile_perm', None)
        if perm is not None:
            q, k, v = q[:, :, perm], k[:, :, perm], v[:, :, perm]
        o = flex_call(q, k, v, block_mask)
        if perm is not None:
            o = o[:, :, block_mask._tile_inv]
        return o.transpose(1, 2).reshape(B, L, C)

    def forward(self, x, rope=None, groups=None, block_mask=None):
        """rope: (cos, sin) tables (N, d/2) for the token grid, or None.
        groups: window partition (see ``window_groups``), or None.
        block_mask: sliding-neighbourhood mask; takes precedence over groups."""
        B, N, C = x.shape
        if block_mask is not None:
            cs = (rope[0].view(1, 1, N, -1), rope[1].view(1, 1, N, -1)) \
                if rope is not None else (None, None)
            return self.proj(self._attend_flex(x, block_mask, *cs))
        if groups is None:
            cs = (rope[0].view(1, 1, N, -1), rope[1].view(1, 1, N, -1)) \
                if rope is not None else (None, None)
            return self.proj(self._attend(x, *cs))
        out = None
        for idx in groups:
            nW, L = idx.shape
            xg = x[:, idx].reshape(B * nW, L, C)
            if rope is not None:
                cos = rope[0][idx].unsqueeze(0).expand(B, nW, L, -1)
                sin = rope[1][idx].unsqueeze(0).expand(B, nW, L, -1)
                cs = (cos.reshape(B * nW, 1, L, -1),
                      sin.reshape(B * nW, 1, L, -1))
            else:
                cs = (None, None)
            og = self._attend(xg, *cs).reshape(B, nW, L, C)
            if out is None:                     # bf16 under autocast
                out = x.new_empty(x.shape, dtype=og.dtype)
            out[:, idx] = og
        return self.proj(out)


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

    def forward(self, x, c, rope=None, groups=None, block_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.attn(h, rope=rope, groups=groups, block_mask=block_mask)
        x = x + _gate(gate1) * h
        h = self.mlp(modulate(self.norm2(x), shift2, scale2))
        x = x + _gate(gate2) * h
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


class TokenConvHead(nn.Module):
    """3x3x3 convolutions over the TOKEN grid, before the linear unpatchify.

    Neighbouring patches are decoded by independent per-token linear maps, so a
    thin feature crossing a patch boundary needs two token vectors to agree
    cell-by-cell; one disagreeing cell merges two bodies. Two earlier fixes both
    failed: an upsample+conv decoder replacing the linear head learns far too
    slowly (translation-equivariant, no sub-patch position), and a residual conv
    stack AFTER the linear head at full resolution reduces MSE by dithering
    every voxel, which binarisation turns into speckle (measured: isolated shale
    voxels 30 -> 63 per 10^6, body-size W1 0.089 -> 0.191).

    This head instead mixes tokens with their neighbours at TOKEN resolution and
    leaves the voxel-sharp linear decode untouched. It cannot dither individual
    cells because it never sees them. Zero-initialised last conv, so it starts
    as the identity.
    """

    def __init__(self, hidden, depth=2, width=None, kernel=(3, 3, 3)):
        super().__init__()
        w = width or hidden
        k = tuple(kernel); pad = tuple(x // 2 for x in k)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # A z-only kernel is applied as Conv1d over depth with x,y folded into
        # the batch: identical arithmetic, and it avoids a cuDNN 3-D path that
        # intermittently fails to finalise for anisotropic kernels.
        self.z_only = (k[0] == 1 and k[1] == 1)
        c_in = hidden
        for j in range(depth):
            c_out = hidden if j == depth - 1 else w
            if self.z_only:
                self.convs.append(nn.Conv1d(c_in, c_out, k[2], padding=pad[2]))
            else:
                self.convs.append(nn.Conv3d(c_in, c_out, k, padding=pad))
            # No norm on the final conv: it is the zero-init residual output and
            # its activation is never normalised, so creating one would leave an
            # unused parameter and DDP would refuse to run.
            if j < depth - 1:
                self.norms.append(nn.GroupNorm(8, c_out))
            c_in = c_out
        self.act = nn.SiLU()
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, h, grid):
        """h: (B, N, hidden) tokens in (gx, gy, gz) flatten order."""
        B, N, C = h.shape
        gx, gy, gz = grid
        if self.z_only:
            # (B, N, C) -> (B*gx*gy, C, gz)
            y = h.reshape(B, gx * gy, gz, C).permute(0, 1, 3, 2).reshape(-1, C, gz)
        else:
            y = h.transpose(1, 2).reshape(B, C, gx, gy, gz)
        for j, cv in enumerate(self.convs):
            y = cv(y) if j == len(self.convs) - 1 else self.act(self.norms[j](cv(y)))
        if self.z_only:
            y = y.reshape(B, gx * gy, C, gz).permute(0, 1, 3, 2).reshape(B, N, C)
        else:
            y = y.reshape(B, C, N).transpose(1, 2)
        return h + y


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
                 conv_io=False, pos_embed='learned', rope_theta=10000.0,
                 window=None, window_shift=True, token_conv=0,
                 attn_mode='window', attn_radius=(8, 8, 8),
                 token_conv_kernel=(3, 3, 3)):
        """pos_embed: 'learned' (absolute table for volume_shape; fixed
        input size) or 'rope' (3D rotary; any input size).
        window: (wx, wy, wz) attention window in TOKENS, or None for global
        attention. With window_shift, odd blocks shift the window grid by
        half a window in x and y.
        token_conv: depth of a TokenConvHead applied to the token grid before
        the linear unpatchify (0 = off). Cross-patch agreement at token
        resolution; see TokenConvHead."""
        super().__init__()
        if conv_io is True:
            conv_io = 'up'
        assert conv_io in (False, None, 'up', 'refine'), conv_io
        assert pos_embed in ('learned', 'rope'), pos_embed
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
        self.pos_type = pos_embed
        self.rope_theta = float(rope_theta)
        self.window = tuple(int(w) for w in window) if window else None
        self.window_shift = bool(window_shift) and self.window is not None
        # Persisted so a checkpoint carries its own head count; the old
        # loader silently assumed 6, which mis-splits a heads=8 model.
        self.register_buffer('num_heads_buf', torch.tensor(int(num_heads)))
        # Likewise for the position scheme (0 learned / 1 rope), RoPE base,
        # window and shift, so a checkpoint rebuilds itself.
        self.register_buffer('pos_type_buf',
                             torch.tensor(1 if pos_embed == 'rope' else 0))
        self.register_buffer('rope_theta_buf', torch.tensor(self.rope_theta))
        self.register_buffer('window_buf',
                             torch.tensor(list(self.window or (0, 0, 0))))
        self.register_buffer('window_shift_buf',
                             torch.tensor(int(self.window_shift)))
        self.token_conv_depth = int(token_conv)
        self.token_conv_kernel = tuple(int(v) for v in token_conv_kernel)
        self.register_buffer('token_conv_buf',
                             torch.tensor(int(token_conv)))
        self.register_buffer('token_conv_k_buf',
                             torch.tensor(list(self.token_conv_kernel)))

        gx, gy, gz = (volume_shape[i] // patch_size[i] for i in range(3))
        self.grid = (gx, gy, gz)
        self.num_patches = gx * gy * gz

        self._inpaint_mask = None
        self._inpaint_data = None
        self._ctx_level = None
        self._rope_cache = {}
        self._group_cache = {}
        self._mask_cache = {}
        # 'window'  : Swin-style partition, shifted in x/y on odd blocks.
        # 'sliding' : translation-equivariant neighbourhood attention. No
        #   block boundaries, so the grid extends along ANY axis -- including
        #   z, which the window partition cannot do because it never shifts
        #   in depth.
        self.attn_mode = str(attn_mode)
        self.attn_radius = tuple(int(r) for r in attn_radius)
        if self.attn_mode == 'sliding' and not _HAS_FLEX:
            raise RuntimeError('attn_mode="sliding" needs torch>=2.5 FlexAttention')
        self.register_buffer('attn_mode_buf',
                             torch.tensor(1 if self.attn_mode == 'sliding' else 0))
        self.register_buffer('attn_radius_buf', torch.tensor(list(self.attn_radius)))

        if self.conv_io:
            self.patch_embed = ConvStem(in_channels, hidden, patch_size)
        else:
            self.patch_embed = nn.Conv3d(in_channels, hidden,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        if self.pos_type == 'learned':
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
        if self.token_conv_depth > 0:
            self.token_head = TokenConvHead(hidden, depth=self.token_conv_depth,
                                            kernel=self.token_conv_kernel)

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
        a = cond[..., i:i + 1]
        return torch.cat([cond[..., :i], torch.sin(2 * math.pi * a),
                          torch.cos(2 * math.pi * a), cond[..., i + 1:]],
                         dim=-1)

    def _cond_tokens(self, cond, grid):
        """Accept cond as (B, C) per sample, (B, N, C) per token, or a
        voxel map (B, C, X, Y, Z) which is average-pooled to the token
        grid. Returns (B, C) or (B, N, C)."""
        if cond.dim() == 5:
            cond = F.avg_pool3d(cond, self.patch_size, self.patch_size)
            cond = cond.flatten(2).transpose(1, 2)
        if cond.dim() == 3:
            assert cond.shape[1] == grid[0] * grid[1] * grid[2], \
                (cond.shape, grid)
        return cond

    def _tables(self, grid, device):
        if self.pos_type != 'rope':
            return None
        ext = tuple(sorted(ROPE_EXTRAP.items())) if ROPE_EXTRAP else None
        key = (grid, str(device), ext)
        if key not in self._rope_cache:
            if len(self._rope_cache) > 64:
                self._rope_cache.clear()
            self._rope_cache[key] = rope_tables(grid, self.blocks[0].attn.head_dim,
                                                self.rope_theta, device)
        return self._rope_cache[key]

    def _block_mask(self, grid, device):
        """Sliding-neighbourhood mask for this token grid, cached per grid."""
        key = (grid, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = build_neighbourhood_mask(
                grid, self.attn_radius, device)
        return self._mask_cache[key]

    def _groups(self, grid, shifted, device):
        if self.window is None:
            return None
        key = (grid, shifted, str(device))
        if key not in self._group_cache:
            w = self.window
            s = ((w[0] // 2, w[1] // 2, 0) if shifted else (0, 0, 0))
            # A window covering the whole axis needs no split at all.
            if all(g <= wi for g, wi in zip(grid, w)) and not shifted:
                self._group_cache[key] = None
            else:
                self._group_cache[key] = window_groups(grid, w, s, device)
        return self._group_cache[key]

    def unpatchify(self, x, grid):
        gx, gy, gz = grid
        px, py, pz = self.patch_size
        B = x.shape[0]
        x = x.reshape(B, gx, gy, gz, px, py, pz, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, self.out_channels, gx * px, gy * py, gz * pz)

    def forward(self, x, *args, drop_mask=None):
        if len(args) > 1 and args[-1].dim() in (2, 3, 5):
            times, cond = args[:-1], args[-1]
        else:
            times, cond = args, None

        grid = tuple(x.shape[2 + i] // self.patch_size[i] for i in range(3))
        if self.pos_type == 'learned':
            assert grid == self.grid, \
                f'learned position table is for grid {self.grid}, got {grid}'

        t_embs = [self.time_mlp(t) for t in times]
        if len(t_embs) < self.num_time_embs:
            t_embs.extend([torch.zeros_like(t_embs[0])]
                          * (self.num_time_embs - len(t_embs)))
        c = self.joint_time_mlp(torch.cat(t_embs, dim=-1))      # (B, hidden)

        if cond is not None:
            cond = self._cond_tokens(cond, grid)
            c_emb = self.cond_mlp(self._process_conditioning(cond))
            if drop_mask is not None:
                c_emb = c_emb.clone()
                # .to(dtype) for autocast: c_emb is bf16, the Parameter fp32.
                c_emb[drop_mask] = self.null_cond_emb.to(c_emb.dtype)
        else:
            c_emb = self.null_cond_emb.unsqueeze(0).expand(x.shape[0], -1)
        c = (c.unsqueeze(1) + c_emb) if c_emb.dim() == 3 else c + c_emb

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
        if self.pos_type == 'learned':
            h = h + self.pos_embed
        rope = self._tables(grid, h.device)
        if ROPE_EXTRAP:
            for blk in self.blocks:
                blk.attn._grid = grid
        if self.attn_mode == 'sliding':
            bm = self._block_mask(grid, h.device)
            for blk in self.blocks:
                h = blk(h, c, rope=rope, block_mask=bm)
        else:
            g_even = self._groups(grid, False, h.device)
            g_odd = self._groups(grid, True, h.device) if self.window_shift else g_even
            for i, blk in enumerate(self.blocks):
                h = blk(h, c, rope=rope, groups=(g_odd if i % 2 else g_even))

        if self.token_conv_depth > 0:
            h = self.token_head(h, grid)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        if self.conv_io == 'up':
            gx, gy, gz = grid
            h = h.transpose(1, 2).reshape(h.shape[0], self.hidden, gx, gy, gz)
            return self.head(h)
        y = self.unpatchify(self.final_linear(h), grid)
        if self.conv_io == 'refine':
            y = self.refine(y, x)
        return y
