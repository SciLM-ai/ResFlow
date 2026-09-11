import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SinusoidalPosEmb


class DownBlock3D(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, attn=False, attn_heads=4):
        super().__init__()
        self.attn = SelfAttention3D(out_c, attn_heads) if attn else None
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.downsample = nn.MaxPool3d(2)

        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None, None]
        h = self.gn2(self.act(self.conv2(h)))
        if self.attn is not None:
            h = self.attn(h)
        return self.downsample(h), h


class UpBlock3D(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_emb_dim, attn=False,
                 attn_heads=4):
        super().__init__()
        self.attn = SelfAttention3D(out_c, attn_heads) if attn else None
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv3d(in_c + skip_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)

        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, res, t_emb):
        x = F.interpolate(x, size=res.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat((x, res), dim=1)
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None, None]
        h = self.gn2(self.act(self.conv2(h)))
        if self.attn is not None:
            h = self.attn(h)
        return h


class SelfAttention3D(nn.Module):
    """Multi-head self-attention over the flattened 3D grid.

    UNet3D is otherwise pure convolution: its theoretical receptive field
    at the bottleneck is ~70 voxels, barely covering a 64-cell block, and
    the effective field is far smaller. That leaves nothing to coordinate
    structure at body scale, which is where the assembly benchmark shows
    the model failing. At the bottleneck the grid is 8x8x4 = 256 tokens,
    so full attention is cheap.

    The output projection is zero-initialised, so at step 0 the block is
    an exact identity and the network starts from the pre-attention
    behaviour rather than from noise injected into a tuned architecture.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divisible by num_heads {num_heads}'
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads,
                                  D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]      # (B, heads, hc, N)
        q = q.transpose(-2, -1)                         # (B, heads, N, hc)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, heads, N, hc)
        out = out.transpose(-2, -1).reshape(B, C, D, H, W)
        return x + self.proj(out)


class UNet3D(nn.Module):
    """3D UNet for volumetric generative modeling with continuous conditioning.

    Conditioning inputs (height, radius, aspect_ratio, angle_deg, ntg) are
    expected to be normalized to [0, 1] by the dataset. Angle is internally
    converted to sin(2*pi*angle_norm) and cos(2*pi*angle_norm) to handle
    the 180-degree periodicity of lobe orientation.
    """

    def __init__(self, in_channels=1, hidden_dims=None, time_dim=256,
                 num_cond=5, num_time_embs=1, out_channels=None,
                 expand_angle_idx=3, attention=False, attn_heads=4,
                 attn_levels=0):
        """
        expand_angle_idx: index in `cond` whose value is replaced by sin/cos
            (used for periodic angle conditioning, lobes default = 3).
            Set to None when conditioning is already pre-processed by the
            dataset (e.g. one-hot + sin/cos baked in).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 128, 128]
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.time_dim = time_dim
        self.num_time_embs = num_time_embs
        self.num_cond = num_cond
        self.expand_angle_idx = expand_angle_idx

        # Inpaint context (set via set_inpaint_context for channel concat)
        self._inpaint_mask = None
        self._inpaint_data = None
        self._ctx_level = None

        # Time embedding
        sub_dim = time_dim // num_time_embs
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sub_dim),
            nn.Linear(sub_dim, sub_dim),
            nn.SiLU()
        )
        self.joint_time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Conditioning embedding
        # If expand_angle_idx is set, one input is replaced by sin+cos -> +1 dim.
        cond_input_dim = num_cond + (1 if expand_angle_idx is not None else 0)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Learned null embedding for classifier-free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(time_dim))

        # Encoder
        self.init_conv = nn.Conv3d(in_channels, hidden_dims[0], 3, padding=1)

        # attn_levels = how many of the DEEPEST encoder/decoder stages also
        # get attention, on top of the mid block. Level resolutions for a
        # 64x64x32 input are 64^2x32, 32^2x16, 16^2x8; the mid block sits at
        # 8x8x4 (256 tokens). One extra level means 16x16x8 = 2048 tokens.
        self.downs = nn.ModuleList()
        in_c = hidden_dims[0]
        channels = [hidden_dims[0]]
        n_stages = len(hidden_dims) - 1
        for k, out_c in enumerate(hidden_dims[1:]):
            use_attn = attention and (k >= n_stages - attn_levels)
            self.downs.append(DownBlock3D(in_c, out_c, time_dim,
                                          attn=use_attn,
                                          attn_heads=attn_heads))
            channels.append(out_c)
            in_c = out_c

        # Mid block
        self.mid_block1 = nn.Conv3d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn1 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_time_mlp = nn.Linear(time_dim, hidden_dims[-1])
        self.mid_block2 = nn.Conv3d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn2 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_act = nn.SiLU()
        self.attention = attention
        self.mid_attn = (SelfAttention3D(hidden_dims[-1], attn_heads)
                         if attention else None)

        # Decoder
        self.ups = nn.ModuleList()
        for k, (skip_c, out_c) in enumerate(
                zip(reversed(channels[1:]), reversed(hidden_dims[:-1]))):
            use_attn = attention and (k < attn_levels)
            self.ups.append(UpBlock3D(in_c, skip_c, out_c, time_dim,
                                      attn=use_attn, attn_heads=attn_heads))
            in_c = out_c

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv3d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dims[0], out_channels, 1)
        )

    def set_inpaint_context(self, mask, data, ctx_level=None):
        """Store inpaint mask and known data for channel concatenation.

        Args:
            mask: (B, 1, D, H, W) float tensor, 1=known 0=unknown
            data: (B, 1, D, H, W) float tensor, context values where mask=1
            ctx_level: only for in_channels == 4 (trajectory conditioning).
                The noise level s of the context: 1.0 means clean (classic
                outpainting), s < 1 means the neighbour is itself only
                partially denoised, which is what lets blocks advance
                together instead of strictly in sequence. Scalar, (B,) or
                a full (B, 1, D, H, W) map; broadcast and masked to the
                context region. Defaults to 1.0 (clean).
        """
        self._inpaint_mask = mask.detach()
        self._inpaint_data = data.detach()
        if ctx_level is None:
            self._ctx_level = None
        else:
            if not torch.is_tensor(ctx_level):
                ctx_level = torch.full((mask.shape[0],), float(ctx_level),
                                       device=mask.device)
            if ctx_level.ndim == 1:
                ctx_level = ctx_level.view(-1, 1, 1, 1, 1).expand_as(mask)
            self._ctx_level = ctx_level.detach()

    def clear_inpaint_context(self):
        """Remove stored inpaint context."""
        self._inpaint_mask = None
        self._inpaint_data = None
        self._ctx_level = None

    def _process_conditioning(self, cond):
        """Optionally replace the periodic angle entry with sin/cos.

        If expand_angle_idx is None, returns cond unchanged (dataset is
        expected to have done any preprocessing already).
        """
        i = self.expand_angle_idx
        if i is None:
            return cond
        angle_norm = cond[:, i:i+1]
        sin_a = torch.sin(2 * math.pi * angle_norm)
        cos_a = torch.cos(2 * math.pi * angle_norm)
        return torch.cat([cond[:, :i], sin_a, cos_a, cond[:, i+1:]], dim=1)

    def forward(self, x, *args, drop_mask=None):
        # Parse args: last 2D tensor is conditioning, rest are time(s)
        # drop_mask: optional BoolTensor (B,), True = replace with null embedding for CFG
        if len(args) > 1 and args[-1].dim() == 2:
            times = args[:-1]
            cond = args[-1]
        else:
            times = args
            cond = None

        # Time embedding
        t_embs = [self.time_mlp(t) for t in times]
        if len(t_embs) < self.num_time_embs:
            t_embs.extend([torch.zeros_like(t_embs[0])] * (self.num_time_embs - len(t_embs)))
        t_emb = torch.cat(t_embs, dim=-1)
        t_emb = self.joint_time_mlp(t_emb)

        # Conditioning embedding
        if cond is not None:
            cond_processed = self._process_conditioning(cond)
            c_emb = self.cond_mlp(cond_processed)

            if drop_mask is not None:
                c_emb = c_emb.clone()
                # .to(dtype) so this works under autocast, where c_emb is
                # bf16 while the null-embedding Parameter stays fp32.
                c_emb[drop_mask] = self.null_cond_emb.to(c_emb.dtype)
        else:
            c_emb = self.null_cond_emb.unsqueeze(0).expand(x.shape[0], -1)

        t_emb = t_emb + c_emb

        # Inpaint channel concatenation (only for in_channels > 1).
        # 3 channels: [noisy_x, context_values, mask]
        # 4 channels: [noisy_x, context_values, mask, context_noise_level]
        if self.in_channels > 1:
            if self._inpaint_mask is not None:
                parts = [x, self._inpaint_data, self._inpaint_mask]
                if self.in_channels >= 4:
                    lvl = self._ctx_level
                    if lvl is None:      # clean context by default
                        lvl = torch.ones_like(self._inpaint_mask)
                    parts.append(lvl * self._inpaint_mask)
                x = torch.cat(parts, dim=1)
            else:
                zeros = torch.zeros(
                    x.shape[0], self.in_channels - 1, *x.shape[2:],
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, zeros], dim=1)

        # Encoder
        x = self.init_conv(x)
        res_stack = [x]

        for down in self.downs:
            x, res = down(x, t_emb)
            res_stack.append(res)

        # Mid
        x = self.mid_gn1(self.mid_act(self.mid_block1(x)))
        x = x + self.mid_act(self.mid_time_mlp(t_emb))[..., None, None, None]
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        x = self.mid_gn2(self.mid_act(self.mid_block2(x)))

        # Decoder
        for up in self.ups:
            res = res_stack.pop()
            x = up(x, res, t_emb)

        return self.final_conv(x)
