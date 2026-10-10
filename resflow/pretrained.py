"""The paper's ResFlow model, ready to generate reservoirs in a few lines.

    import resflow
    model = resflow.load_pretrained()                       # weights from huggingface.co/SciLM/ResFlow
    vols = model.generate('meander', n=8)                   # (8, 64, 64, 32) uint8, 1 = sand
    field = model.generate_field('lobe', shape=(512, 512, 32))

Volumes are binary facies on the dataset's grid: axes (x, y, z) with z the
LAST axis and z = 0 the base. Parameters use the dataset's units (cells for
width and depth, degrees for azimuth); any parameter left out takes the
environment's typical (median training) value. ``model.parameters(env)`` lists
them with their training ranges.

Sampling is the paper's: Heun's method, 100 steps, classifier-free guidance 3.
64-cubes use global attention as in training; fields use sliding-window
attention (radius 12 tokens) over the whole extent in one pass, so any size
works without tiling.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models.dit3d import DiT3D

REPO_ID = 'SciLM/ResFlow'
VOLUME_SHAPE = (64, 64, 32)

# Short names accepted by generate(); the full dataset names work too.
ENV_ALIASES = {
    'lobe': 'lobe',
    'pv_shoestring': 'channel:PV_SHOESTRING', 'shoestring': 'channel:PV_SHOESTRING',
    'cb_labyrinth': 'channel:CB_LABYRINTH', 'labyrinth': 'channel:CB_LABYRINTH',
    'cb_jigsaw': 'channel:CB_JIGSAW', 'jigsaw': 'channel:CB_JIGSAW',
    'sh_distal': 'channel:SH_DISTAL',
    'sh_proximal': 'channel:SH_PROXIMAL',
    'meander': 'channel:MEANDER_OXBOW', 'meander_oxbow': 'channel:MEANDER_OXBOW',
    'delta': 'delta',
}


@dataclass
class Well:
    """A vertical well through column (x, y): ``facies[z]`` is 1 for sand, 0
    for mud and -1 where nothing was observed (length = the volume's depth).
    For inclined or partial trajectories pass ``observed=`` instead."""
    x: int
    y: int
    facies: object


class PretrainedResFlow:
    """A loaded ResFlow model. Build it with :func:`load_pretrained`."""

    def __init__(self, net: DiT3D, config: dict, device):
        self.net, self.config, self.device = net, config, torch.device(device)
        self.layer_types = list(config['layer_types'])
        self.cont_cols = list(config['cont_cols'])
        self.n_universal = int(config['n_universal'])
        self.cont_min = np.asarray(config['cont_min'], np.float32)
        self.cont_max = np.asarray(config['cont_max'], np.float32)
        self.defaults = config['env_defaults']
        self.ranges = config['env_ranges']
        self.sampler = config['sampler']

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, name_or_path: str = REPO_ID, device=None, revision=None):
        """Load from a Hugging Face model repo id (default ``SciLM/ResFlow``)
        or a local directory holding ``config.json`` and ``model.safetensors``."""
        from safetensors.torch import load_file
        p = Path(name_or_path)
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(name_or_path, revision=revision,
                                       allow_patterns=['config.json', 'model.safetensors']))
        config = json.loads((p / 'config.json').read_text())
        arch = {k: tuple(v) if isinstance(v, list) else v for k, v in config['architecture'].items()}
        net = DiT3D(**arch)
        net.load_state_dict(load_file(str(p / 'model.safetensors')))
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        return cls(net.to(device).eval(), config, device)

    # --------------------------------------------------------------- conditions
    @property
    def environments(self):
        """Short environment names accepted by :meth:`generate`."""
        return ['lobe', 'pv_shoestring', 'cb_labyrinth', 'cb_jigsaw', 'sh_distal',
                'sh_proximal', 'meander', 'delta']

    def _env(self, environment):
        name = ENV_ALIASES.get(str(environment).lower(), environment)
        if name not in self.layer_types:
            raise ValueError(f'unknown environment {environment!r}; use one of {self.environments}')
        return name

    def parameters(self, environment):
        """Parameters of ``environment`` with their typical value and training
        range: ``{name: (typical, low, high)}``. Azimuth (degrees, any value,
        default 0 = flow along +x) applies to every environment."""
        env = self._env(environment)
        return {c: (self.defaults[env][c], *self.ranges[env][c]) for c in self.defaults[env]}

    def _raw(self, environment, params):
        """Validated raw parameters: (layer index, azimuth, {column: value})."""
        env = self._env(environment)
        params = dict(params)
        azimuth = params.pop('azimuth', 0.0)
        allowed = self.defaults[env]
        bad = [k for k in params if k not in allowed]
        if bad:
            raise ValueError(f'{env} does not take {bad}; its parameters are '
                             f'{["azimuth"] + list(allowed)}')
        raw = {**allowed, **params}
        for k, v in params.items():
            lo, hi = self.ranges[env][k]
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if vmin < lo - 1e-6 or vmax > hi + 1e-6:
                warnings.warn(f'{env}: {k} outside the training range [{lo:.3g}, {hi:.3g}]; '
                              'the model was never trained there', stacklevel=3)
        return self.layer_types.index(env), azimuth, raw

    def _cond_array(self, layer_idx, azimuth, raw, n):
        """(n, 18) condition rows from scalars or flattened per-token arrays."""
        onehot = np.zeros((n, len(self.layer_types)), np.float32)
        onehot[:, layer_idx] = 1.0
        cont = np.zeros((n, len(self.cont_cols)), np.float32)
        for k, col in enumerate(self.cont_cols):
            if col in raw:
                v = np.broadcast_to(np.asarray(raw[col], np.float32), (n,))
                cont[:, k] = (v - self.cont_min[k]) / (self.cont_max[k] - self.cont_min[k] + 1e-8)
        az = (np.broadcast_to(np.asarray(azimuth, np.float64), (n,)) % 360.0) / 360.0
        sin_a = np.sin(2 * math.pi * az).astype(np.float32)[:, None]
        cos_a = np.cos(2 * math.pi * az).astype(np.float32)[:, None]
        u = self.n_universal
        return np.concatenate([onehot, cont[:, :u], sin_a, cos_a, cont[:, u:]], axis=1)

    def condition(self, environment, **params):
        """The model's 18-number condition vector for one parameter set."""
        return self._cond_array(*self._raw(environment, params), 1)[0]

    # ---------------------------------------------------------------- generation
    def _observed(self, shape, n, observed, wells):
        """mask (n,1,*shape) = 1 at observed cells; known = +-1 there, 0 elsewhere."""
        obs = np.full((n,) + tuple(shape), -1, np.int8)
        if observed is not None:
            o = np.asarray(observed)
            if o.shape not in (tuple(shape), (n,) + tuple(shape)):
                raise ValueError(f'observed must have shape {tuple(shape)} or {(n,) + tuple(shape)}, got {o.shape}')
            obs[:] = o
        for w in wells or []:
            col = np.asarray(w.facies)
            if col.shape != (shape[2],):
                raise ValueError(f'well facies must have length {shape[2]}, got {col.shape}')
            obs[:, int(w.x), int(w.y), :] = np.where(col < 0, obs[:, int(w.x), int(w.y), :], col)
        mask = torch.from_numpy((obs >= 0).astype(np.float32))[:, None]
        known = torch.from_numpy(np.where(obs >= 0, obs * 2.0 - 1.0, 0.0).astype(np.float32))[:, None]
        return mask, known

    @torch.no_grad()
    def generate(self, environment, n=1, *, seed=0, wells=None, observed=None,
                 steps=None, guidance=None, batch_size=32, progress=True, **params):
        """Generate ``n`` volumes of 64 x 64 x 32 cells.

        environment : 'lobe', 'pv_shoestring', 'cb_labyrinth', 'cb_jigsaw',
                      'sh_distal', 'sh_proximal', 'meander' or 'delta'
        **params    : ntg, width_cells, depth_cells, azimuth and the
                      environment's family parameters (see parameters())
        wells       : list of Well, honoured exactly in every volume
        observed    : int array (64, 64, 32) or (n, 64, 64, 32); 1 sand, 0 mud,
                      -1 unobserved: any trajectory or set of known cells
        seed        : volume i starts from noise numpy.random.default_rng([seed, i])
        Returns uint8 array (n, 64, 64, 32), 1 = sand.
        """
        steps = steps or self.sampler['steps']
        guidance = self.sampler['guidance'] if guidance is None else guidance
        cond = torch.from_numpy(self._cond_array(*self._raw(environment, params), n))
        mask, known = self._observed(VOLUME_SHAPE, n, observed, wells)
        x0 = torch.from_numpy(np.stack([np.random.default_rng([seed, i]).standard_normal(
            VOLUME_SHAPE, dtype=np.float32) for i in range(n)]))[:, None]
        out = []
        bar = _bar(range(0, n, batch_size), progress, 'batches')
        for b in bar:
            sl = slice(b, b + batch_size)
            x = sample_cubes(self.net, x0[sl].to(self.device), cond[sl].to(self.device),
                             mask[sl].to(self.device), known[sl].to(self.device), steps, guidance)
            out.append((x[:, 0] > 0).to(torch.uint8).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def generate_field(self, environment, shape=(512, 512, 32), *, seed=0, wells=None,
                       observed=None, steps=None, guidance=None, radius=None,
                       progress=True, **params):
        """Generate one field of any size in a single pass (no tiling).

        shape    : (X, Y, Z) cells; Z = 32 as in training
        **params : scalars, or 2-D arrays of shape (X, Y) to vary a parameter
                   across the field (e.g. lobes that shrink down-fan)
        wells / observed : as in generate(), on the field's grid
        seed     : starting noise torch.Generator(device).manual_seed(seed)
        Returns uint8 array of ``shape``, 1 = sand.
        """
        steps = steps or self.sampler['steps']
        guidance = self.sampler['guidance'] if guidance is None else guidance
        radius = radius or self.sampler['field_radius']
        X, Y, Z = (tuple(shape) + (VOLUME_SHAPE[2],))[:3]
        layer_idx, azimuth, raw = self._raw(environment, params)
        px, py, pz = self.net.patch_size
        Px, Py, Pz = -(-X // px) * px, -(-Y // py) * py, -(-Z // pz) * pz
        gx, gy, gz = Px // px, Py // py, Pz // pz
        # Per-token parameters: maps are read at token centres.
        ix = np.minimum(np.arange(gx) * px + px // 2, X - 1)
        iy = np.minimum(np.arange(gy) * py + py // 2, Y - 1)

        def per_token(v):
            v = np.asarray(v, np.float64)
            if v.ndim == 0:
                return v
            if v.shape != (X, Y):
                raise ValueError(f'parameter maps must have shape {(X, Y)}, got {v.shape}')
            return np.repeat(v[np.ix_(ix, iy)][:, :, None], gz, axis=2).reshape(-1)

        n_tok = gx * gy * gz
        cond = self._cond_array(layer_idx, per_token(azimuth),
                                {k: per_token(v) for k, v in raw.items()}, n_tok)
        if all(np.ndim(v) == 0 for v in list(raw.values()) + [azimuth]):
            cond = cond[:1].repeat(n_tok, 0)          # constant: identical rows, as in the paper
        cond_tok = torch.from_numpy(cond)[None].to(self.device)
        mask, known = self._observed((X, Y, Z), 1, observed, wells)
        pad = (0, Pz - Z, 0, Py - Y, 0, Px - X)
        mask = torch.nn.functional.pad(mask, pad).to(self.device)
        known = torch.nn.functional.pad(known, pad).to(self.device)
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(1, 1, Px, Py, Pz, device=self.device, generator=g)
        with sliding_attention(self.net, radius):
            x = sample_field(self.net, x, cond_tok, mask, known, steps, guidance, progress)
        return (x[0, 0, :X, :Y, :Z] > 0).to(torch.uint8).cpu().numpy()


def load_pretrained(name_or_path: str = REPO_ID, device=None, revision=None) -> PretrainedResFlow:
    """Load the paper's ResFlow model (default: huggingface.co/SciLM/ResFlow).
    ``device`` defaults to CUDA when available, else CPU (works, but slowly)."""
    return PretrainedResFlow.from_pretrained(name_or_path, device=device, revision=revision)


# ---------------------------------------------------------------------- samplers
def sample_cubes(net, x, cond, mask, known, steps=100, guidance=3.0):
    """Heun + classifier-free guidance on 64-cubes, the paper's ResBench sampler:
    the conditional and unconditional passes are separate calls, wells enter
    through the inpainting context and are written back after the last step."""
    amp = x.is_cuda
    net.set_inpaint_context(mask, known)
    B = x.shape[0]

    def vel(xs, t):
        tt = torch.full((B,), t, device=xs.device) * 1000
        with torch.autocast('cuda' if amp else 'cpu', dtype=torch.bfloat16, enabled=amp):
            v_c = net(xs, tt, cond).float()
            v_u = net(xs, tt).float()
        return v_u + guidance * (v_c - v_u)

    dt = 1.0 / steps
    for s in range(steps):
        t = s * dt
        v = vel(x, t)
        v = 0.5 * (v + vel(x + v * dt, t + dt))
        x = x + v * dt
    net.clear_inpaint_context()
    return x * (1 - mask) + known * mask


def sample_field(net, x, cond_tok, mask, known, steps=100, guidance=3.0, progress=False,
                 max_batched_tokens=400_000):
    """Heun + classifier-free guidance over a whole field with per-token
    conditions, the paper's field sampler (resflow.assembly.wholefield):
    both guidance passes run as one batch of two up to ``max_batched_tokens``."""
    amp = x.is_cuda
    n_tok = cond_tok.shape[1]
    batched = guidance > 0 and n_tok <= max_batched_tokens
    reps = 2 if batched else 1
    net.set_inpaint_context(mask.repeat(reps, 1, 1, 1, 1), known.repeat(reps, 1, 1, 1, 1))
    drop = torch.tensor([False, True], device=x.device)
    cond2 = cond_tok.repeat(2, 1, 1) if batched else None

    def vel(xs, t):
        with torch.autocast('cuda' if amp else 'cpu', dtype=torch.bfloat16, enabled=amp):
            if batched:
                tt = torch.full((2,), t, device=xs.device) * 1000
                v = net(xs.repeat(2, 1, 1, 1, 1), tt, cond2, drop_mask=drop).float()
                return v[1:2] + guidance * (v[0:1] - v[1:2])
            tt = torch.full((1,), t, device=xs.device) * 1000
            if guidance > 0:
                v_c = net(xs, tt, cond_tok).float()
                v_u = net(xs, tt).float()
                return v_u + guidance * (v_c - v_u)
            return net(xs, tt, cond_tok).float()

    dt = 1.0 / steps
    for s in _bar(range(steps), progress, 'steps'):
        t = s * dt
        v = vel(x, t)
        v = 0.5 * (v + vel(x + v * dt, t + dt))
        x = x + v * dt
    net.clear_inpaint_context()
    return x * (1 - mask) + known * mask


class sliding_attention:
    """Context manager: switch the model to sliding-window attention of
    ``radius`` tokens per axis (the trained 64-cube mode is global)."""

    def __init__(self, net, radius):
        self.net, self.radius = net, (radius,) * 3 if isinstance(radius, int) else tuple(radius)

    def __enter__(self):
        self.saved = (self.net.attn_mode, self.net.attn_radius)
        self.net.attn_mode, self.net.attn_radius, self.net._mask_cache = 'sliding', self.radius, {}
        return self.net

    def __exit__(self, *exc):
        (self.net.attn_mode, self.net.attn_radius), self.net._mask_cache = self.saved, {}
        return False


def _bar(it, on, unit):
    if not on:
        return it
    try:
        from tqdm.auto import tqdm
        return tqdm(it, unit=unit, leave=False)
    except ImportError:
        return it
