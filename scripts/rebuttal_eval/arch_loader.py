"""One architecture-agnostic checkpoint loader for every evaluation generator.

Each generator used to hardcode ``UNet3D(...)`` plus a bare ``load_state_dict``.
That silently made the whole benchmark UNet-only: a DiT checkpoint could not be
loaded by any of them, so no DiT was ever scored on the main ResBench master
table or on the Addendum A/B/C entropy components -- only on the assembly
addendum, whose generator happened to use the tier-2 loader.

ResBench itself is model-agnostic (it scores directories of volumes, per
EVAL.md section 8). The lock-in was entirely on the generation side, so the fix
belongs here: one loader, used by all generators, that dispatches on what the
checkpoint actually contains.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'scripts' / 'tier2'))

from model_factory import load_checkpoint as _load_checkpoint  # noqa: E402


def load_any(ckpt, device, cond_dim=18, volume_shape=(64, 64, 32), verbose=True):
    """Load `ckpt` as whatever architecture it actually is, ready for eval."""
    model, arch = _load_checkpoint(ckpt, cond_dim, volume_shape, device)
    model.eval()
    if verbose:
        print(f'loaded {Path(ckpt).name} arch={arch}', flush=True)
    return model, arch
