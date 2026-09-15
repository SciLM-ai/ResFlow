"""Every evaluation generator must load checkpoints architecture-agnostically.

Regression guard for a defect that silently made the whole benchmark UNet-only:
each generator hardcoded ``UNet3D(...)``, so a DiT checkpoint could not be
loaded and no DiT was ever scored on the ResBench master table or on the
Addendum A/B/C entropy components. ResBench is model-agnostic by design (it
scores directories of volumes); the lock-in was on the generation side.
"""
import ast
from pathlib import Path

import pytest

GEN_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'rebuttal_eval'
GENERATORS = sorted(GEN_DIR.glob('generate_*.py'))


def test_there_are_generators_to_check():
    assert GENERATORS, f'no generators found under {GEN_DIR}'


@pytest.mark.parametrize('path', GENERATORS, ids=lambda p: p.name)
def test_generator_does_not_hardcode_an_architecture(path):
    tree = ast.parse(path.read_text())
    bad = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {'UNet3D', 'UNetAttn3D', 'DiT3D'}
    ]
    assert not bad, (
        f'{path.name} instantiates {sorted(set(bad))} directly. Use '
        'arch_loader.load_any(ckpt, device) so any checkpoint can be scored.'
    )


@pytest.mark.parametrize('path', GENERATORS, ids=lambda p: p.name)
def test_generator_with_a_ckpt_flag_uses_the_shared_loader(path):
    src = path.read_text()
    if '--ckpt' not in src:
        pytest.skip('generator takes no checkpoint')
    # Both names resolve to the same dispatching loader: arch_loader.load_any
    # is a thin wrapper over model_factory.load_checkpoint. The assembly
    # generator used the latter directly, which is why it was the one component
    # a DiT could always be scored on.
    assert ('load_any' in src) or ('load_checkpoint' in src), (
        f'{path.name} takes --ckpt but never calls an architecture-dispatching '
        'loader (arch_loader.load_any / model_factory.load_checkpoint)'
    )
