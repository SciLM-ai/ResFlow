"""resflow.pretrained must reproduce the paper's generators exactly.

A tiny random DiT3D, exported in the Hugging Face layout (config.json +
model.safetensors), is loaded through load_pretrained() and compared with the
code the paper's numbers came from: build_cond_vector for the conditions,
scripts/resbench/gen_v1_cubes.heun_sample for 64-cubes and
resflow.assembly.wholefield.generate_wholefield for fields. Runs on CPU in
seconds, with no download.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'resbench'))
sys.path.insert(0, str(REPO / 'scripts' / 'rebuttal_eval'))
sys.path.insert(0, str(REPO / 'scripts' / 'tier2'))
sys.path.insert(0, str(REPO))

safetensors = pytest.importorskip('safetensors.torch')
from resflow import pretrained as rp                                         # noqa: E402
from resflow.assembly.big_reservoir_multi import (BlockSpec, build_cond_vector,  # noqa: E402
                                                  LAYER_TYPE_TO_IDX, LAYER_TYPES, CONT_COLS)
from resflow.assembly.wholefield import generate_wholefield                  # noqa: E402
from resflow.models.dit3d import DiT3D, _HAS_FLEX                            # noqa: E402

ARCH = dict(in_channels=3, out_channels=1, volume_shape=[64, 64, 32], patch_size=[4, 4, 2], hidden=32,
            depth=2, num_heads=2, num_cond=18, num_time_embs=1, expand_angle_idx=None, qk_norm=True,
            conv_io=False, pos_embed='rope', rope_theta=100.0, window=None, window_shift=False,
            token_conv=0, token_conv_kernel=[3, 3, 3], attn_mode='window', attn_radius=[8, 8, 8])


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp('resflow_tiny')
    torch.manual_seed(0)
    net = DiT3D(**{k: tuple(v) if isinstance(v, list) else v for k, v in ARCH.items()})
    for p in net.parameters():                       # adaLN gates start at zero; make the net non-trivial
        torch.nn.init.normal_(p, std=0.05)
    safetensors.save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(d / 'model.safetensors'))
    stats = np.load(REPO / 'assets' / 'cond_stats.npz', allow_pickle=True)
    defaults = {'lobe': {'ntg': 0.5, 'width_cells': 46.0, 'depth_cells': 12.5, 'asp': 1.75}}
    for env in LAYER_TYPES[1:]:
        defaults[env] = {'ntg': 0.4, 'width_cells': 10.0, 'depth_cells': 7.0, 'mCHsinu': 1.3,
                         'mFFCHprop': 0.2, 'probAvulInside': 0.1}
    defaults['delta']['trunk_length_fraction'] = 0.3
    config = {'architecture': ARCH, 'layer_types': list(LAYER_TYPES), 'cont_cols': list(CONT_COLS), 'n_universal': 3,
              'cont_min': [float(v) for v in stats['cont_min']], 'cont_max': [float(v) for v in stats['cont_max']],
              'env_defaults': defaults,
              'env_ranges': {e: {k: [0.0, 100.0] for k in v} for e, v in defaults.items()},
              'sampler': {'solver': 'heun', 'steps': 100, 'guidance': 3.0, 'field_radius': 12}}
    (d / 'config.json').write_text(json.dumps(config))
    return rp.load_pretrained(str(d), device='cpu')


def test_conditions_match_build_cond_vector(model):
    stats = np.load(REPO / 'assets' / 'cond_stats.npz', allow_pickle=True)
    for short in model.environments:
        env = model._env(short)
        raw = {**model.defaults[env], 'ntg': 0.3}
        ref = build_cond_vector(BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[env], azimuth_deg=37.5, raw_scalars=raw),
                                stats['cont_min'], stats['cont_max'])
        assert np.array_equal(model.condition(short, ntg=0.3, azimuth=37.5), ref), short


def test_bad_inputs_raise(model):
    with pytest.raises(ValueError):
        model.condition('meander', asp=1.5)          # lobe-only parameter
    with pytest.raises(ValueError):
        model.condition('river')


def test_cubes_match_paper_sampler(model):
    import gen_v1_cubes as G
    cond = torch.from_numpy(model.condition('meander')[None]).repeat(2, 1)
    x0 = torch.from_numpy(np.stack([np.random.default_rng([3, i]).standard_normal((64, 64, 32), dtype=np.float32)
                                    for i in range(2)]))[:, None]
    col = (np.arange(32) % 3 == 0).astype(np.int8)
    mask, known = model._observed((64, 64, 32), 2, None, [rp.Well(7, 9, col)])
    a = rp.sample_cubes(model.net, x0.clone(), cond, mask, known, steps=2)
    b = G.heun_sample(model.net, x0.clone(), cond, mask, known, 2, 3.0)
    assert torch.equal(a, b)
    v = model.generate('meander', n=2, seed=3, wells=[rp.Well(7, 9, col)], steps=2, progress=False)
    assert np.array_equal(v, (a[:, 0] > 0).numpy().astype(np.uint8))
    assert (v[:, 7, 9, :] == col).all()


@pytest.mark.skipif(not _HAS_FLEX, reason='sliding attention needs FlexAttention (torch>=2.5)')
def test_field_matches_paper_sampler(model):
    from model_factory import set_inference_attention
    stats = np.load(REPO / 'assets' / 'cond_stats.npz', allow_pickle=True)
    env, shape = 'channel:PV_SHOESTRING', (96, 64, 32)
    cond = build_cond_vector(BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[env], azimuth_deg=0.0,
                                       raw_scalars=dict(model.defaults[env])), stats['cont_min'], stats['cont_max'])
    set_inference_attention(model.net, 12)
    g = torch.Generator().manual_seed(5)
    ref, _ = generate_wholefield(model.net, cond, n_steps=2, cfg_scale=3.0, device='cpu', solver='heun',
                                 generator=g, verbose=False, field_shape=shape)
    set_inference_attention(model.net, None)
    out = model.generate_field('pv_shoestring', shape=shape, seed=5, steps=2, progress=False)
    assert np.array_equal(out, (np.asarray(ref) > 0).astype(np.uint8))
    assert model.net.attn_mode == 'window'


@pytest.mark.skipif(not _HAS_FLEX, reason='sliding attention needs FlexAttention (torch>=2.5)')
def test_field_parameter_maps_and_wells(model):
    shape = (64, 64, 32)
    col = (np.arange(32) % 4 == 0).astype(np.int8)
    width = np.linspace(30, 60, shape[0])[:, None].repeat(shape[1], 1)
    f = model.generate_field('lobe', shape=shape, seed=1, steps=1, progress=False,
                             width_cells=width, wells=[rp.Well(3, 4, col)])
    assert f.shape == shape and (f[3, 4] == col).all()
