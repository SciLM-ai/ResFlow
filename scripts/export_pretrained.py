"""Export a trained ResFlow checkpoint as a Hugging Face model folder.

Writes model.safetensors (the EMA inference weights) and config.json (architecture,
condition normalisation, per-environment typical values and training ranges, sampler
defaults), which resflow.load_pretrained() reads. The paper model:

  python scripts/export_pretrained.py \
      --ckpt specialist_runs/found8_regen/checkpoints/inference_epoch040.pt \
      --data-dir $SCRATCH/SiliciclasticReservoirs --out hf/ResFlow

Typical values are medians, and ranges min/max, of each environment's parameters over
its params_slim shards; the condition bounds are assets/cond_stats.npz, the fixed
normalisation the model was trained with.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts' / 'tier2'))
import model_factory as mf                                             # noqa: E402
from resflow.utils.data_reservoirs import UNIVERSAL_CONT, CONT_COLS    # noqa: E402

SLUG = {'lobe': 'lobe', 'channel:PV_SHOESTRING': 'channel_pv_shoestring', 'channel:CB_LABYRINTH': 'channel_cb_labyrinth',
        'channel:CB_JIGSAW': 'channel_cb_jigsaw', 'channel:SH_DISTAL': 'channel_sh_distal',
        'channel:SH_PROXIMAL': 'channel_sh_proximal', 'channel:MEANDER_OXBOW': 'channel_meander_oxbow', 'delta': 'delta'}


def env_stats(data_dir, env):
    fs = sorted(glob.glob(f'{data_dir}/{SLUG[env]}/shard_*/params_slim.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in fs])
    cols = [c for c in CONT_COLS if c in df.columns]
    return ({c: round(float(np.median(df[c])), 4) for c in cols},
            {c: [round(float(df[c].min()), 4), round(float(df[c].max()), 4)] for c in cols}, len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats', default=str(REPO / 'assets' / 'cond_stats.npz'))
    a = ap.parse_args()

    state = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    assert mf.infer_arch_from_state(state) == 'dit'
    hidden, depth, patch, heads, qk_norm, conv_io = mf.dit_dims_from_state(state)
    pos, theta, window, wshift, tconv, tkern, amode, aradius = mf.dit_pos_from_state(state)
    arch = dict(in_channels=int(mf.infer_shape_from_state(state, 'dit')['in_channels']), out_channels=1,
                volume_shape=[64, 64, 32], patch_size=list(patch), hidden=hidden, depth=depth, num_heads=heads,
                num_cond=18, num_time_embs=1, expand_angle_idx=None, qk_norm=bool(qk_norm), conv_io=conv_io,
                pos_embed=pos, rope_theta=float(theta), window=window, window_shift=bool(wshift),
                token_conv=int(tconv), token_conv_kernel=list(tkern), attn_mode=amode, attn_radius=list(aradius))
    st = np.load(a.stats, allow_pickle=True)
    layer_types = [str(x) for x in st['layer_types']]
    defaults, ranges = {}, {}
    for env in layer_types:
        defaults[env], ranges[env], n = env_stats(a.data_dir, env)
        print(env, n, defaults[env])
    config = {
        'model_type': 'resflow-dit3d',
        'architecture': arch,
        'layer_types': layer_types,
        'cont_cols': list(CONT_COLS), 'n_universal': len(UNIVERSAL_CONT),
        'cont_min': [float(v) for v in st['cont_min']], 'cont_max': [float(v) for v in st['cont_max']],
        'env_defaults': defaults, 'env_ranges': ranges,
        'sampler': {'solver': 'heun', 'steps': 100, 'guidance': 3.0, 'field_radius': 12},
        'checkpoint': Path(a.ckpt).name,
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(out / 'model.safetensors'))
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    print('wrote', out / 'model.safetensors', out / 'config.json',
          f'{sum(v.numel() for k, v in state.items() if not k.endswith("_buf")) / 1e6:.1f}M parameters')


if __name__ == '__main__':
    main()
