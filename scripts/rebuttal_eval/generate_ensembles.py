"""Phase 4: generate the two evaluation ensembles with the paper's Table 6
inference settings.

  (a) --ensemble a : parameter-conditioned, unconditional on wells. One sample
      per manifest row (512/env), same conditioning vector as the reference
      instance, per-row fresh noise seed. 4,096 volumes.
  (b) --ensemble b : well-conditioned exactitude ensemble. First 256 rows/env;
      wells extracted from the reference volume with the paper's Figure 3
      builder (vertical full-depth wells at y=32), config assigned in the
      manifest (row mod 3 -> 1well/2wells/3wells). Masks saved alongside.

Inference settings (Euler ODE, NFE, CFG scale) are imported from the repo's
paper-figure config (examples/reservoirs/paper_figures/figure2.py) — not
hardcoded. The Euler/CFG loop is replicated here only to inject the per-row
seeded x0 (resflow's FlowMatching.sample draws its own noise); --self-test
verifies the replica is bit-identical to the library sampler.

Noise seeding: sample k of a condition uses torch CPU seed
fresh_noise_seed * 1000 + k  (k = 0 for the default --samples-per-condition 1).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from resflow.models.unet3d import UNet3D                      # noqa: E402
from resflow.methods.flow_matching import FlowMatching        # noqa: E402
from resflow.utils.masking import apply_inpaint_output        # noqa: E402
from resflow.utils.data_reservoirs import VOLUME_SHAPE        # noqa: E402

# -- Table 6 inference settings, read from the repo's figure config ----------
_FIGDIR = REPO / 'examples' / 'reservoirs' / 'paper_figures'


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, _FIGDIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_FIGDIR))  # figure3 does `import figure2`
    spec.loader.exec_module(mod)
    return mod


fig2 = _load_module('figure2')
fig3 = _load_module('figure3')
N_STEPS = fig2.N_STEPS      # 50
CFG = fig2.CFG              # 3.0

WELL_FRACTIONS = {          # the paper's three published Figure 3 configs
    '1well': [0.5],
    '2wells': [0.33, 0.66],
    '3wells': [0.25, 0.5, 0.75],
}

DEFAULT_CKPT_DIR = os.path.join(
    os.environ.get('WORK', '.'),
    'genflows_runs_backup_ls6/reservoirs_inpainting/checkpoints')


def build_well_mask(config):
    """Figure-3 XZ-row mask (vertical full-depth wells at y=32) via the
    paper's own builder, parameterized by its --well-fractions mechanism."""
    fig3.WELL_FRACTIONS = list(WELL_FRACTIONS[config])
    return fig3.build_xz_mask(fig3.Y_SLICE_DEFAULT)   # (1, X, Y, Z)


def euler_cfg_sample(model, x0, cond, cfg_scale=CFG, n_steps=N_STEPS):
    """Verbatim replica of resflow FlowMatching.sample (Euler + CFG) with an
    externally supplied x0. Verified against the library in --self-test."""
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        t_emb = t * 1000
        v_cond = model(x, t_emb, cond)
        v_uncond = model(x, t_emb)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + v * dt
    return x


def self_test(model, device):
    """Replica sampler must match FlowMatching.sample bit-for-bit when both
    consume the same RNG stream."""
    method = FlowMatching(model)
    cond = torch.rand(2, 18, device=device)
    shape = (2, 1, *VOLUME_SHAPE)
    model.set_inpaint_context(torch.zeros(shape, device=device),
                              torch.zeros(shape, device=device))
    torch.manual_seed(0)
    ref = method.sample(shape, device, cond=cond, cfg_scale=CFG, n_steps=N_STEPS)
    torch.manual_seed(0)
    x0 = torch.randn(shape, device=device)
    ours = euler_cfg_sample(model, x0, cond)
    assert torch.equal(ref, ours), 'euler_cfg_sample deviates from resflow sampler'
    print('self-test OK: replica sampler is bit-identical to FlowMatching.sample')


def seeded_noise(seeds, shape):
    """Per-row CPU-seeded x0, independent of batch composition."""
    outs = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        outs.append(torch.randn((1, *shape), generator=g))
    return torch.cat(outs)


def load_reference(ref_dir, slug):
    files = sorted(Path(ref_dir, slug).glob('volumes_*.npz'))
    ids, vols = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ids += list(d['ids'])
        vols.append(d['volumes'])
    return {i: v for i, v in zip(ids, np.concatenate(vols))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--conds', required=True, help='conds.npz from Phase 2')
    ap.add_argument('--ref-dir', help='reference volume dir (ensemble b)')
    ap.add_argument('--ckpt',
                    default=os.path.join(DEFAULT_CKPT_DIR, 'flow_matching.pt'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ensemble', choices=['a', 'b'], required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--samples-per-condition', type=int, default=1)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(in_channels=3, out_channels=1, num_cond=18,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=True))
    model.eval()
    if args.self_test:
        with torch.no_grad():
            self_test(model, device)

    mf = pd.read_csv(args.manifest, keep_default_na=False)
    cz = np.load(args.conds, allow_pickle=True)
    cond_by_id = {i: c for i, c in zip(cz['ids'], cz['cond'])}

    if args.ensemble == 'b':
        assert args.ref_dir, '--ref-dir required for ensemble b'
        mf = mf[mf['well_config'] != ''].copy()

    out = Path(args.out_dir)
    t_start = time.time()
    n_total = 0
    for lt, grp in mf.groupby('environment', sort=False):
        grp = grp.sort_values('row_index').reset_index(drop=True)
        slug = lt.replace(':', '_')
        refs = load_reference(args.ref_dir, slug) if args.ensemble == 'b' else None

        gen_vols, gen_ids, gen_masks = [], [], []
        for k in range(args.samples_per_condition):
            for lo in range(0, len(grp), args.batch_size):
                batch = grp.iloc[lo:lo + args.batch_size]
                B = len(batch)
                shape = (B, 1, *VOLUME_SHAPE)
                cond = torch.from_numpy(
                    np.stack([cond_by_id[i] for i in batch['row_id']])).to(device)
                x0 = seeded_noise(
                    batch['fresh_noise_seed'].to_numpy() * 1000 + k,
                    (1, *VOLUME_SHAPE)).to(device)

                if args.ensemble == 'a':
                    mask = torch.zeros(shape, device=device)
                    known = torch.zeros(shape, device=device)
                else:
                    mask = torch.stack(
                        [build_well_mask(c) for c in batch['well_config']]
                    ).to(device)
                    ref_pm = torch.stack([
                        torch.from_numpy(
                            refs[i].astype(np.float32) * 2.0 - 1.0).unsqueeze(0)
                        for i in batch['row_id']]).to(device)
                    known = ref_pm * mask

                with torch.no_grad():
                    model.set_inpaint_context(mask, known)
                    x = euler_cfg_sample(model, x0, cond)
                    if args.ensemble == 'b':
                        x = apply_inpaint_output(x, mask, known)
                model.clear_inpaint_context()

                gen_vols.append((x[:, 0] > 0).to(torch.int8).cpu().numpy())
                sfx = '' if args.samples_per_condition == 1 else f'#{k}'
                gen_ids += [i + sfx for i in batch['row_id']]
                if args.ensemble == 'b':
                    gen_masks.append(
                        mask[:, 0].to(torch.uint8).cpu().numpy())
                n_total += B
                print(f'{lt} k={k} rows {lo}-{lo + B - 1}: '
                      f'{time.time() - t_start:.0f}s elapsed', flush=True)

        vols = np.concatenate(gen_vols)
        d = out / f'ensemble_{args.ensemble}' / slug
        d.mkdir(parents=True, exist_ok=True)
        tag = f'r0000-r{len(grp) - 1:04d}'
        np.savez_compressed(d / f'volumes_{tag}.npz',
                            ids=np.array(gen_ids), volumes=vols)
        if args.ensemble == 'b':
            md = out / 'ensemble_b_masks' / slug
            md.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(md / f'masks_{tag}.npz',
                                ids=np.array(gen_ids),
                                masks=np.concatenate(gen_masks))
        print(f'{lt}: saved {vols.shape} ntg_mean={vols.mean():.4f}')

    gm = {
        'ensemble': args.ensemble,
        'n_volumes': n_total,
        'n_steps': N_STEPS, 'cfg_scale': CFG,
        'solver': 'explicit Euler (verbatim FlowMatching.sample replica)',
        'ckpt': args.ckpt,
        'samples_per_condition': args.samples_per_condition,
        'batch_size': args.batch_size,
        'noise_rule': 'torch CPU Generator seed = fresh_noise_seed*1000 + k',
        'binarization': 'x > 0',
        'hard_replacement': args.ensemble == 'b',
        'wall_clock_s': round(time.time() - t_start, 1),
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
    }
    p = out / f'generation_manifest_{args.ensemble}.json'
    p.write_text(json.dumps(gm, indent=2))
    print(f'done: {n_total} volumes in {gm["wall_clock_s"]}s -> {p}')


if __name__ == '__main__':
    main()
