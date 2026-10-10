# ResFlow

ResFlow generates 3D reservoir facies (sand and mud) with one flow-matching model for eight simulated
siliciclastic environments: deep-water lobes, a distributary delta, and six fluvial classes. It is conditioned on
geological parameters (net-to-gross, body width and thickness, flow azimuth, and each environment's own controls)
and on wells, and it generates single 64×64×32 volumes or whole fields of any size in one pass.

**Weights:** [huggingface.co/SciLM/ResFlow](https://huggingface.co/SciLM/ResFlow) ·
**Dataset:** [SciLM/SiliciclasticReservoirs](https://huggingface.co/datasets/SciLM/SiliciclasticReservoirs) ·
**Benchmark:** [ResBench](https://github.com/SciLM-ai/ResBench) ·
**Simulator:** [ResMill](https://github.com/SciLM-ai/ResMill) ·
**Paper:** NeurIPS 2026, Track on Evaluations and Datasets ([citation](#citation))

## Quick start

```bash
pip install git+https://github.com/SciLM-ai/ResFlow
```

```python
import resflow

model = resflow.load_pretrained()                     # downloads the weights once (130 MB)

vols = model.generate('meander', n=8)                 # (8, 64, 64, 32) uint8, 1 = sand
vols = model.generate('lobe', n=4, ntg=0.35, width_cells=30, azimuth=45)
ens = model.generate('meander', n=32,                 # every realisation honours the well exactly
                     wells=[resflow.Well(x=32, y=32, facies=column)])
field = model.generate_field('lobe', shape=(512, 512, 32), ntg=0.5)
```

The walkthrough in [`examples/pretrained/quickstart.ipynb`](examples/pretrained/quickstart.ipynb) (or
`quickstart.py`) covers all eight environments, parameters, wells, and fields whose parameters vary across them.

A GPU is recommended: on an NVIDIA GH200 a 64×64×32 volume takes about 1.6 s in batches of 128 and a 512×512×32
field about 4 minutes. A CPU works, but takes tens of minutes per volume.

### Inputs

| Input | Meaning |
|---|---|
| environment | `lobe`, `pv_shoestring`, `cb_labyrinth`, `cb_jigsaw`, `sh_distal`, `sh_proximal`, `meander`, `delta` |
| `ntg` | sand fraction of the volume (0–1) |
| `width_cells`, `depth_cells` | characteristic body width and thickness, in cells |
| `azimuth` | flow direction in degrees, 0 = along +x (default 0) |
| `asp` | lobe aspect ratio (lobe) |
| `mCHsinu` | channel sinuosity (channels, delta) |
| `mFFCHprop` | fraction of abandoned channels plugged with mud (channels, delta) |
| `probAvulInside` | in-belt avulsion probability (channels, delta) |
| `trunk_length_fraction` | share of the trunk kept free of bifurcations (delta) |
| `wells` / `observed` | vertical `Well(x, y, facies)`, or an array of known cells: 1 sand, 0 mud, −1 unknown |

Parameters left out take the environment's typical value, and `model.parameters(env)` lists each one with its
training range. In `generate_field` any parameter can also be an (X, Y) map. Volumes have axes (x, y, z) with
z = 0 at the base; cells are 100 m laterally for lobes and 10 m for channels and deltas, and 1 m vertically. The
model was trained on simulated reservoirs only (see the paper's limitations); it has not been validated on real
subsurface data.

## Reproducing the NeurIPS 2026 paper (DiT3D, ResBench v1)

The paper model is a 32.6M-parameter 3D DiT (4×4×2 patches, 12 blocks, width 384, 6 heads, QK-norm, 3D RoPE with
θ = 100, adaLN) trained with conditional flow matching on all eight environments of SiliciclasticReservoirs.
`assets/cond_stats.npz` holds the fixed min–max bounds of the 18-D condition vector; training and generation both
read it. The released weights are the final (epoch-40) EMA checkpoint of this run, exported with
`scripts/export_pretrained.py`; `tests/test_pretrained.py` checks that `resflow.load_pretrained()` reproduces these
generators exactly.

```bash
# Train (16 GPUs, one per node, DDP): global batch 512, AdamW (β2 0.95), peak LR 5.77e-4, 40 epochs
torchrun ... scripts/tier2/train_assembly.py --data-mode native64 --seed 2026 --arch dit --dit-pos rope \
    --dit-patch 4 4 2 --dit-rope-theta 100 --masked-loss --amp bf16 --beta2 0.95 --ema-warmup --save-raw --context-share 0 \
    --envs lobe,channel:PV_SHOESTRING,channel:CB_LABYRINTH,channel:CB_JIGSAW,channel:SH_DISTAL,channel:SH_PROXIMAL,channel:MEANDER_OXBOW,delta \
    --micro-batch 32 --global-batch 512 --lr 5.77e-4 --epochs 40 --total-epochs 40 --save-every 1 \
    --subset-size 900000 --num-workers 8 --run-dir RUN

# ResBench v1 submission (Heun 100 steps, CFG 3): 64×64×32 tasks, then whole fields with sliding-window attention
export RESBENCH_REF=/path/to/resbench_v1_ref SILICICLASTIC_ROOT=/path/to/SiliciclasticReservoirs
python scripts/resbench/gen_v1_cubes.py  --ckpt RUN/checkpoints/inference_epochNNN.pt --env <env> --part uncond --out SUB
python scripts/resbench/gen_v1_cubes.py  --ckpt RUN/checkpoints/inference_epochNNN.pt --env <env> --part well   --out SUB
python scripts/resbench/gen_v1_fields.py --ckpt RUN/checkpoints/inference_epochNNN.pt --env <env> --radius 12 --compile --out SUB
resbench score SUB --reference $RESBENCH_REF          # https://github.com/SciLM-ai/ResBench
```

Whole-field generation uses `set_inference_attention(model, 12)` (FlexAttention, each token attends to ±12 tokens per
axis); the model is trained with global attention on 64×64×32 volumes.

The paper's other experiments use the same script and flags (`COMMON` below), each scored with the same
generators and ResBench; every run is scored at its final epoch.

```bash
COMMON="--data-mode native64 --seed 2026 --arch dit --dit-pos rope --dit-patch 4 4 2 --dit-rope-theta 100 \
  --masked-loss --amp bf16 --beta2 0.95 --ema-warmup --context-share 0 --num-workers 8 --compile-model --micro-batch 64"

# Single-environment specialists (Table 2): one environment's full training split, the paper recipe.
# --subset-size 180000 for lobe, 135000 for channel:CB_JIGSAW and delta, 90000 for the five others.
torchrun ... scripts/tier2/train_assembly.py $COMMON --envs channel:PV_SHOESTRING --subset-size 90000 \
    --global-batch 512 --lr 5.77e-4 --epochs 40 --total-epochs 40 --run-dir SPEC

# Held-out environment: pretrain on the seven others (810k volumes), then fine-tune on 1k or 10k meander volumes.
torchrun ... scripts/tier2/train_assembly.py $COMMON --subset-size 810000 \
    --envs lobe,channel:PV_SHOESTRING,channel:CB_LABYRINTH,channel:CB_JIGSAW,channel:SH_DISTAL,channel:SH_PROXIMAL,delta \
    --global-batch 512 --lr 5.77e-4 --epochs 40 --total-epochs 40 --run-dir HO
torchrun ... scripts/tier2/train_assembly.py $COMMON --envs channel:MEANDER_OXBOW \
    --init-from HO/checkpoints/inference_epoch040.pt --subset-size 1024 \
    --global-batch 128 --lr 2.89e-4 --epochs 1500 --total-epochs 1500 --run-dir FT1K   # 10k: --subset-size 10240 --epochs 150 --total-epochs 150

# Dataset-size curve: from scratch on meander only, 12k steps at batch 128 for every size
# (--subset-size 1024 / 10240 / 30720 / 90000 with --epochs and --total-epochs 1500 / 150 / 50 / 17).
torchrun ... scripts/tier2/train_assembly.py $COMMON --envs channel:MEANDER_OXBOW --subset-size 1024 \
    --global-batch 128 --lr 2.89e-4 --epochs 1500 --total-epochs 1500 --run-dir SC1K
```

## Earlier research code

The sections below document the code that preceded the paper model: UNet inpainting models, tiled multi-block assembly, and the method comparison on a smaller lobe benchmark. It is kept for reference.

## Highlights

- **Eight reservoir architectures from one model.** A single Flow-Matching network handles channel-belt jigsaw, channel-belt labyrinth, meander-oxbow, point-bar shoestring, sheet-distal, sheet-proximal, lobe, and delta layer types via a layer-type one-hot in the conditioning vector.
- **Well-conditioned inpainting.** The same weights produce unconditional samples (empty mask) or well-conditioned samples (1–N vertical/L-shaped wells) — no separate "inpainting" model. Hard replacement at the final denoising step guarantees exact agreement at known voxels.
- **Big-reservoir multi-block assembly.** Parallel block denoising with overlap blending lets you generate reservoirs of arbitrary plan size (e.g. 600×600×32) by tiling 64×64×32 blocks with hard or soft transitions between layer types. See `resflow/assembly/big_reservoir_multi.py`.
- **Round-trip property evaluation.** A separately-trained CNN3D property predictor scores whether generated samples actually preserve the requested NTG, geometry, and azimuth.
- **Method comparison.** Diffusion (DDPM/DDIM), Flow Matching, MeanFlow, and Rectified Flow share an apples-to-apples training loop on a smaller geological lobe benchmark. Flow Matching was chosen for the reservoir scaling work because it gave the best tradeoff between sample quality, NFE, and CFG stability — see "Methods" below.

## Training the earlier UNet models

```bash
pip install -e .
```

End-to-end on the SiliciclasticReservoirs dataset (`SciLM/SiliciclasticReservoirs`):

```bash
cd examples/reservoirs/inpainting

# Train the well-conditioned inpainting model (auto-downloads dataset to $SCRATCH on first run)
python train.py                        # single GPU
sbatch run_A100.sh                     # 4 nodes × 3 A100s on a 3-A100-per-node HPC
sbatch run_GH200.sh                    # 8 nodes × 1 GH200 on a GH200 HPC

# Sample one cube per layer type, with and without 5-well "+"-pattern conditioning
python sample_30min_demo.py

# Build a 10×10 big reservoir of, say, lobe blocks with mixed scalars
cd ../big_reservoir/lobes
python generate.py
python visualize.py
```

## Datasets

| Dataset | Volumes | Conditioning | Role |
|---|---|---|---|
| **SiliciclasticReservoirs** | 1M binary 64×64×32 cubes across 8 layer-type families | layer-type one-hot + 4 universal scalars (NTG, width, depth, azimuth) + 5 family-specific scalars | **Headline.** Used for inpainting + big-reservoir assembly. |
| **Lobes** | ~89k binary 50×50×50 single-lobe cubes | 5 continuous scalars (height, radius, aspect ratio, angle, NTG) | Smaller geological benchmark used during method comparison. |

## What's in here

```
resflow/
├── models/
│   ├── unet3d.py            # 3D UNet, continuous-cond + learned null embedding for CFG
│   └── cnn3d.py             # 3D CNN property predictor (round-trip evaluation)
├── methods/
│   ├── diffusion.py         # DDPM + DDIM sampling, x0-clipping for stable CFG
│   ├── flow_matching.py     # OT Flow Matching, Euler integration  ← used for reservoirs
│   ├── meanflow.py          # MeanFlow with JVP-based training, 1-step capable
│   └── rectified_flow.py    # 2-Rectified Flow with forward/backward/bidirectional reflow
├── utils/
│   ├── data_reservoirs.py   # Sharded loader for SiliciclasticReservoirs
│   ├── data_lobes.py        # Lobe dataset + on-the-fly inpaint mask wrapper
│   ├── masking.py           # Wells / boundaries / cross-sections (3D inpainting)
│   ├── training.py          # AdamW + cosine LR + EMA + train_model_inpaint
│   └── plotting{,_lobes}.py # Cross-section grids, wells overlay, loss curves
└── assembly/
    └── big_reservoir_multi.py  # Multi-block parallel denoising with overlap blending

examples/
├── reservoirs/                # ← centerpiece
│   ├── inpainting/            #   training, sampling, eval-loss, 30-min smoke run
│   └── big_reservoir/         #   per-layer-type uniform/long generators + 3-type sequence
├── lobes/                     # smaller geological benchmark
│   ├── standard/              #   unconditional + class-CFG generation
│   ├── inpainting/            #   wells / boundaries / cross-sections inpainting
│   ├── CNN/                   #   CNN3D property predictor train/evaluate
│   └── big_reservoir/         #   lobe-only multi-block assembly (precursor to reservoirs/)
```

## Reservoir details

### Conditioning vector (18-D)

```
[ layer_one_hot (8) | NTG | width_cells | depth_cells | sin(az) | cos(az) |
  asp | mCHsinu | mFFCHprop | probAvulInside | trunk_length_fraction ]
```

The five family-specific scalars are zeroed for layer types that don't use them. Universal scalars are normalized to the global per-feature min/max from the training cond cache. Azimuth is encoded as `(sin, cos)` for 360° periodicity.

### Inpainting via channel concatenation

The 3D UNet takes 3 input channels — `[noisy_x, known_data, mask]` — and outputs 1 channel. Inpaint context is set on the model via stateful `set_inpaint_context(mask, data)` / `clear_inpaint_context()`, so methods (`FlowMatching`, `Diffusion`, …) stay completely inpainting-unaware. Mask convention: `1 = known (keep)`, `0 = unknown (generate)`.

Training distribution mixes unconditional and well-conditioned samples (30% empty mask, 70% with 1–5 wells), so a single set of weights serves both modes at sampling time.

### Big-reservoir multi-block assembly

`generate_big_reservoir_multi` tiles a grid of `BlockSpec`s across X×Y, where each block has its own layer type and per-block scalars. Adjacent blocks share an overlap region (`overlap_xy ∈ {12, 16, 24}` in our experiments) — denoising is run jointly across all blocks in parallel, with the overlapping noise updated as a smooth blend of the contributing blocks at every step. Two transition modes:

- **Hard**: each block sees only its own conditioning everywhere, blending happens only in the noise update.
- **Soft**: blocks linearly interpolate cond vectors across overlaps, producing smoother facies transitions at the cost of a small in-distribution drift inside the overlap.

## Methods (comparison summary)

| Method | What it learns | Sampling | NFE / sample | Notes |
|---|---|---|---|---|
| **Diffusion** (DDPM/DDIM) | Noise prediction | Iterative denoising | ~50 | Linear β-schedule; DDIM clips x0 + recomputes ε for CFG stability |
| **Flow Matching** ✅ | Velocity field | Euler ODE | ~50 | Chosen for reservoirs |
| **MeanFlow** | Mean velocity | Single-step capable | 1 (embedded CFG) or 2/step | JVP target, EMA target network |
| **Rectified Flow** | Straightened velocity | Euler ODE | <50 | Forward / backward / bidirectional reflow on coupled pairs |

We picked Flow Matching for the reservoir work because it gave clean, stable samples at ~50 steps, integrated with channel-concat inpainting without any architectural changes, and produced no visible CFG drift even at high guidance scales.

## Architecture: model ↔ method decoupling

Methods and models communicate through a minimal interface:

```python
model(x, t, cond)                # conditional forward pass
model(x, t)                      # unconditional (model uses its own null representation)
model(x, t, cond, drop_mask=m)   # mixed batch for training-time CFG
```

Methods decide *when* to drop conditioning and *how* to combine cond/uncond at sampling time. Models decide *what* "unconditional" means internally (learned null embedding, null class token, …). Any method works with any model — including the inpainting variant, since inpaint context lives on the model, not the method.

## Reproducing the reservoir results

```bash
# 1. Train the FM-inpaint model on 1M cubes
cd examples/reservoirs/inpainting
sbatch run_A100.sh           # or run_GH200.sh

# 2. Sample 8-layer-type demo (one cube per type, with and without 5 wells)
python sample_30min_demo.py

# 3. Evaluate held-out FM loss on train/val/test
sbatch run_eval.sh

# 4. Build big reservoirs by layer family
cd ../big_reservoir
python setup_uniform.py             # regenerate per-family generate.py templates
python long_setup.py                # 1×10 long variants
cd lobes && python generate.py && python visualize.py

# 5. Train CNN3D property predictor and run round-trip eval
cd ../../lobes/CNN
python train.py
python evaluate.py
```

Checkpoints land in `$SCRATCH/genflows_runs/...` by default; override with `RESERVOIR_DATA_DIR` and the `--ckpt` flags shown in each script.

## Citation

```bibtex
@inproceedings{baghishov2026siliciclastic,
  title     = {SiliciclasticReservoirs: A Million-Reservoir Dataset and Flow-Matching Foundation Model
               for 3D Siliciclastic Reservoir Generation},
  author    = {Baghishov, Ilgar and Rustamzade, Elnara and Henkelman, Graeme and Foster, John T. and Pyrcz, Michael J.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```

## License

MIT, for the code and the pretrained weights.
