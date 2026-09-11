# CLAUDE.md

## Project Overview

Generative modeling methods with a shared, modular architecture. Methods are pure math (model-agnostic); models handle all conditioning details. Currently supports two datasets:

- **MNIST** — 2D digit generation (32x32), class-conditional (10 digits)
- **Lobes** — 3D geological lobe generation (50x50x50 binary voxels), continuous-conditional (height, radius, aspect_ratio, angle, net-to-gross), with optional inpainting (wells, boundaries, cross-sections)
- **Reservoirs** — 3D siliciclastic-reservoir binary facies (64x64x32) from AnonymouScientist/SiliciclasticReservoirs (1M cubes across 8 architectures); cond is layer-type one-hot + universal scalars (ntg, width_cells, depth_cells, azimuth sin/cos) + family-specific scalars (asp / mCHsinu / mFFCHprop / probAvulInside / trunk_length_fraction; 0 outside family). Caption / poro_ave / perm_ave excluded.

Methods:
- **Diffusion** (DDPM + DDIM sampling, with x0-clipping for stable CFG)
- **Flow Matching** (Optimal Transport)
- **MeanFlow** (mean velocity with JVP-based training — paper included in `meanflow_paper_latex/`)
- **Rectified Flow** (Liu et al., "Flow Straight and Fast" — forward, backward, bidirectional reflow variants)

## Project Structure

```
resflow/
├── models/
│   ├── unet.py            # 2D UNet for images (class-conditional via learned embedding)
│   └── unet3d.py          # 3D UNet for volumes (continuous-conditional via MLP + learned null embedding)
├── methods/
│   ├── diffusion.py       # Diffusion: noise prediction, DDPM/DDIM sampling, 1000-step schedule
│   ├── flow_matching.py   # Flow Matching: velocity field prediction, Euler integration
│   ├── meanflow.py        # MeanFlow: mean velocity with JVP-based training
│   └── rectified_flow.py  # Rectified Flow: forward/backward/bidirectional reflow with coupled pairs
└── utils/
    ├── data.py            # MNIST loading (padded to 32x32, normalized to [-1,1])
    ├── data_lobes.py      # Lobe dataset: facies loading, NTG computation, filtering, normalization + inpaint wrapper
    ├── data_reservoirs.py # SiliciclasticReservoirs sharded loader (binary facies + slim params, layer-type aware cond cache)
    ├── masking_lobes.py   # Inpainting mask generation: wells, boundaries, cross-sections, combinations
    ├── training.py        # Training loop (AdamW, cosine LR, grad clipping, EMA with target network) + train_reflow + train_model_inpaint
    └── plotting.py        # Sample grids and loss curves

examples/
├── mnist/
│   ├── train.py           # Train all 8 MNIST models
│   └── sample.py          # Generate MNIST samples from checkpoints
└── lobes/
│   ├── data/              # facies.npy, parameters.csv, failed_cases.npy (shared)
│   ├── standard/          # Standard generation (no inpainting)
│   │   ├── train.py       # Train all 8 lobe models
│   │   ├── sample.py      # Generate 3D lobe samples from checkpoints
│   │   └── sample_and_plot.ipynb
│   └── inpainting/        # Inpainting (3-channel: noisy_x + known_data + mask)
│       ├── train.py       # Train inpainting models
│       ├── sample.py      # Generate inpainted 3D samples
│       └── sample_and_plot.ipynb
└── reservoirs/
    └── inpainting/        # Wells-only inpainting on SiliciclasticReservoirs
        ├── train.py       # Trains FM-inpaint on full 1M cubes; data dir defaults to $SCRATCH
        ├── train_30min.py # ~30-min smoke run on a 3-A100 node (50K subset)
        ├── sample_30min_demo.py  # Demo: 8 layer types × {uncond, 5-well "+"} -> 3 cross-sections
        ├── run_A100.sh    # 3-A100-per-node HPC launcher (4 nodes × 3 A100s)
        └── run_GH200.sh   # GH200 HPC launcher (8 nodes × 1 GH200)

meanflow_paper_latex/      # LaTeX source for the MeanFlow paper
meanflow.pdf               # Compiled paper
```

## Installation

```bash
pip install -e .
```

This installs the `resflow` package (ResFlow) and all dependencies (torch, torchvision, matplotlib, tqdm, accelerate, pandas, pyarrow, huggingface_hub) via `pyproject.toml`.

> **NEVER install into an existing conda env without asking the user first.**
> The `torch` dep in `pyproject.toml` is unpinned and *will* upgrade an
> existing torch (it has done so silently in the past, breaking other
> packages in shared envs). Default to proposing a **fresh project-named
> env** (e.g. `conda create -n resflow python=3.12 -y && conda activate
> resflow && pip install -e .`) and let the user confirm or pick a name.
> Convenience of an existing env that "happens to have torch" is **not** a
> reason to install there.
>
> **Default to Python 3.12 for new envs unless the user specifies otherwise.**

## Running

### MNIST (2D)
```bash
python examples/mnist/train.py      # Train all 8 models
python examples/mnist/sample.py     # Generate samples from checkpoints
accelerate launch examples/mnist/train.py  # Multi-GPU
```

### Lobes (3D)
```bash
cd examples/lobes/standard
python train.py                    # Train all 8 models
python sample.py                   # Generate 3D samples from checkpoints
accelerate launch train.py         # Multi-GPU
```

### Lobes Inpainting (3D)
```bash
cd examples/lobes/inpainting
python train.py                    # Train inpainting models
python sample.py                   # Generate inpainted samples
accelerate launch train.py         # Multi-GPU
```

### Reservoirs (3D, SiliciclasticReservoirs) — well-conditioned inpainting
```bash
cd examples/reservoirs/inpainting
# Dataset defaults to $SCRATCH/SiliciclasticReservoirs (auto-downloaded
# on first use). Override: RESERVOIR_DATA_DIR=/path/to/dataset python train.py
python train.py                    # Single GPU
sbatch run_A100.sh                 # 3-A100-per-node HPC (4 nodes × 3 A100s)
sbatch run_GH200.sh                # GH200 HPC (8 nodes × 1 GH200)
```
The 3-channel inpainting model handles **both** unconditional generation
(empty mask = 30% of training samples) and well-conditioned generation
(non-empty mask) from a single set of weights, so there is no separate
"standard" variant for reservoirs. First run builds a per-split cond
cache under `<data_dir>/_cond_cache/` (~10 s after the per-shard parquet
sweep) so subsequent runs start instantly.

## Architecture: Model ↔ Method Interface

Methods and models are decoupled through a minimal interface:

```python
model(x, t, cond)                # conditional forward pass
model(x, t)                      # unconditional forward pass (model uses its own null representation)
model(x, t, cond, drop_mask=m)   # mixed batch: null conditioning for masked samples (training CFG)
```

- **Methods** decide CFG *strategy* — when to drop (drop_mask), how to combine cond/uncond at sampling time
- **Models** decide CFG *representation* — what "unconditional" means internally (null class token, learned null embedding, etc.)
- Any method works with any model. No model-specific logic in methods.

### UNet (2D, MNIST)
- Hidden dims [64, 128, 256], 2 down/up stages (32→16→8)
- Class conditioning: `nn.Embedding(num_classes + 1, time_dim)`, null token = `num_classes`
- Strided conv downsampling, ConvTranspose upsampling

### UNet3D (Lobes)
- Hidden dims [64, 64, 128, 128], 3 down/up stages (50→25→12→6)
- MaxPool3d downsampling, `F.interpolate` (trilinear) upsampling — handles odd spatial dims
- Continuous conditioning: 5 inputs [height, radius, aspect_ratio, angle_deg, ntg] normalized to [0,1] by dataset
- Angle internally converted to sin(2π·angle_norm) and cos(2π·angle_norm) for 180° periodicity
- Learned null embedding vector (`nn.Parameter`) for CFG unconditional
- **Inpainting mode** (`in_channels=3, out_channels=1`): takes `[noisy_x, known_clean_values, binary_mask]` as 3 input channels via channel concatenation. Inpaint context is set via stateful `set_inpaint_context(mask, data)` / `clear_inpaint_context()` — methods remain completely untouched

## Key Implementation Details

- MeanFlow's UNet takes `num_time_embs=2` (for t and t-r); others use 1
- MeanFlow training uses `torch.func.jvp` and `torch.func.functional_call` for Jacobian-vector products; JVP target uses EMA shadow weights as a target network for training stability
- MNIST: images 32x32 (28x28 padded by 2px each side), batch size 128
- Lobes: volumes 50x50x50 (binary int8 facies), batch size 32, facies mapped {0,1}→{-1,1}
- Lobe NTG computed from actual voxel data (fraction of 1s), NOT from parameters.csv. Samples with NTG < 0.05 or > 0.95 filtered out (~89k samples remain)
- Optimizer: AdamW with lr=1e-3, cosine annealing LR schedule, gradient clipping (max_norm=1.0), EMA (decay=0.9999)
- `Diffusion` class supports both DDPM (stochastic) and DDIM (deterministic when `eta=0`) sampling via the `sampler` argument; both use proper posterior variance for arbitrary step counts (strided timesteps). DDIM clips x0 predictions to [-1,1] and recomputes eps for consistency (prevents CFG drift)
- Diffusion uses linear beta schedule (1e-4 to 0.02, 1000 steps)
- Rectified Flow inherits from FlowMatching; 2-Rectified Flow uses coupled pairs generated by ODE integration of the trained FM model. Three pair generation modes:
  - **Forward** (`generate_reflow_pairs`): exact noise → approximate data
  - **Backward** (`generate_reflow_pairs_backward`): approximate noise ← exact data
  - **Bidirectional**: concatenation of forward + backward pairs
- **Classifier-Free Guidance (CFG)**:
  - Methods create `drop_mask` (10% probability) and pass to model via `drop_mask=` kwarg
  - Models apply the mask using their own null representation
  - During sampling: `output = uncond + cfg_scale * (cond - uncond)`, default `cfg_scale=3.0`
  - **MeanFlow supports two CFG modes** (`cfg_mode` argument):
    - `'standard'`: CFG applied at sampling time (2 NFE/step, same as other methods)
    - `'embedded'`: CFG baked into training target per paper Sec 4.2 / Eq. 17-21; replaces `v_t` with `ṽ_t = ω·v_t + κ·u_cond + (1-ω-κ)·u_uncond` in both JVP tangent and target, enabling 1-NFE sampling
- **3D Inpainting** (lobe-specific):
  - Channel concatenation: model takes `[noisy_x, known_values, mask]` (3 input channels), outputs 1 channel
  - Stateful context: `model.set_inpaint_context(mask, data)` stores mask/data as instance attributes; `forward()` concatenates them to input before `init_conv`. Methods are completely unaware of inpainting
  - MeanFlow JVP compatible: stored context tensors are `.detach()`ed constants with zero tangent in `torch.func.jvp`
  - Mask convention: 1 = known (keep), 0 = unknown (generate)
  - Training distribution: 30% no mask (unconditional), 70% with mask (1/6 wells, 1/6 boundaries, 1/6 cross-sections, 1/2 combinations)
  - Mask types (`resflow/utils/masking_lobes.py`):
    - **Wells**: 1-5 per sample, variable depth, 1 voxel wide, 50% vertical / 50% L-shaped horizontal, non-intersecting
    - **Boundaries**: 1-6 faces, 1-3 voxels thick per face (for autoregressive reservoir generation)
    - **Cross-sections**: 1 voxel thick, any angle in x-y plane, up to 30° z-tilt (for seismic conditioning)
    - **Combinations**: union of any 2 or all 3 types
  - Sampling: no re-injection during denoising (matches training); hard replacement of known voxels only at the final step via `apply_inpaint_output(samples, mask, known_data)`
  - Data: `LobeInpaintDataset` wraps `LobeDataset` with on-the-fly mask generation; `get_lobe_inpaint_loaders` returns DataLoaders yielding `(facies, cond, mask)` triples
  - Training: `train_model_inpaint` in `training.py` unpacks 3-element batches, sets inpaint context on unwrapped model before `compute_loss`

## DiT3D / assembly work (Sept 2026) — see `scripts/tier2/DIT_FINDINGS.md`

- Stable DiT recipe: `resflow/models/dit3d.py` (QK-norm attention, optional
  `conv_io='refine'`), `scripts/tier2/train_assembly.py --arch dit --ema-warmup`.
  Sample DiTs with Heun (`solver='heun'`), not the UNet's Euler-50.
- Assembly: prefer 4-stage conditioning (`resflow/assembly/schedulers.py::generate_staged`,
  `order='stage4'`, overlap 12–16) over MultiDiffusion for DiTs; all assemblers accept a
  per-block condition grid, `solver=`, and `trajectory_path=`. Tested and rejected:
  `generate_coupled` (per-block copies drift apart) and `generate_spot` (SpotDiffusion-style
  shifted non-overlapping windows: merges more than MultiDiffusion) — see DIT_FINDINGS §5, §9.
- **Future work (not started): cheaper late stages in 4-stage.** Stage-2/3 blocks are
  50–75% known and stage-4 blocks 75–94% known (overlap 16 / 24), yet every stage runs
  the full 64×64×32 block for all ODE steps. Options: fewer steps for stages 3–4
  (one-line change, ~25% saving), or crop late stages to the unknown region plus a
  context margin (~0.56× total cost; DiT needs positional-embedding sub-grid cropping
  and a quality check with reduced context). Worth a training campaign, in this order:
  (1) a whole-field DiT with 3D RoPE + windowed attention + per-token adaLN, which
  removes tiling altogether (DIT_FINDINGS §9); (2) a per-token noise-level DiT
  (AsyncPatch / Rolling-Diffusion style) so a noise frontier can sweep the reservoir.
