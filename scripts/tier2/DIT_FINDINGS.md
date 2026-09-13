# DiT3D on the lobe assembly problem — findings and recipe (2026-09-08/10)

Working notes from the two-day sweep on Vista (jobs 981959, 983447). Every
number below is reproducible from the run directories listed at the end.
Metrics are ResBench Addendum-E native scores unless stated (geobody W1 /
extent W1, lower is better) — see the caveat in §6 before reading too much
into geobody W1 alone.

## 1. Why the August DiT runs failed

Three independent problems, all fixed:

1. **Attention without QK-normalisation.** `nn.MultiheadAttention` with raw
   q·k logits. Logit growth is slow weight drift, gradient clipping cannot
   stop it, and one overflow poisons every parameter (NaN grads → NaN
   weights). Every 2026-08-02 run NaN'd or diverged, including the LR 1e-4
   run at an effective LR of 4e-5. Fix: per-head RMSNorm on q and k with a
   learnable gain (`resflow/models/dit3d.py::Attention`), fused
   `F.rms_norm`, `F.scaled_dot_product_attention`.
2. **EMA lag.** Inference checkpoints are an EMA at decay 0.9999 (10k-step
   time constant) on 37k-step runs, so epoch-80 checkpoints still held 2.4%
   initial weights (31% at epoch 25). For adaLN-Zero DiTs that outputs
   nothing: `fx_dit_p884` raw val 0.298 vs its EMA checkpoint 0.425. All old
   DiT val sweeps measured staleness. Fix: `--ema-warmup` (decay ramps as
   (1+k)/(10+k)); `--save-raw` keeps non-EMA weights.
3. **Linear patchify/unpatchify at 8×8×4.** Stable, but samples are a
   checkerboard of patch tiles (99.8% of sand in one connected body). Fix:
   patch 4×4×4 (or 4×4×2); the residual conv `RefineHead`
   (`--dit-conv-io refine`) helps val but not geology (§4).

Plus a sampler mismatch: the paper's Euler-50 + CFG 3 is the UNet's tuned
operating point; the DiT's velocity field is more curved and is
under-integrated by it.

## 2. Training recipe that works

```
scripts/tier2/train_assembly.py --data-mode crops192 --arch dit --masked-loss \
  --amp bf16 --micro-batch 48 --num-workers 8 --beta2 0.95 --ema-warmup --save-raw \
  --dit-patch 4 4 4 --lr 5e-4 --epochs 80 --total-epochs 80          # 33M
  [--dit-hidden 512 --dit-depth 16 --dit-heads 8]                    # 77M
  [--dit-patch 4 4 2]                                                # 8192 tokens
```
Global batch 384 via `launch_assembly_ddp.sh`; 4-epoch warmup → cosine;
AdamW wd 1e-2; grad clip 1.0; non-finite steps are skipped (never fired).
Zero NaNs in 20+ runs. LR 1e-3 is also stable and slightly better.

## 3. Scoreboard (native 64³ blocks, lobe, CFG 3)

| model | params | val | Euler-50 | Heun-25 (same cost) | Heun-50 |
|---|---|---|---|---|---|
| UNet bf_unet (EMA) | 5.4M | 0.165 | 0.109 / 0.056 | 0.137 / 0.075 | 0.186 / 0.097 |
| DiT p444 seed 1 / 2 | 33M | 0.164 / 0.174 | 0.304 / 0.466 | 0.207 / 0.365 | 0.162 / 0.323 |
| DiT p444 LR 1e-3 seed 1 / 2 / 3 | 33M | 0.156 / 0.158 / 0.157 | 0.247 / 0.234 / 0.249 | | |
| DiT p444 160 ep | 33M | 0.154 | 0.263 / 0.118 | | |
| DiT p444 + refine | 34M | 0.145 | 0.255 / 0.115 | 0.150 / 0.066 | 0.094 / 0.036 |
| DiT p442 (4×4×2) seed 1 / 2 | 33M | 0.149 / 0.147 | 0.159 / 0.158 | 0.073 / 0.025 | 0.061 / 0.021 |
| DiT 77M seed 1 / 2 | 78M | 0.145 / 0.148 | 0.192 / 0.194 | 0.088 / 0.097 | 0.063 / 0.071 |
| DiT 77M + refine | 79M | 0.135 | 0.196 / 0.085 | 0.110 / 0.044 | 0.069 / 0.023 |
| DiT 77M + refine LR 1e-3 | 79M | 0.129 | 0.215 / 0.093 | 0.134 / 0.055 | 0.075 / 0.023 |
| DiT 77M LR 1e-3 | 78M | 0.137 | 0.182 / 0.076 | | |
| DiT 33M + refine LR 1e-3 | 34M | 0.138 | 0.223 / 0.096 | | |
| DiT 150M + refine (hidden 640, depth 20) | 150M | 0.127 | 0.187 / 0.077 | 0.114 / 0.043 | 0.072 / 0.024 |
| DiT 77M + traj (`--traj-prob 0.5`, 4-ch) | 78M | 0.142* | 0.162 / 0.067 | 0.079 / 0.027 | 0.068 / 0.024 |
| DiT 33M + traj LR 1e-3 seed 1 / 2 | 33M | 0.161* / 0.159* | 0.230 / 0.288 | 0.131 / 0.056 | 0.107 / 0.042 |

(*) val of the traj models includes trajectory-conditioned samples, not comparable.
Heun-25 costs the same 100 model calls as the paper's Euler-50. The UNet
gets worse with more accurate integration; DiTs get better. CFG 3 is right
for the DiT too (1.5/2.0 worse, 4.5 marginal). Seed spread at 33M is large
(see §6); the 77M reproduces across seeds.

## 4. What helped and what did not

Helped (geology): Heun / more steps; patch 4×4×4 → 4×4×2; width (77M);
sequential fusion instead of MultiDiffusion (§5); trajectory-conditioned
training (`--traj-prob 0.5`: half the samples see a neighbour at a random
noise level) as a regulariser — the 77M traj model is the best 77M at
Heun-25 (0.079 vs 0.088) although it was trained for the coupled sampler.
Width beyond 77M did not: the 150M model has the best val (0.127) but
the 77M's geology (0.114 vs 0.088 at Heun-25). Helped (val only): refine
head, LR 1e-3, 160 epochs — below val ≈ 0.145 the loss no longer predicts
texture. Did not help: the upsample-conv head (`--dit-conv-io up`, cannot
draw sharp facies), raster-only masks, CFG ≠ 3.

## 5. Assembly (10×10 blocks, overlap 24), hard case = paper Figure-4 grid

All samplers now accept a per-block condition grid and `solver='heun'`
(`resflow/assembly/{big_reservoir_multi,outpaint,schedulers}.py`;
`scripts/tier2/generate_schedulers.py --solver heun`).

- **MultiDiffusion** (velocity averaging): with the DiT + Heun it merges far
  fewer lobes than the paper model (large-lobe window: 38 vs 29 sand regions
  per slice, largest region 20% vs 28%, chords > 9.6 km 0.02% vs 0.92%) but
  leaves thin broken shale rims (sliver fraction 0.22 vs native 0.14).
- **Outpaint** (raster wavefronts, left/top context): intact rims (0.126)
  but over-continues lobes along the azimuth (0.55% long chords).
- **4-stage** (4-colour staging, 4 rounds, both-side context): best overall
  — 37 regions/slice, 19% largest, 0.04% long chords, rims 0.17.
- **Hybrid** (MultiDiffusion warm-up for the first 15–30%, then 4-stage) and
  the fully parallel **coupled** sampler (needs `--traj-prob 0.5` models,
  `fx_dit_p444_big_traj`, `fx_dit_p444_traj`) are implemented; results in
  `resbench_eval/fig4_variants/` and the study lanes in `specialist_runs/study_lane*.log`.
- The DiT holds the requested NTG 0.70 in every fusion (UNet 0.66).
- **Overlap** (benchmark: uniform condition, Heun-25, 10 assemblies, ResBench
  tiled geobody W1 B_vs_C; UNet EMA reference 0.114 multi / 0.087 outpaint):

  | fusion | ov 24 | ov 16 | ov 12 | ov 8 |
  |---|---|---|---|---|
  | 77M MultiDiffusion | 0.281 | 0.375 | | |
  | 77M 4-stage | 0.316 | 0.209 | 0.184 | 0.185 |
  | 77M 4-stage, Heun-50 | | | 0.154 | |
  | 77M traj 4-stage | 0.300 | | 0.174 | |
  | 150M refine 4-stage | | | 0.415 | |
  | p442 MultiDiffusion | 0.220 | 0.321 | | |
  | p442 4-stage | 0.248 | 0.157 | **0.139** | 0.141 |
  | p442 4-stage, Heun-50 | | | **0.118** | |
  | p442 seed-2 model, 4-stage (Heun-25 / Heun-50) | | | 0.174 / 0.139 | |

  The refine-head models tile badly under every fusion (150M refine:
  MultiDiffusion 0.579, 4-stage ov12 0.415) — never use `--dit-conv-io
  refine` for assembly. Training-seed spread of the tiled score is real (p442 seed 1 vs 2: 0.139
  vs 0.174 with identical native scores, 0.159 vs 0.158 at Euler-50).
  4-stage improves as the overlap shrinks and saturates at 8–12;
  MultiDiffusion does the opposite (less averaging to hide the seams).
  Overlap 32 breaks 4-stage (same-stage blocks touch). Smaller overlap is
  also cheaper (fewer blocks per area). Seed-2 77M 4-stage ov24: 0.305;
  traj-77M 0.300; traj-33M 0.248.
- **Seed robustness of the recipe (hard case, p442, Heun-25, large-lobe
  window; seeds 11 / 12 / 13):** 4-stage ov12 largest-region share 0.182 /
  0.181 / 0.164, shale slivers 0.137 / 0.121 / 0.126, long chords 0.0017 /
  0.0005 / 0.0003, 32–36 regions per slice; MultiDiffusion ov24 on the same
  seeds 0.237 / 0.219 / 0.236, 0.175 / 0.162 / 0.185, 0.0002 / 0.0054 /
  0.0006 (paper model 0.276, 0.125, 0.0092). The fusion effect is 3–5× the
  seed spread; 4-stage rims are at the paper model's sliver level with a
  quarter of its merging. 77M over the same three seeds: 4-stage ov12
  0.181 / 0.209 / 0.204 with slivers 0.120 / 0.124 / 0.144, MultiDiffusion
  0.258 / 0.221 / 0.212 with slivers 0.193 / 0.175 / 0.190. The
  trajectory-trained 77M merges slightly more under 4-stage (0.219, long
  chords 0.0040) — its native gain does not carry into the assembly.
  Model-to-model spread is larger than noise-seed spread: the second p442
  training seed (`fx_dit_p442_s2`, noise seed 11) gives 4-stage ov12 0.215 /
  slivers 0.123 / long chords 0.0078 vs its MultiDiffusion 0.251 / 0.190 /
  0.0001 — still less merged and cleaner rims, but the long-chord fraction
  is close to the paper model's: 4-stage can over-continue lobes along y
  the way outpainting does when both neighbours are already finished.
  Check long chords, not only the largest-region share, when tuning.
  Figure `figures_dit_vs_unet/fig13_tau2d_final_recipe.pdf`.
- **Shared global noise ablation** (p442 4-stage ov12 Heun-25, hard case,
  seeds 11 / 12): without the shared noise field, largest share 0.194 /
  0.177 vs 0.182 / 0.181 with it, slivers 0.140 / 0.124 vs 0.137 / 0.121 —
  no measurable effect; the neighbour conditioning carries the coherence.
  On the tiled benchmark it helps modestly: 0.162 without vs 0.139 with
  (smaller than the 0.035 training-seed spread). Keep it (free), but do
  not build a claim on it.
- Hybrid (30% MultiDiffusion warm-up then 4-stage): 0.578, worse than either.
- **Coupled** (fully parallel trajectory conditioning, Voronoi assembly of
  independent block copies): 0.82 (77M traj) / 0.69 (33M traj) on the
  benchmark and visibly speckled on the hard case — rejected in this form.
  The trained conditioning is sound (see the traj rows in §3); the assembly
  of drifting per-block copies is what fails.
- **Long-range connectivity** (`specialist_runs/longrange_tau2d.py`,
  plan-view τ2D(h) on the whole 424² hard-case reservoir, so cross-block
  continuity counts, unlike the tiled benchmark). Along y (lobe long axis)
  at lags 64 / 128: paper 0.125 / 0.022, DiT MultiDiffusion 0.092 / 0.019,
  4-stage 0.071 / 0.008, 4-stage ov16 0.038 / 0.003. Along x the DiT
  MultiDiffusion keeps a 0.02–0.04 tail out to lag 256 (rim slivers
  bridging lobes) that 4-stage does not (0.014 → 0.000).
  `figures_dit_vs_unet/fig10_longrange_connectivity_hardcase.pdf`,
  `fig10b_planview_connectivity_hardcase.pdf`, `fig11_tau2d_spot.pdf`.
- Trajectories: all assemblers take `trajectory_path=` and record the
  evolving field (per step for MultiDiffusion/coupled, per group for
  4-stage/outpaint) as fp16 npz.

Recommended: p442 or 77M DiT, Heun-25, 4-stage fusion at overlap 12 (8–16
all fine). Best DiT tiled score 0.139 (p442, ov12, Heun-25) vs UNet
0.087–0.114; at Heun-50 (2× the calls) p442 4-stage ov12 reaches 0.118 /
extent 0.047, inside the UNet's range — the residual tiled gap is mostly
integration error, not the fusion. But Heun-50 does NOT reduce merging on
the hard case (p442 4-stage ov12, seeds 11/12/13: largest share 0.192 /
0.210 / 0.172 vs 0.182 / 0.181 / 0.164 at Heun-25, slivers 0.14 vs 0.13):
the tiled benchmark re-cuts into 64³ tiles and rewards within-block
texture; merging is decided by the fusion. Use Heun-25 for assembly.

## 6. Measurement caveats

ResBench geobody W1 uses 3D 26-connectivity: at NTG ≈ 0.5 sand percolates,
so it mostly measures slivers and percolation, not lobes. Two 33M seeds with
chord statistics 0.3 cells apart differ 2× in geobody W1. Use
segmentation-free chord statistics (`specialist_runs/chord_native_vs_tiled.py`)
and per-slice 2D regions for merging, and look at the figures
(`resbench_eval/figures_dit_vs_unet/fig1…fig7`). By chords, the 77M and p442
DiTs at converged sampling match the engine as well as the UNet does.

## 7. Where things are

- Runs: `/scratch/08405/ilgar/specialist_runs/<tag>/` (purge-prone); key
  checkpoints backed up to `$WORK/dit_runs_backup/<tag>/`.
- Scores: `$WORK/resbench_eval/assembly_<tag>[_heunN|_nfeN|_cfgX_nfeN]/scored*/`.
- Hard-case volumes: `$WORK/resbench_eval/fig4_variants/fig4lobe_<model>_<fusion>_<sampler>_seed11.npz`.
- Tools (in `specialist_runs/`): `nfe_sweep.sh`, `solver_sweep.sh`,
  `cfg_nfe_sweep.sh`, `asm_nfe.sh`, `fig4_variants.py`, `plot_topviews.py`,
  `plot_zooms.py`, `plot_heun_hardcase.py`, `chord_native_vs_tiled.py`,
  `parallel_val.sh`, drivers `j983447_group5.sh`.
- Drivers for the last day: `spot_study.sh`, `study3.sh`, `study4.sh`,
  `coupled_eval.sh`, `handoff_983447.sh`, `j983447_group5.sh`.
- Open: engine reference at hard-grid conditions (ResMill) to judge drape
  thickness; Heun-based `score_assembly_model.sh` as the default for DiTs;
  a coupled sampler that keeps one shared field instead of per-block copies.

## 8. Scaling to 30×30 blocks, and cheaper late stages (parked)

Measured on one unshared GH200 with the 77M DiT, Heun-25, CFG 3, overlap 12
(`specialist_runs/scaling_demo.py`, volumes + full trajectories in
`specialist_runs/scaling/`, whole-field stats via `field_stats.py`):

| grid | field | 4-stage | MultiDiffusion |
|---|---|---|---|
| 10×10 (hard case) | 532² / 424² × 32 | 102 s | 109 s |
| 30×30 (uniform, NTG 0.51) | 1572² × 32 (79 M cells) | 851 s, 13 GB peak | 1029 s, 3.5 GB peak |

4-stage is linear in the number of blocks (9× blocks → 8.3× time) and 17%
faster than MultiDiffusion at identical model-call counts, because its
stage batches hold 225 blocks (vs batches of 24 plus a scatter/gather per
ODE step); it also needs only 3 cross-block exchanges instead of 25, which
is what matters once blocks are spread over GPUs. Whole-field quality at
30×30 (per-slice 2D regions per 10⁴ cells / largest share / long chords /
shale slivers): 4-stage 14.1 / 0.002 / 0.0000 / 0.048, MultiDiffusion
15.1 / 0.004 / 0.0003 / 0.072 — at NTG 0.5 lobes are separated either way;
the fusion difference is again in the rims. Figure
`figures_dit_vs_unet/fig15_scaling_30x30_topviews.pdf`.

**Cheaper late stages (not started).** 4-stage and MultiDiffusion cost the
same in model evaluations (N blocks × steps), and
4-stage's unexploited saving: the
model always runs on the full 64×64×32 block, but the unknown region is
100% / 25% / 25% / 6% of the block over stages 1–4 at overlap 24 (100 / 50 /
50 / 25% at overlap 16). (a) Fewer ODE steps for stages 3–4: one-line change,
~25% off. (b) Crop stages 2–4 to the unknown strip/hole plus a context margin
(32×64 and 32×32 windows): per-block cost 1 + 0.5 + 0.5 + 0.25 vs 4, i.e.
~0.56× MultiDiffusion; the UNet can do this now, the DiT needs its learned
positional embedding sub-grid cropped and a check that quality holds with
less context than in training. Batch-size effects are secondary: the DiT
saturates a GH200 by batch 8–16 (73 → 162 samples/s from batch 1 to 16), so
stage batches of N/4 lose nothing for grids of 8×8 and above.

## 9. Shifted non-overlapping windows (SpotDiffusion-style) — tested, rejected

The image community's averaging-free alternative to MultiDiffusion
(SpotDiffusion, 2024): at every ODE step partition the field into disjoint
block-size windows, displace the partition by a fresh 2D offset, run every
window independently and write back without blending; seams of one step
lie inside the next step's windows. Implemented as
`resflow/assembly/schedulers.py::generate_spot` (`--sampler spot`,
fig4 fusion `spot` / `spotrand`; Halton or random offsets; nearest-window
ownership at the reservoir edge instead of wrap-around). It is the
cheapest scheduler (137 s vs 190–220 s for 4-stage / MultiDiffusion on the
hard case at Heun-25: one evaluation per voxel per step) and fully parallel
with no exchange inside a step.

Hard case, 77M DiT, large-lobe window (regions/slice ↑ better, largest
share ↓, chords > 9.6 km ↓, shale slivers ↓; plan-view τ2D_x at lag 64 ↓):

| fusion | regions | largest | long chords | slivers | τ2D_x(64) |
|---|---|---|---|---|---|
| paper model, MultiDiffusion, Euler-50 | 29.1 | 0.276 | 0.0092 | 0.125 | 0.053 |
| DiT MultiDiffusion, Heun-25 | 39.3 | 0.258 | 0.0006 | 0.193 | 0.084 |
| DiT 4-stage, Heun-25 | 36.6 | 0.190 | 0.0004 | 0.149 | 0.043 |
| DiT 4-stage ov16, Heun-25 | 32.6 | 0.172 | 0.0001 | 0.155 | 0.029 |
| DiT Spot (Halton), Heun-25 | 38.8 | 0.326 | 0.0029 | 0.256 | 0.118 |
| DiT Spot (random), Heun-25 | 40.6 | 0.283 | 0.0019 | 0.249 | 0.109 |
| DiT Spot (Halton), Heun-50 | 43.0 | 0.232 | 0.0003 | 0.215 | 0.083 |

Spot merges more than MultiDiffusion at equal steps and leaves the most
broken rims; doubling the steps (more distinct partitions) only brings it
back to MultiDiffusion. So removing the averaging is not what fixes
merging — conditioning each block on *finished* neighbours is, which is
what 4-stage and outpainting do and no parallel image-tiling scheme does.
Figure: `figures_dit_vs_unet/fig12_spot_vs_multi_vs_stage4_topviews_3z.pdf`.
Benchmark tiled geobody W1 (uniform condition, Heun-25, ov24 layout): Spot 0.438
vs MultiDiffusion 0.281, 4-stage 0.316 (0.184 at ov12) — consistent with the
hard case (`resbench_eval/sched_big_spot_heun25/scored/`).

Related image-community ideas not tried (both need retraining):

1. **Whole-field DiT with 3D RoPE + windowed attention (strongest
   candidate).** Our DiT has a learned absolute position table for the
   16×16×8 token grid, which is the only reason it must be tiled. With
   rotary embeddings attention depends on relative offsets only, so a model
   trained on 64×64×32 crops runs on any field; windowed (neighbourhood or
   shifted-window) attention of one block width keeps every offset inside
   the trained range (no interpolation — cell size must stay physical),
   makes cost linear in field size (~Spot's cost, no stages, no exchanges),
   and gives exact halo tiling if memory ever forces cutting. Needs
   per-token adaLN from a condition map for varying conditions. ~1 day of
   code, 5 h (33M) / 9 h (77M) of training. If it works it retires the
   fusion-rule question.
2. Per-token noise levels (AsyncPatch / Rolling Diffusion): a noise frontier
   sweeping the reservoir, 4-stage and outpainting as two settings of one
   continuous scheme.

## 10. Whole-field RoPE DiT (2026-09-11, job 988877, 32 nodes)

Goal: a DiT that generates the entire reservoir in one pass — no tiling,
no fusion rule — and beats tiled 4-stage on the hard case and the
benchmark. Design (`resflow/models/dit3d.py`, `pos_embed='rope'`):

- **3D axial RoPE** instead of the learned absolute table: head_dim split
  22/22/20 over x/y/z, each with its own frequency ladder θ^(-2i/d); q and
  k are rotated after QK-norm, so attention depends on relative offsets
  only and the weights run on any token grid.
- **Windowed attention** of 16×16×8 tokens (= one 64×64×32 block at patch
  4³); odd blocks shift the window grid by half a window in x and y.
  Windows are cut at the grid edge (smaller border windows) instead of
  wrapped or padded: no masks, flash kernels everywhere, any field size.
  No relative offset larger than the window ever occurs, so there is no
  extrapolation at inference; cost is linear in field size.
- **Per-token adaLN**: the condition may be (B, C), (B, N, C) tokens or a
  voxel map; the hard case's 10×10 block grid becomes one condition map
  (nearest block centre per token).
- Training on **128×128×32 crops** (32×32×8 tokens = 4 windows, so the
  shifted blocks see full interior windows), wells-only masks
  (`--context-share 0`), otherwise the §2 recipe (batch 384, 80 ep,
  β2 0.95, EMA warm-up, bf16). 4× the tokens of the 64³ recipe per epoch.
- Sampler `resflow/assembly/wholefield.py::generate_wholefield`
  (`generate_schedulers.py --sampler wholefield`, fig4 fusion
  `wholefield`): Heun, CFG batched as (cond, uncond) in one call, bf16.
  Untrained 33M: a 532²×32 field (141k tokens) is 0.26 s per forward,
  2.8 GB; 640² is 0.21 s, 4.1 GB → ~45 s per Heun-25 field vs 102 s for
  4-stage. Old checkpoints load unchanged (position/window buffers are
  self-describing).

Sweep (all RoPE, window 16×16×8, crops 128², LR 5e-4 unless stated;
run tags in `specialist_runs/`, driver `j988877_rope.sh`, eval
`rope_eval.sh` armed per arm):

| arm | nodes | model | RoPE θ | notes |
|---|---|---|---|---|
| rope_big_t100 | 8 | 77M (512/16/8) | 100 | micro 12 |
| rope_p442_t100 | 8 | 33M patch 4×4×2, window 16×16×16 | 100 | 16k tokens per crop |
| rope_t100 | 4 | 33M | 100 | main 33M |
| rope_t10k | 4 | 33M | 10000 | RoPE base ablation |
| rope_t100_lr1e3 | 4 | 33M | 100 | LR 1e-3 |
| rope_t100_w8 | 4 | 33M, window 8×8×8 (32-cell) | 100 | locality / cost |

~190 samples/s per arm → ~20 h per 80 epochs (done ~2026-09-12 09:00).
Evaluation per arm (automatic): native 64³ Heun-25 ensemble (A_vs_C),
whole-field 10 assemblies on the ov12 layout (532², B_vs_C tiled score),
hard case seeds 11/12/13 vs p442 4-stage ov12 (merge metrics, τ2D, top
views fig16/fig17), 30×30 whole-field timing (`scaling/`). Targets to
beat: p442 4-stage ov12 Heun-25 tiled 0.139 (0.118 at Heun-50); hard case
largest share 0.16–0.18, slivers 0.12–0.14, long chords ≤ 0.002; UNet
native 0.109 / 0.056, p442 native 0.073 / 0.025.

### 10.1 Round-1 results (2026-09-12)

First two arms evaluated (`rope_t100_w8` = 33M, 8×8×8-token windows, the
deliberately cheapest arm; `rope_big_t100` = 77M, 16×16×8 windows).

**The architectural claim holds.** ResBench separates the model's own
quality (A_vs_C) from what assembly costs on top (B_vs_A). Whole-field
generation halves the damage of the best fusion rule (geobody W1,
Heun-25 CFG 3, ov12 layout):

| assembler | assembly damage B_vs_A |
|---|---|
| whole-field RoPE, one pass | **0.053** |
| tiled p442, 4-stage ov12 | 0.092 |
| tiled 77M, 4-stage ov12 | 0.106 |
| tiled 77M, MultiDiffusion | 0.203 |

and it is 3× faster: 30×30 blocks (1572²×32, 900 blocks) in 286 s / 24 GB
vs 851 s (4-stage) and 1029 s (MultiDiffusion) on one GH200, with equal
whole-field statistics (regions per 10⁴ cells 14.6 vs 14.1, slivers 0.055
vs 0.048). A 532² hard case is 13.6 s on an idle GH200 (42 s for the 77M).

**Hard case, 77M whole-field at CFG 3 vs tiled p442 4-stage ov12**
(3 seeds each): long chords 0.0005 / 0.0000 / 0.0003 vs 0.0017 / 0.0005 /
0.0003, plan-view τ2D lower at every lag (x@128 0.019 / 0.007 vs 0.037;
x@256 0.014 / 0.000 vs 0.026) — the seam-sliver bridges that survived
4-stage are gone. Largest-region share 0.197 mean vs 0.176, shale slivers
0.157 vs 0.128: thinner rims are the remaining whole-field weakness.
NTG 0.702–0.704 for a requested 0.70.

**CFG is a metric trap — do not chase the benchmark number.** Whole-field
benchmark geobody W1 (33M w8 arm) falls monotonically with guidance:

| sampler | benchmark geobody | hard-case largest share | long chords | NTG |
|---|---|---|---|---|
| Heun-25 CFG 3 | 0.231 | 0.215 | 0.0032 | 0.703 |
| Heun-25 CFG 4.5 | 0.109 | (not run) | | |
| Heun-25 CFG 6 | 0.053 | 0.245 | 0.0010–0.0089 | 0.712 |
| Heun-25 CFG 7.5 | 0.047 | | | |
| Heun-50 CFG 4.5 | 0.048 | | | |
| Euler-50 CFG 3 (paper cost) | 0.311 | | | |
| Heun-50 CFG 3 | 0.121 | (testing) | | |

At CFG 6 the score is 4× better and the geology is worse: drapes thin and
break, neighbouring lobes join, one seed reaches the paper model's own
long-chord fraction (0.0089 vs 0.0092), and NTG drifts +1.2 pp. Geobody
W1 rewards removing small spurious bodies, which is what sharpening does.
Report CFG 3 numbers; use the CFG sweep only as evidence that the metric
is fragile (§6). Solver findings that ARE real: Heun-25 beats Euler-50 at
equal cost (0.231 vs 0.311), Heun-15 and Euler-25 are far worse.

### 10.2 HEADLINE (2026-09-12): whole-field 77M at Heun-50 is the best result so far

Operating point: `--sampler wholefield`, Heun **50**, CFG **3**, no tiling.
More integration steps (not sharper guidance) is what a whole field needs:
at 25 steps the rim slivers of a 532² field are 0.176, at 50 steps 0.113.

**Benchmark (uniform condition, 10 assemblies, ResBench tiled scoring):**

| model / assembler | geobody W1 | extent W1 | NTG err |
|---|---|---|---|
| engine reference band | 0.019 | 0.010 | 0.0001 |
| **whole-field 77M RoPE, Heun-50** | **0.089** | 0.033 | 0.007 |
| — its native 64³ blocks (best native ever) | 0.056 | 0.018 | 0.003 |
| tiled p442 4-stage ov12, Heun-50 | 0.118 | 0.047 | |
| tiled p442 4-stage ov12, Heun-25 | 0.139 | 0.057 | |
| tiled 77M 4-stage ov12, Heun-25 | 0.184 | 0.075 | |
| UNet EMA, outpaint | 0.087 | 0.038 | |
| UNet EMA, MultiDiffusion | 0.114 | 0.056 | |

**Hard case (paper Fig-4 grid, 3 seeds), whole-field 77M Heun-50 vs the
best tiled DiT and the paper model** — whole-field wins on every measure:

| | whole-field H50 | tiled p442 4-stage | paper model |
|---|---|---|---|
| largest region share | 0.174 / 0.146 / 0.193 | 0.182 / 0.181 / 0.164 | 0.276 |
| shale slivers (rim continuity) | 0.099 / 0.104 / 0.136 | 0.137 / 0.121 / 0.126 | 0.125 |
| long chords (> 9.6 km) | 0.0008 / 0.0004 / 0.0004 | 0.0017 / 0.0005 / 0.0003 | 0.0092 |
| τ2D_x @128 cells | 0.011 | 0.037 | 0.003 |
| NTG (requested 0.70) | 0.701–0.703 | 0.701 | 0.690 |

**Cost:** 84 s per 532² field (Heun-50) vs 102 s for tiled 4-stage
(Heun-25); 900 blocks (1572²) in 409 s / 31 GB vs 851 s (4-stage) and
1029 s (MultiDiffusion).

**Size stability (the architectural claim, measured).** Whole-field
statistics are flat from a 2×2-block field to a 10×10 one — 77M slivers
0.064 / 0.055 / 0.054 at 2² / 4² / 6² blocks, 33M-w8 0.068 / 0.061 /
0.057 / 0.058 at 2² / 4² / 6² / 10² — so composition across attention
windows does not degrade with field size, and the model is indifferent to
how large the reservoir is. (`fieldsize_probe.sh`.)

**Caveat on B_vs_A.** That column is assembly-vs-own-native, so a model
with excellent native blocks (77M: 0.056) shows a *larger* B_vs_A than a
weak one (33M-w8: 0.182 native, 0.053 B_vs_A) for the same field quality.
Compare assemblers with B_vs_C, or with B_vs_A only at matched native.

### 10.3 Training crops MUST span several attention windows (the key ablation)

`rope_big_crop64`: the round-1 77M recipe in every respect except the
training crop, which is 64×64×32 — the paper's resolution, and exactly ONE
16×16×8-token attention window. Trained 80 epochs on 8 nodes (4.7 h).

| | native 64³ blocks | whole 532² field | composition damage |
|---|---|---|---|
| trained on 64-cell crops (1 window) | 0.065 | **0.677** | 0.714 |
| trained on 128-cell crops (4 windows) | 0.056 | 0.089 | 0.086 |

The one-window model makes *excellent* individual blocks and cannot compose
a field at all: an order of magnitude worse, at identical architecture,
sampler and parameter count. Its validation loss gives no warning
(0.137 vs 0.133), and it also overfits — best val at epoch 50, decayed to
0.157 by epoch 80, while the 128-crop model is flat from 60 to 80. Larger
crops are therefore both a *requirement* for whole-field composition and a
regulariser (4× the tokens and far more distinct context per sample).

The failure mode is not what the seam hypothesis predicts. There are no
visible discontinuities at the window period: the whole field is uniformly
shredded, with ragged fragmented bodies everywhere
(`resbench_eval/figures/fig19_crop_ablation_window_artefacts.pdf`, red
lines mark the 64-cell window period). A model that never saw a window
boundary never learns body shape beyond its own window, at any scale.

Attribution of the validation gain over the tiled 77M (0.145): 0.008 of it
is RoPE + windowed attention + the wells-only mask set (64-crop control
0.137), and 0.005 is the larger crop (0.133). Neither dominates.

### 10.4 Uniform Heun-50 comparison of every round-1 arm

Same sampler (Heun-50, CFG 3), same layout, native reference regenerated at
Heun-50 for each arm, so these are like-for-like (`h50_sweep_all.sh`):

| arm | geobody | extent | connectivity | NTG err |
|---|---|---|---|---|
| 33M rope θ=100 | **0.086** | 0.035 | 0.032 | 0.0050 |
| 77M rope θ=100 | 0.089 | **0.033** | 0.033 | 0.0073 |
| 33M rope θ=10⁴ | 0.096 | 0.037 | 0.028 | 0.0053 |
| 33M rope LR 1e-3 | 0.098 | 0.038 | 0.031 | 0.0061 |
| 33M rope p442 | 0.117 | 0.046 | **0.027** | 0.0072 |
| 33M rope window 8³ | 0.121 | 0.053 | 0.028 | 0.0060 |
| 64-crop control | 0.677 | 0.305 | 0.088 | 0.0067 |
| UNet EMA outpaint | 0.087 | 0.040 | 0.018 | 0.0086 |
| UNet EMA MultiDiffusion | 0.114 | 0.043 | 0.015 | 0.0082 |
| best tiled DiT (p442 4-stage ov12 H50) | 0.118 | 0.047 | 0.039 | 0.0003 |

Model size barely matters once the recipe is right (33M 0.086 vs 77M 0.089,
inside the seed spread); every whole-field arm beats every tiled DiT; all
of them still trail the UNet on connectivity (0.027–0.033 vs 0.018), which
is the thin-drape/patch-boundary issue §10.5 tests.

### 10.5 New best, and a failed control (2026-09-13)

**Best model in the project: `rope_big_p442`** — 77M, 3D RoPE, patch 4×4×2,
16×16×16-token windows, 128² crops, 50 epochs — sampled whole-field at
Heun-50 / CFG 3:

| benchmark (B_vs_C) | geobody | extent | connectivity | NTG err |
|---|---|---|---|---|
| engine band | 0.019 | 0.010 | 0.004 | 0.0001 |
| **77M p442 whole-field** | **0.061** | **0.028** | 0.034 | 0.0066 |
| 33M whole-field (`rope_t100`) | 0.086 | 0.035 | 0.032 | 0.0050 |
| 77M whole-field (`rope_big_t100`) | 0.089 | 0.033 | 0.033 | 0.0073 |
| UNet EMA outpaint | 0.087 | 0.040 | **0.018** | 0.0086 |
| best tiled DiT (p442 4-stage ov12 H50) | 0.118 | 0.047 | 0.039 | 0.0003 |

Hard case (3 seeds, Heun-50): 77M p442 largest share 0.174, slivers
**0.106** (best of any model, paper model 0.125), long chords ≤ 0.0016,
NTG 0.698–0.699. 33M: 0.169 / 0.111 / ≤0.0006 / 0.698–0.700 at 64 s per
field vs 151 s — the 33M is the value choice, the 77M p442 the best score.

**~~Whole-field beats generating the same blocks in isolation.~~ RETRACTED
2026-09-13.** The frozen geobody W1 said 0.061 for the assembled field vs
0.112 for the same model's isolated 64³ blocks. Weighted by body VOLUME
instead of body COUNT the two are the same (0.114 vs 0.122 for the 33M at
Heun-100). 66% of bodies are < 27 voxels, so the frozen metric was
comparing speckle populations, which differ between a tile cut from a field
and an independently generated block. No geological difference is
demonstrated. See `ResBench/analysis/assembly_stats_ext.py`.

**Scaling.** 45×45 blocks = 2352²×32 = 177 M cells in 596 s / 69 GB on one
GH200, statistics identical to 30×30 (regions/10⁴ 14.19 vs 14.20, slivers
0.054 both). Size invariance now measured over a 400× range in area.

**FAILED CONTROL — do not cite the fixed-weight assembler table.** Running
the whole-field weights through the tiled samplers (`sampler_at_fixed_weights.sh`)
was meant to isolate the assembler with the model held fixed. It cannot:
these models trained with `--context-share 0`, so a context slab is out of
distribution, and the outcome flips with capacity — 77M 4-stage 0.073
(better than its own whole-field 0.089!), 33M 4-stage 0.544 (7× worse;
17.1 regions/10⁴ and 0.085 slivers vs 13.8 / 0.049 whole-field).
MultiDiffusion 0.402 and Spot 0.496 for the 77M. Conclusions that DO hold:
(a) the tiling-vs-whole-field choice is made at TRAINING time, a model
trained one way does not transfer to the other; (b) whole-field is robust
across every model trained here (0.061–0.089) while tiling the same models
ranges 0.073–0.544. The headline comparison must stay best-vs-best:
0.061 whole-field vs 0.118 tiled.

**Clue for connectivity.** The 33M's tiled run has the best connectivity
measured anywhere, 0.012 (UNet 0.018, whole-field 0.032), while its bodies
are wrong. Staged conditioning COPIES known context voxels verbatim
(`out = out*(1-mask) + ctx*mask`) instead of re-decoding them, so drapes
inside an overlap are exact by construction. The whole-field decoder has no
such anchor — every voxel is decoded, and a one-cell error merges bodies.

### 10.6 Connectivity: the conv-overlap fix FAILED, and why (2026-09-13)

`rope_big_ovl` = the round-1 77M with `--dit-conv-io refine`: a 3×3×3 conv
at full resolution before the strided patch projection (each token's input
mixes a one-cell halo) and a residual 3-layer 3×3×3 stack after the linear
unpatchify (sees both sides of every patch boundary). Intent: close the
one-cell holes where a drape crosses a patch boundary and two independent
per-token linear decoders disagree.

Result — best validation of the whole campaign (0.128) and worse geology:

| Heun-50, whole field | plain 77M | + conv overlap |
|---|---|---|
| benchmark geobody | 0.089 | 0.191 |
| benchmark extent | 0.033 | 0.084 |
| benchmark connectivity | 0.033 | **0.026** |
| hard case largest share | 0.174 | 0.198 |
| hard case slivers | 0.106 | 0.141 |

**Mechanism, measured, not inferred.** Per 532² field the overlap model has
90% more separate shale bodies (757 vs 398) and 93% more isolated single
shale cells (455 vs 236), plus 19% more sand bodies ≤ 8 voxels (396 vs
327). It did not repair drapes: it added cell-scale speckle. Specks of
shale inside sand break sand-to-sand paths (connectivity improves) and
fragment the sand (geobody collapses). One cause, two opposite-looking
effects. The best model, `rope_big_p442`, has the LEAST speckle of all
(265 tiny bodies, 163 shale singletons) and the best geobody.

**Why a conv head speckles.** It is applied to the velocity at all 50 ODE
steps and trained on MSE. The cheapest MSE reduction is small
high-frequency corrections everywhere (hence the record validation loss);
at the cell scale those act as dither on cells near zero, and binarising at
zero turns dither into speckle. Repairing one specific drape cell buys
almost no squared error, so the head never learns to.

**Methodological consequence — BOTH benchmark metrics have an exploit, and
they pull opposite ways.** Sharpening (CFG > 3) deletes small bodies:
geobody improves 4× while real lobes merge (§10.2). Speckling (this head)
adds small bodies: connectivity improves while real bodies fragment.
Always report a speckle count (isolated single cells per phase, bodies
≤ 8 voxels) and a plan view next to these scores.

## 11. Final results (2026-09-13, job 988877)

### 11.1 Operating point: Heun-**100**, CFG 3 — but tune it per model

Step count is the only lever found that improves body size, extent AND
connectivity together; the CFG and conv-overlap levers each improve one
metric by wrecking another (§10.2, §10.6). 33M: geobody 0.086 → 0.070 →
0.058 → 0.059 at Heun 50 / 75 / 100 / 150, i.e. saturating at 100.
**But it is not monotone for every model**: `rope_big_p442` is 0.061 at
Heun-50 and 0.078 at Heun-100. Report each model at its own best step
count, and say which.

### 11.2 Every arm at Heun-100 / CFG 3 (`specialist_runs/final_table.py`)

| model | geobody | extent | connectivity | NTG err |
|---|---|---|---|---|
| engine split-half band (noise floor) | 0.0190 | 0.0100 | 0.0040 | 0.0001 |
| **77M RoPE, LR 1e-3** (`rope_big_lr1e3`) | **0.0534** | **0.0157** | 0.0316 | 0.0076 |
| 33M RoPE θ=100 (`rope_t100`) | 0.0584 | 0.0181 | 0.0299 | 0.0055 |
| 77M RoPE (`rope_big_t100`) | 0.0627 | 0.0196 | 0.0318 | 0.0081 |
| 33M RoPE θ=10⁴ | 0.0697 | 0.0236 | 0.0288 | 0.0057 |
| 33M RoPE patch 4×4×2 | 0.0704 | 0.0233 | 0.0287 | 0.0069 |
| 33M RoPE LR 1e-3 | 0.0715 | 0.0225 | 0.0271 | 0.0070 |
| 77M RoPE patch 4×4×2 (best at Heun-**50**: 0.0613 / 0.0280) | 0.0781 | 0.0424 | 0.0315 | 0.0075 |
| UNet EMA outpaint | 0.0867 | 0.0401 | **0.0183** | 0.0086 |
| 33M RoPE window 8³ | 0.0927 | 0.0369 | 0.0266 | 0.0065 |
| UNet EMA MultiDiffusion | 0.1135 | 0.0425 | 0.0150 | 0.0082 |
| tiled p442 4-stage ov12 (best tiled DiT, Heun-50) | 0.1176 | 0.0474 | 0.0393 | **0.0003** |
| 77M + conv patch overlap | 0.1263 | 0.0499 | 0.0287 | 0.0072 |
| 64-crop control (one attention window) | 0.6677 | 0.3017 | 0.0804 | 0.0064 |

Best model is **38% better than the UNet** on geobody and **61% better on
extent**, at 0.0157 vs a 0.0100 engine noise floor — i.e. within 2× of the
engine's own Monte-Carlo scatter on body extent.

### 11.3 The two metrics are ANTI-CORRELATED across the whole model family

geobody 0.053 → conn 0.032; 0.058 → 0.030; 0.070 → 0.029; 0.072 → 0.027;
0.093 → 0.027. Every change that improved body size worsened connectivity
and vice versa, over seven independently trained models. They are not two
independent quality axes here; they are two ends of one merge-vs-fragment
axis. Which one to optimise is an application decision (flow simulation
wants connectivity, volumetrics wants body size).

Verified direction (`ResBench/analysis/assembly_stats_ext.py`): our models
are UNDER-connected and OVER-fragmented relative to the engine — τ_x(32)
0.046 vs 0.143, 46 244 bodies vs 15 643, largest-body share 0.195 vs 0.295.
So the conv-overlap head helped connectivity by MERGING large bodies
(τ up at every lag, largest share 0.195 → 0.206, both toward the engine)
while simultaneously dithering the margins into speckle (isolated shale
cells 30 → 63 per 10⁶, engine 4.7) which is what destroyed geobody W1.
An earlier note here said it helped connectivity by disconnecting; that was
wrong and is corrected by the τ table above.

**A specific defect no frozen metric names:** our models emit 6–13× more
isolated single shale voxels than the engine (30–63 vs 4.7 per 10⁶ cells).
Worth a targeted fix and a reported diagnostic.

### 11.4 Where tiling still wins

The tiled 4-stage reference has 20× better NTG accuracy (0.0003 vs
0.005–0.008) because staged conditioning copies context voxels verbatim,
preserving the conditioned proportion exactly. Whole-field decodes every
voxel and drifts ~1 pp. If exact NTG matters more than body geometry, tile.

### 11.5 Reproducing the best model

```
scripts/tier2/train_assembly.py --data-mode crops192 --arch dit --masked-loss \
  --amp bf16 --num-workers 8 --beta2 0.95 --ema-warmup --save-raw \
  --context-share 0 --dit-pos rope --dit-window 16 16 8 --crop 128 128 32 \
  --dit-patch 4 4 4 --dit-hidden 512 --dit-depth 16 --dit-heads 8 \
  --dit-rope-theta 100 --lr 1e-3 --epochs 60 --total-epochs 60 --micro-batch 12
# sample:
scripts/tier2/generate_schedulers.py --sampler wholefield --solver heun \
  --n-steps 100 --cfg 3.0 --overlap 12
```
Checkpoints backed up: `$WORK/dit_runs_backup/{rope_big_lr1e3,rope_big_p442,rope_big_t100,rope_t100}`.

### 11.6 The connectivity deficit is VERTICAL, not lateral (2026-09-13)

Extended diagnostics on the best model's benchmark ensemble
(`ResBench/analysis/assembly_stats_ext.py`) against the engine:

| quantity | best model | engine |
|---|---|---|
| plan-view regions per 10⁴ cells | 22.82 | 22.03 |
| plan-view largest-region share | 0.4031 | 0.4022 |
| 3D bodies (6-connectivity) | 43 982 | 15 643 |
| largest 3D body share | 0.199 | 0.295 |
| τ_z(16) | 0.017 | 0.054 |
| τ_x(32) | 0.047 | 0.143 |
| isolated shale voxels per 10⁶ | 28.1 | 4.7 |
| isolated sand voxels per 10⁶ | 217 | 202 |

**Map view is essentially exact** — region density within 4%, largest-region
share equal to three decimals — while the 3D body count is 2.8× too high and
vertical connectivity is a third of the engine's. Each depth slice is right;
the slices fail to STACK. The 6× excess of isolated shale voxels is the
plausible mechanism: one stray shale cell between two slices separates what
should be a single body under face connectivity.

Consequence for the whole connectivity campaign: every fix tried this week
(conv patch overlap, boundary-weighted loss, smaller windows, finer lateral
patches) acts in x-y, which is why none moved connectivity much. **The
target is the z axis.** The token grid is 16×16×**8** and each token spans
**4 of the 32** depth cells; patch 4×4×2 halves that and did give the best
connectivity among the 33M arms (0.0287) but was tested at only one width.
Next campaign: patch 4×4×1, z-only window widening, or an explicit
vertical-continuity term, and measure τ_z(h) directly rather than the
axis-averaged connectivity MAE which buries it.
