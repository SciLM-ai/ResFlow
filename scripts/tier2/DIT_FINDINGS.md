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
