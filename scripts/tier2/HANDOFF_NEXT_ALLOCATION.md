# Next allocation: start here

Full write-up: `ResFlow_ls6/scripts/tier2/DIT_FINDINGS.md` §10–§11.
Job 988877 (32 nodes) ended 2026-09-13 12:27. Everything below is committed;
checkpoints are in `$WORK/dit_runs_backup/` (7.3 GB, 14 runs).

## The model to build on
`rope_t100` — 33M, whole-field RoPE DiT. Train:
```
scripts/tier2/train_assembly.py --data-mode crops192 --arch dit --masked-loss \
  --amp bf16 --num-workers 8 --beta2 0.95 --ema-warmup --save-raw \
  --context-share 0 --dit-pos rope --dit-window 16 16 8 --crop 128 128 32 \
  --dit-patch 4 4 4 --dit-rope-theta 100 --lr 5e-4 --epochs 80 --total-epochs 80
```
Sample: `--sampler wholefield --solver heun --n-steps 100 --cfg 3 --overlap 12`.

## The one open problem, with its diagnosis
Connectivity 0.030 vs the UNet's 0.018. **The deficit is vertical.** Plan view
matches the engine (regions/1e4 22.8 vs 22.0; largest-region share 0.403 vs
0.402) but 3D bodies are 2.8× too many and τ_z(16) is 0.017 vs 0.054 — the
slices are right and fail to stack. Also 6× too many isolated shale voxels
(28 vs 4.7 per 10⁶).

Four interventions already tried and rejected, ALL of them lateral:
conv patch overlap (doubles geobody, replicated on two models),
boundary-weighted loss (worsens both; `--boundary-weight` is implemented),
smaller windows (trade-off), more ODE steps (helps, saturates at 100).

**Try next, in this order:** patch 4×4×1 (or 4×4×2 at 77M width, which gave
the best connectivity of the 33M arms); a taller token grid; an explicit
vertical-continuity penalty. Report τ_z(h) directly — the axis-averaged
connectivity MAE hides it, and lag 1 contributes exactly zero for every model.

## Two ablations a reviewer will demand
1. RoPE vs learned positions at matched crop size (the 64-crop control
   `rope_big_crop64` covers crop size, not position scheme).
2. The mask distribution: all whole-field runs used `--context-share 0`.

## Traps, measured
- CFG > 3 improves benchmark geobody 4× while merging lobes. Report CFG 3.
- Step count is not monotone for every model (`rope_big_p442`: 0.061 at
  Heun-50, 0.078 at Heun-100). Tune per model and say which.
- A model trained for whole-field CANNOT be tiled afterwards, and vice versa;
  the choice is made at training time.
- Never edit a running shell script (bash re-reads by byte offset).
- Never run evaluation on a node that is training (user's standing rule).
