#!/bin/bash
# Post-training pipeline for one specialist (EVAL.md Addendum E, phases E.3-E.5):
#   1. per-checkpoint validation loss + anti-undertraining verdict (GPU)
#   2. 512-volume generation from the best-val checkpoint (GPU, Table 6 settings)
#   3. ResBench scoring vs the frozen reference (CPU)
#   4. per-volume compartmentalization posthoc (CPU)
# Run on a node whose GPU is free (i.e. after that specialist finished).
#
# Usage: postrain_pipeline.sh <env> <tag>
#   postrain_pipeline.sh channel:PV_SHOESTRING pv_shoestring
#   postrain_pipeline.sh lobe lobe
set -euo pipefail
ENV_NAME="$1"; TAG="$2"

SPEC=/work/08405/ilgar/vista/codes/ResFlow_ls6/scripts/specialists
EVAL=/work/08405/ilgar/vista/resbench_eval
RB=/work/08405/ilgar/vista/codes/ResBench
RUN=/scratch/08405/ilgar/specialist_runs/$TAG
OUT=$RB/results/specialists
mkdir -p "$OUT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "=== [1/4] validation losses ($ENV_NAME) ==="
conda activate genflows
python "$SPEC/eval_val_specialist.py" --env "$ENV_NAME" \
  --ckpt-dir "$RUN/checkpoints" --out "$RUN/checkpoints/val_losses.json"

BEST=$(python -c "
import json; d = json.load(open('$RUN/checkpoints/val_losses.json'))
print(d['best_epoch'])")
FIRES=$(python -c "
import json; d = json.load(open('$RUN/checkpoints/val_losses.json'))
print('YES' if d['anti_undertraining_fires'] else 'NO')")
CKPT=$RUN/checkpoints/inference_epoch$(printf '%03d' "$BEST").pt
echo "BEST_EPOCH=$BEST ANTI_UNDERTRAINING_FIRES=$FIRES CKPT=$CKPT"

echo "=== [2/4] generation (512 volumes, Table 6, seed offset +500000) ==="
python /work/08405/ilgar/vista/codes/ResFlow_ls6/scripts/rebuttal_eval/generate_ensembles.py \
  --manifest "$EVAL/specialist_manifest_$TAG.csv" \
  --conds "$EVAL/conds.npz" \
  --ckpt "$CKPT" \
  --out-dir "$EVAL/specialist_$TAG" \
  --ensemble a --self-test

echo "=== [3/4] ResBench scoring ==="
conda activate resbench
python -m resbench.run \
  --pred-dir "$EVAL/specialist_$TAG/ensemble_a" \
  --ref-dir "$EVAL/reference" \
  --out "$OUT/${TAG}_metrics.parquet" \
  --workers 8
mv "$RB/results/specialists/report.npy" "$OUT/${TAG}_report.npy" 2>/dev/null || true

echo "=== [4/4] compartmentalization posthoc ==="
cd "$RB"
python analysis/posthoc_geobody.py \
  --ref-dir "$EVAL/reference" \
  --pred-dir "$EVAL/specialist_$TAG/ensemble_a" \
  --out-dir "$OUT/posthoc_$TAG"

echo "PIPELINE_DONE tag=$TAG best_epoch=$BEST fires=$FIRES"
