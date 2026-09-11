#!/bin/bash
# Full post-training chain for a scheduler-comparison model:
# wait for training -> validation sweep -> E.3 argmin -> generate every
# scheduler the checkpoint supports -> score them all against the engine.
#
#   auto_schedulers.sh <tag> [extra samplers...]
set -uo pipefail
TAG="$1"; shift
SAMPLERS="${*:-native multi raster stage4}"

REPO=/work/08405/ilgar/vista/codes/ResFlow_ls6
BENCH=/work/08405/ilgar/vista/codes/ResBench
RUN=/scratch/08405/ilgar/specialist_runs/$TAG
OUT=/work/08405/ilgar/vista/resbench_eval/sched_$TAG
MAN=$BENCH/results/assembly_reference/lobe/engine_manifest.json

echo "[$TAG] waiting for training"
while ! grep -qh "^Done\." $RUN/train_ddp_rank0_*.log 2>/dev/null; do sleep 60; done

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows
export PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd "$REPO"

if [ ! -f "$RUN/checkpoints/val_losses.json" ]; then
  echo "[$TAG] validation sweep"
  python scripts/tier2/eval_val_assembly.py --env lobe \
    --ckpt-dir "$RUN/checkpoints" --out "$RUN/checkpoints/val_losses.json" \
    > "$RUN/val_eval.log" 2>&1
fi
tail -3 "$RUN/val_eval.log"

conda activate resbench
read -r EPOCH VLOSS <<<"$(python - "$RUN/checkpoints/val_losses.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))['val_loss_by_epoch']
ep, v = min(((int(e), float(x)) for e, x in d.items()), key=lambda t: t[1])
print(ep, v)
PY
)"
echo "[$TAG] ARGMIN epoch=$EPOCH val=$VLOSS"
CKPT="$RUN/checkpoints/inference_epoch$(printf '%03d' "$EPOCH").pt"

conda activate genflows
mkdir -p "$OUT"
for S in $SAMPLERS; do
  echo "--- [$TAG] generating $S ---"
  python scripts/tier2/generate_schedulers.py --sampler "$S" --ckpt "$CKPT" \
    --out-dir "$OUT/$S" --engine-manifest "$MAN" --overlap 24 \
    || echo "[$TAG] sampler $S FAILED (continuing)"
done

echo "--- [$TAG] scoring ---"
conda activate resbench
cd "$BENCH"
ARGS=""
for S in $SAMPLERS; do
  [ "$S" = native ] && ARGS="$ARGS native=$OUT/native" || ARGS="$ARGS $S=$OUT/$S"
done
python analysis/scheduler_compare.py $ARGS --out "$OUT/scheduler_scores.json"
echo "DONE $TAG"
