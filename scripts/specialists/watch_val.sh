#!/bin/bash
# Live per-checkpoint validation watcher (Addendum E.3 early-visibility aid).
# Runs alongside training on a node that shares the run dir; evaluates each
# new EMA checkpoint as it appears (small batch, concurrent with training)
# and appends "VAL epoch N: <loss>" lines to stdout. Exits when training
# logs "Done." or on 30 min with no new checkpoint after the last.
# Usage: watch_val.sh <run_dir> <env>
set -u
R="$1"; ENVN="$2"
SPEC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows
TMP=$(mktemp -d "/scratch/08405/ilgar/valwatch.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
seen=" "
while true; do
  for f in "$R"/checkpoints/inference_epoch*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case "$seen" in *" $b "*) continue;; esac
    sleep 5
    rm -rf "$TMP/one"; mkdir -p "$TMP/one"; cp "$f" "$TMP/one/"
    if python "$SPEC/eval_val_specialist.py" --env "$ENVN" \
         --ckpt-dir "$TMP/one" --batch-size 32 \
         --out "$TMP/one/v.json" > /dev/null 2>&1; then
      python - "$TMP/one/v.json" << 'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for e, l in d['val_loss_by_epoch'].items():
    print(f"VAL epoch {e}: {l:.6f}", flush=True)
EOF
    else
      echo "VALWATCH: eval failed for $b"
    fi
    seen="$seen$b "
  done
  L=$(ls -t "$R"/train_ddp_rank0_*.log 2>/dev/null | head -1)
  if [ -n "$L" ] && grep -q "^Done\.$" "$L"; then
    echo "VALWATCH: training done"; break
  fi
  sleep 60
done
