#!/bin/bash
# Wait for a run's validation sweep, apply the Addendum E.3 argmin rule,
# then generate and score that checkpoint.
#
#   finalize_assembly_model.sh <run_dir> <tag>
set -uo pipefail

RUN_DIR="$1"; TAG="$2"
VAL="$RUN_DIR/checkpoints/val_losses.json"

echo "[$TAG] waiting for $VAL"
while [ ! -f "$VAL" ]; do sleep 60; done

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate resbench
read -r EPOCH VLOSS <<<"$(python - "$VAL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
by = d['val_loss_by_epoch']
ep, v = min(((int(e), float(x)) for e, x in by.items()), key=lambda t: t[1])
print(ep, v)
PY
)"
echo "[$TAG] ARGMIN epoch=$EPOCH val=$VLOSS"

exec /work/08405/ilgar/vista/codes/ResFlow_ls6/scripts/tier2/score_assembly_model.sh \
  "$RUN_DIR" "$EPOCH" "$TAG"
