#!/bin/bash
# Full post-training chain for one arm: wait for training to finish,
# sweep validation over every checkpoint, apply the E.3 argmin rule,
# generate the three ensembles and score both fusion rules.
#
#   auto_finalize.sh <tag>
set -uo pipefail
TAG="$1"
RUN=/scratch/08405/ilgar/specialist_runs/$TAG
REPO=/work/08405/ilgar/vista/codes/ResFlow_ls6

echo "[$TAG] waiting for training to finish"
while ! grep -qh "^Done\." $RUN/train_ddp_rank0_*.log 2>/dev/null; do sleep 60; done
echo "[$TAG] training done"

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

exec "$REPO/scripts/tier2/finalize_assembly_model.sh" "$RUN" "$TAG"
