#!/bin/bash
# Specialist training launcher (ResBench EVAL.md Addendum E).
#
# Usage:  launch_specialist.sh <env> <seed> <run_dir> [extra train args...]
#   e.g.  launch_specialist.sh channel:PV_SHOESTRING 8101 $SCRATCH/specialist_runs/pv_shoestring
#         launch_specialist.sh lobe                  8102 $SCRATCH/specialist_runs/lobe
# Wave 2 is one more line, e.g.:
#         launch_specialist.sh delta 8103 $SCRATCH/specialist_runs/delta
#
# Resume after a kill: rerun the exact same line — training auto-resumes
# from <run_dir>/checkpoints/training_state.pt. To reset, delete that file.
#
# Intended to run inside tmux/screen on one GH200 node:
#   tmux new-session -d -s spec_pv \
#     "bash .../launch_specialist.sh channel:PV_SHOESTRING 8101 $SCRATCH/specialist_runs/pv_shoestring"
set -u
ENV_NAME="$1"; SEED="$2"; RUN_DIR="$3"; shift 3

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "env=$ENV_NAME seed=$SEED run_dir=$RUN_DIR node=$(hostname) log=$LOG"

python "$SCRIPT_DIR/train_specialist.py" \
  --env "$ENV_NAME" --seed "$SEED" --run-dir "$RUN_DIR" "$@" \
  2>&1 | tee -a "$LOG"
