#!/bin/bash
# Multi-node DDP specialist launcher (ResBench EVAL.md Addendum E, Amendment E-1).
#
# Run ONE copy per node, e.g. lobe across two GH200 nodes:
#   node A (master):  launch_specialist_ddp.sh lobe 8102 $SCRATCH/specialist_runs/lobe 2 0 <masterhost>
#   node B:           launch_specialist_ddp.sh lobe 8102 $SCRATCH/specialist_runs/lobe 2 1 <masterhost>
#
# Global batch stays 384 (per-rank 384/nnodes, micro-batch 96 accumulation);
# training auto-resumes from <run_dir>/checkpoints/training_state.pt.
set -u
ENV_NAME="$1"; SEED="$2"; RUN_DIR="$3"; NNODES="$4"; NODE_RANK="$5"; MASTER="$6"; shift 6

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train_ddp_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "env=$ENV_NAME seed=$SEED run_dir=$RUN_DIR node=$(hostname) node_rank=$NODE_RANK master=$MASTER log=$LOG"

torchrun --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="${MASTER}:29513" \
  --rdzv_id="spec_${ENV_NAME//:/_}" \
  "$SCRIPT_DIR/train_specialist.py" \
  --env "$ENV_NAME" --seed "$SEED" --run-dir "$RUN_DIR" "$@" \
  2>&1 | tee -a "$LOG"
