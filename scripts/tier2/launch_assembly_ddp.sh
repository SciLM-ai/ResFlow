#!/bin/bash
# Multi-node DDP launcher for assembly-aware lobe training.
#
# Mirrors scripts/specialists/launch_specialist_ddp.sh; the only
# differences are the target script and that the first argument selects
# the data mode instead of the environment (this trains `lobe` only).
#
# Run ONE copy per node:
#   launch_assembly_ddp.sh <mode> <seed> <run_dir> <nnodes> <node_rank> <master> [extra args]
#
# Global batch stays 384 (per-rank 384/nnodes). Auto-resumes from
# <run_dir>/checkpoints/training_state.pt.
set -u
MODE="$1"; SEED="$2"; RUN_DIR="$3"; NNODES="$4"; NODE_RANK="$5"; MASTER="$6"; shift 6

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows
# TACC's XALT wrapper shadows the conda env's libcrypto under srun, which
# breaks pyarrow (used by the parquet-backed datasets). Put the env's own
# libs first. See memory: vista-srun-conda-xalt-openssl.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train_ddp_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "mode=$MODE seed=$SEED run_dir=$RUN_DIR node=$(hostname) node_rank=$NODE_RANK master=$MASTER log=$LOG"

# Distinct port and rendezvous id per concurrent run. RDZV_PORT/RDZV_ID
# let several arms of an ablation share a node set without colliding;
# they default to the per-mode values used by the first two runs.
case "$MODE" in
  native64) DEF_PORT=29601 ;;
  crops192) DEF_PORT=29602 ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
PORT="${RDZV_PORT:-$DEF_PORT}"
RID="${RDZV_ID:-asm_${MODE}}"

torchrun --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="${MASTER}:${PORT}" \
  --rdzv_id="$RID" \
  "$SCRIPT_DIR/train_assembly.py" \
  --data-mode "$MODE" --seed "$SEED" --run-dir "$RUN_DIR" "$@" \
  2>&1 | tee -a "$LOG"
