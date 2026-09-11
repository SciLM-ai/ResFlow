#!/bin/bash
# Standing GPU-idle watchdog.
#
# The failure this exists to prevent: a scoring step is run in the
# foreground, finishes, and nothing is left watching -- so the whole
# allocation sits idle until somebody notices. That happened for ~2.5 h.
#
# Polls GPU utilisation across the allocation and exits (which fires a
# task notification) as soon as the allocation has been mostly idle for
# two consecutive checks, or when a named marker file appears.
#
#   keepalive.sh [idle_threshold_nodes] [poll_seconds]
set -uo pipefail
THRESH="${1:-24}"      # exit if >= this many nodes are idle
POLL="${2:-120}"
NNODES=$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)
strikes=0

while true; do
  if ! squeue -u "$USER" -s 2>/dev/null | grep -q "$SLURM_JOB_ID"; then
    echo "ALLOCATION ENDED $(date)"; exit 0
  fi
  idle=$(srun --overlap -N "$NNODES" -n "$NNODES" --ntasks-per-node=1 \
           bash -c 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader' \
           2>/dev/null | tr -d ' %' | awk '$1<=50{c++} END{print c+0}')
  busy=$((NNODES - idle))
  echo "$(date +%H:%M:%S) busy=$busy idle=$idle"
  if [ "$idle" -ge "$THRESH" ]; then
    strikes=$((strikes + 1))
    if [ "$strikes" -ge 2 ]; then
      echo "IDLE: $idle/$NNODES nodes idle for 2 consecutive checks at $(date)"
      exit 0
    fi
  else
    strikes=0
  fi
  sleep "$POLL"
done
