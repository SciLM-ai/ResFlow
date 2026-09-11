#!/bin/bash
# Generate + score one assembly-aware model against the frozen Addendum E
# lobe reference.
#
#   score_assembly_model.sh <run_dir> <epoch> <out_tag>
#
# Produces three model ensembles at the E.2/E.7 shared condition:
#   A  native      -- 250 volumes, Table 6 settings, empty mask
#   B1 assembly    -- 10 MultiDiffusion assemblies (velocity averaging)
#   B2 outpaint    -- 10 raster-order outpainting assemblies
# then scores B1-vs-C and B2-vs-C (and A-vs-C) with the frozen band.
# Everything but the fusion rule is identical between B1 and B2.
set -euo pipefail

RUN_DIR="$1"; EPOCH="$2"; TAG="$3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
REPO=/work/08405/ilgar/vista/codes/ResFlow_ls6
BENCH=/work/08405/ilgar/vista/codes/ResBench
MAN=$BENCH/results/assembly_reference/lobe/engine_manifest.json
OUT=/work/08405/ilgar/vista/resbench_eval/assembly_$TAG
CKPT="$RUN_DIR/checkpoints/inference_epoch$(printf '%03d' "$EPOCH").pt"

mkdir -p "$OUT/multi" "$OUT/outpaint"
echo "=== $TAG :: ckpt=$CKPT ==="

conda activate genflows
export PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd "$REPO"

echo "--- [1/4] native (A) ---"
python scripts/rebuttal_eval/generate_assembly_ensembles.py native \
  --engine-manifest "$MAN" --out-dir "$OUT" --env lobe --ckpt "$CKPT"

echo "--- [2/4] MultiDiffusion assemblies (B1) ---"
python scripts/rebuttal_eval/generate_assembly_ensembles.py assembly \
  --engine-manifest "$MAN" --out-dir "$OUT/multi" --env lobe --ckpt "$CKPT"

echo "--- [3/4] outpaint assemblies (B2) ---"
python scripts/rebuttal_eval/generate_assembly_ensembles.py outpaint \
  --engine-manifest "$MAN" --out-dir "$OUT/outpaint" --env lobe --ckpt "$CKPT"

echo "--- [4/4] scoring ---"
conda activate resbench
cd "$BENCH"
for FUSION in multi outpaint; do
  python analysis/assembly_stats.py --env-slug lobe --split-seed 20260815 \
    --native-dir "$OUT" --assembly-dir "$OUT/$FUSION" \
    --engine-dir results/assembly_reference \
    --out-dir "$OUT/scored_$FUSION"
  echo "### $TAG / $FUSION"
  cat "$OUT/scored_$FUSION/assembly_stats.md"
done
echo "DONE $TAG"
