#!/bin/bash
# Finalize the lobe extension (E.3 fired branch): official paired val eval on
# the 80-run checkpoints, argmin over the UNION of both PV runs, generation
# from the selected checkpoint, ResBench scoring, posthoc, comparison rebuild.
set -euo pipefail
SPEC=/work/08405/ilgar/vista/codes/ResFlow_ls6/scripts/specialists
EVAL=/work/08405/ilgar/vista/resbench_eval
RB=/work/08405/ilgar/vista/codes/ResBench
R40=/scratch/08405/ilgar/specialist_runs/lobe
R80=/scratch/08405/ilgar/specialist_runs/lobe_80ep
OUT=$RB/results/specialists

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genflows

echo "=== [1/5] official paired val eval on 80-run checkpoints ==="
python "$SPEC/eval_val_specialist.py" --env lobe \
  --ckpt-dir "$R80/checkpoints" --out "$R80/checkpoints/val_losses.json"

BEST=$(python - << 'EOF'
import json
a = json.load(open('/scratch/08405/ilgar/specialist_runs/lobe/checkpoints/val_losses.json'))
b = json.load(open('/scratch/08405/ilgar/specialist_runs/lobe_80ep/checkpoints/val_losses.json'))
cand = [(v, '40', int(e)) for e, v in a['val_loss_by_epoch'].items()]
cand += [(v, '80', int(e)) for e, v in b['val_loss_by_epoch'].items()]
v, run, ep = min(cand)
print(f'{run} {ep} {v:.6f}')
EOF
)
set -- $BEST; RUN=$1; EP=$2; V=$3
if [ "$RUN" = "40" ]; then DIR=$R40; else DIR=$R80; fi
CKPT=$DIR/checkpoints/inference_epoch$(printf '%03d' "$EP").pt
echo "UNION_ARGMIN run=$RUN epoch=$EP val=$V ckpt=$CKPT"

echo "=== [2/5] generation (512, Table 6, offset manifest) ==="
python /work/08405/ilgar/vista/codes/ResFlow_ls6/scripts/rebuttal_eval/generate_ensembles.py \
  --manifest "$EVAL/specialist_manifest_lobe.csv" \
  --conds "$EVAL/conds.npz" --ckpt "$CKPT" \
  --out-dir "$EVAL/specialist_lobe_ext" --ensemble a --self-test

echo "=== [3/5] ResBench scoring ==="
conda activate resbench
python -m resbench.run \
  --pred-dir "$EVAL/specialist_lobe_ext/ensemble_a" \
  --ref-dir "$EVAL/reference" \
  --out "$OUT/lobe_ext_metrics.parquet" --workers 8
mv "$OUT/report.npy" "$OUT/lobe_ext_report.npy" 2>/dev/null || true

echo "=== [4/5] posthoc ==="
cd "$RB"
python analysis/posthoc_geobody.py --ref-dir "$EVAL/reference" \
  --pred-dir "$EVAL/specialist_lobe_ext/ensemble_a" \
  --out-dir "$OUT/posthoc_lobe_ext"

echo "=== [5/5] rebuild comparison ==="
python analysis/build_specialists_comparison.py

echo "LOBE80_FINALIZE_DONE best_run=$RUN best_epoch=$EP best_val=$V"
