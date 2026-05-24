#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
python -m py_compile src/fhis/train_bad_step_hazard_v21.py
mkdir -p classifier/v2_runs/bad_step_v21/variant_logs

run_one() {
  local name="$1"
  local calibration="$2"
  echo "starting ${name} calibration=${calibration}"
  conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21     --output-dir "classifier/v2_runs/bad_step_v21/${name}"     --layer-embed-dim 384     --sequence-hidden-dim 384     --step-mlp-dim 384     --sequence-num-layers 2     --trace-batch-size 16     --max-epochs 120     --patience 18     --positive-weight-multiplier 1.25     --hard-positive-weight 6.0     --hard-negative-weight 6.0     --correct-negative-weight 3.0     --prefhis-negative-weight 3.0     --dropout 0.25     --calibration-method "${calibration}"
  echo "finished ${name}"
}

run_one hazard_recall_big_iso isotonic
run_one hazard_recall_big_platt platt
