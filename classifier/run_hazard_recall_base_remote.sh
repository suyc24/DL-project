#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
python -m py_compile src/fhis/train_bad_step_hazard_v21.py
mkdir -p classifier/v2_runs/bad_step_v21/variant_logs

run_one() {
  local name="$1"
  local calibration="$2"
  local pos_mult="$3"
  echo "starting ${name} calibration=${calibration} pos_mult=${pos_mult}"
  conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21     --output-dir "classifier/v2_runs/bad_step_v21/${name}"     --layer-embed-dim 192     --sequence-hidden-dim 192     --step-mlp-dim 192     --sequence-num-layers 1     --trace-batch-size 32     --max-epochs 100     --patience 16     --positive-weight-multiplier "${pos_mult}"     --hard-positive-weight 8.0     --hard-negative-weight 4.0     --correct-negative-weight 2.0     --prefhis-negative-weight 2.5     --dropout 0.20     --calibration-method "${calibration}"
  echo "finished ${name}"
}

run_one hazard_recall_base_p15_iso isotonic 1.5
run_one hazard_recall_base_p15_platt platt 1.5
run_one hazard_recall_base_p20_iso isotonic 2.0
run_one hazard_recall_base_p20_platt platt 2.0
