#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project

python -m py_compile src/fhis/train_bad_step_hazard_v21.py

mkdir -p classifier/v2_runs/bad_step_v21/variant_logs

nohup conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21 \
  --output-dir classifier/v2_runs/bad_step_v21/hazard_causal_gru_platt \
  --positive-weight-multiplier 0.75 \
  --hard-negative-weight 6.0 \
  --correct-negative-weight 2.0 \
  --prefhis-negative-weight 3.0 \
  --calibration-method platt \
  > classifier/v2_runs/bad_step_v21/variant_logs/hazard_causal_gru_platt.log 2>&1 &

echo "started hazard_causal_gru_platt pid=$!"
