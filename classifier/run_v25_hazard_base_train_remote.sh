#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
export PYTHONPATH=src

conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21 \
  --config classifier/v25_probe_remote.yaml \
  --features classifier/v2_runs/bad_step_v25_train2000/step_hidden_states.pt \
  --manifest classifier/v2_runs/bad_step_v25_train2000/manifest.jsonl \
  --output-dir classifier/v2_runs/bad_step_v25_train2000/hazard_recall_base_platt \
  --layer-embed-dim 192 \
  --sequence-hidden-dim 192 \
  --step-mlp-dim 192 \
  --sequence-num-layers 1 \
  --trace-batch-size 32 \
  --max-epochs 100 \
  --patience 16 \
  --positive-weight-multiplier 1.5 \
  --positive-sample-weight 6.0 \
  --correct-negative-weight 1.5 \
  --prefhis-negative-weight 2.0 \
  --dropout 0.20 \
  --calibration-method platt
