#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
export PYTHONPATH=src

features="classifier/v2_runs/bad_step_v25_train2000/step_hidden_states.pt"
manifest="classifier/v2_runs/bad_step_v25_train2000/manifest.jsonl"
config="classifier/v25_probe_remote.yaml"

run_one() {
  local calibration="$1"
  local out_dir="classifier/v2_runs/bad_step_v25_train2000/hazard_recall_big_${calibration}"
  mkdir -p "${out_dir}"
  conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21 \
    --config "${config}" \
    --features "${features}" \
    --manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --layer-embed-dim 384 \
    --sequence-hidden-dim 384 \
    --step-mlp-dim 384 \
    --sequence-num-layers 2 \
    --trace-batch-size 16 \
    --max-epochs 120 \
    --patience 18 \
    --positive-weight-multiplier 1.5 \
    --positive-sample-weight 6.0 \
    --correct-negative-weight 1.5 \
    --prefhis-negative-weight 2.0 \
    --dropout 0.25 \
    --calibration-method "${calibration}"
}

run_one platt
run_one isotonic
