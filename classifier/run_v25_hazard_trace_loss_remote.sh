#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
export PYTHONPATH=src

features="classifier/v2_runs/bad_step_v25_train2000/step_hidden_states.pt"
manifest="classifier/v2_runs/bad_step_v25_train2000/manifest.jsonl"
config="classifier/v25_probe_remote.yaml"

run_variant() {
  local name="$1"
  local correct_weight="$2"
  local prefhis_weight="$3"
  local positive_weight="$4"
  local neg_margin="$5"
  local pos_margin="$6"
  conda run -n fhis-v2 python -m fhis.train_bad_step_hazard_v21 \
    --config "${config}" \
    --features "${features}" \
    --manifest "${manifest}" \
    --output-dir "classifier/v2_runs/bad_step_v25_train2000/${name}" \
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
    --calibration-method platt \
    --correct-trace-max-logit-weight "${correct_weight}" \
    --prefhis-trace-max-logit-weight "${prefhis_weight}" \
    --positive-trace-logit-weight "${positive_weight}" \
    --trace-negative-logit-margin "${neg_margin}" \
    --trace-positive-logit-margin "${pos_margin}"
}

run_variant hazard_trace_loss_base_c05_pre03_pos05_platt 0.5 0.3 0.5 -1.0 1.0
run_variant hazard_trace_loss_base_c10_pre05_pos08_platt 1.0 0.5 0.8 -1.0 1.0
