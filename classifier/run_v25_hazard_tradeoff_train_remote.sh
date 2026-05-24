#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
export PYTHONPATH=src

features="classifier/v2_runs/bad_step_v25_train2000/step_hidden_states.pt"
manifest="classifier/v2_runs/bad_step_v25_train2000/manifest.jsonl"
config="classifier/v25_probe_remote.yaml"

run_variant() {
  local name="$1"
  local pos_mult="$2"
  local pos_weight="$3"
  local neg_weight="$4"
  local dropout="$5"
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
    --positive-weight-multiplier "${pos_mult}" \
    --positive-sample-weight "${pos_weight}" \
    --correct-negative-weight "${neg_weight}" \
    --prefhis-negative-weight "${neg_weight}" \
    --dropout "${dropout}" \
    --calibration-method platt
}

run_variant hazard_tradeoff_base_neg3_pos5_platt 1.25 5.0 3.0 0.22
run_variant hazard_tradeoff_base_neg4_pos4_platt 1.00 4.0 4.0 0.25
