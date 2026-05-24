#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/DL-project
mkdir -p classifier/v2_runs/bad_step_v25_train2000

export PYTHONPATH=src
conda run -n fhis-v2 python -m fhis.extract_hidden_states_transformers \
  --config classifier/v25_probe_remote.yaml \
  --traces classifier/v2_runs/label_expansion/v25_train_2000_20260516/generated_traces.jsonl \
  --labels classifier/v2_runs/label_expansion/v25_train_2000_20260516/fhis_labels_assistant_merged_2000_high.jsonl \
  --output classifier/v2_runs/bad_step_v25_train2000/step_hidden_states.pt \
  --device cuda
