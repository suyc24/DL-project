# v2.5 Train Expansion Label Queue

Status: train expansion generated and copied locally; clean eval is still remote/in progress at the time this queue was prepared.

Files:

- `train_queue_metadata.jsonl`: metadata-only trace queue for 2000 train-expansion traces.
- `train_blank_labels.jsonl`: blank FHIS label template with one row per train trace.
- `manifest.json`: queue summary and leakage checks for the current snapshot.

Important policy/safety note:

This directory intentionally does not contain problem text, prompts, completions, parsed steps, reference solutions, token ids, or token logprobs in the queue metadata/template files. Actual FHIS labels should be filled by an approved offline/local labeler or human annotation path that does not send private trace contents to an external Codex/OpenAI service.

Train summary:

- traces: 2000
- unique trace ids: 2000
- unique problem ids: 500
- traces per problem: 4
- dataset/subset: Hothan/OlympiadBench / OE_TO_maths_en_COMP
- rough final correctness: True=717, False=1139, None=144
- step-parseable: 2000/2000
- average steps: 5.826

Clean eval handling:

The natural clean eval set must remain held out from training, calibration, and threshold/model selection. Re-run `classifier/v25_prepare_label_queues.py` with the completed clean-eval trace file once it reaches 500 traces.
