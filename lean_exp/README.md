# Lean transition-contract experiments

This directory now keeps only successful artifacts for the non-geometry CMO 2023
step-contract exploration.

Main files:

- `non_geometry_step_contracts.lean`: consolidated successful local transition
  contracts for all non-geometry problems in the supplied pages.
- `successful_contracts/`: successful checked Lean contracts copied out of the
  earlier experiment runs.
- `successful_contracts/negative_error_examples.lean`: a successful Lean proof
  that the earlier wrong Day1 P2 candidate `2 - sqrt 3` is incompatible with the
  trusted `1/2` target.
- `deepseek_pro_step_suite.py`: the current per-step DeepSeek v4 Pro compiler
  feedback suite. Raw outputs from previous runs were removed.
- `success_only_report.md`: coverage and interpretation.

Raw failed trajectories and exploratory outputs have been deleted.
