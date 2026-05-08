# lean_fhis

Lean 4 + mathlib environment for checking small proof fragments during FHIS
dataset/probe experiments.

## Setup

Lean is managed by `elan`; this project pins the toolchain in
`lean-toolchain` and pins mathlib in `lakefile.toml` / `lake-manifest.json`.

```bash
source "$HOME/.elan/env"
cd /root/DL-project/lean_fhis
lake exe cache get
lake build
```

Use `lake env lean <file>.lean` to check a single Lean file in this
environment.
