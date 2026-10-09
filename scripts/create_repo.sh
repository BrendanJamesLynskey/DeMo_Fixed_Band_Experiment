#!/usr/bin/env bash
# Create the GitHub repo for this experiment and push main. Run it yourself (the local Claude
# session cannot create repos). Usage: scripts/create_repo.sh [--private|--public]
set -euo pipefail
cd "$(dirname "$0")/.."
VIS="${1:---private}"
gh repo create BrendanJamesLynskey/DeMo_Fixed_Band_Experiment "$VIS" \
  --description "Can DeMo's top-k be replaced by a fixed frequency band (no comparisons) for a photonic link module? Simulated data parallelism on CPU." \
  --source . --remote origin --push
