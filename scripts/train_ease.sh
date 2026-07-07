#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python tasks/ease/extract_xvectors_mead.py
python tasks/ease/extract_units_mead.py
python tasks/ease/build_mead_binary.py
python tasks/ease/train_mead_ease.py
