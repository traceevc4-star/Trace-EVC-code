#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

python tasks/ease/extract_units_mead.py
python tasks/ease/extract_xvectors_mead.py
python tasks/ease/build_mead_binary.py
python tasks/preprocess/extract_f0.py
python tasks/preprocess/extract_energy.py --norm
python tasks/preprocess/extract_energy.py --raw
python tasks/preprocess/extract_vad.py

echo "done. next: scripts/train_ease.sh -> train_emo_compass.sh -> train_synthesis.sh"
