#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=1

CONFIG=${CONFIG:-configs/exp/trace_evc.yaml}
EXP=${EXP:-TRACE_EVC}

python tasks/run.py --config "$CONFIG" --exp_name "$EXP" ${RESET:+--reset}
