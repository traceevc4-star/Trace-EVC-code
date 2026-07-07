#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1

python tasks/emo_compass/train_emotion_head_esd.py
python tasks/emo_compass/train_emotion_head_mead.py
python tasks/emo_compass/train_intensity_head.py

python tasks/emo_compass/encode_prompts.py \
  --prompts-jsonl data/prompts/esd_relative_textrol_qwen_prompts.jsonl \
  --out data/prompts/esd_relative_textrol_qwen_prompt_emb_e5.pt
python tasks/emo_compass/encode_prompts.py \
  --prompts-jsonl data/prompts/mead_intra_qwen_prompts.jsonl \
  --out data/prompts/mead_intra_qwen_prompt_emb_e5.pt

python tasks/emo_compass/train_flow_esd.py
python tasks/emo_compass/train_flow_mead.py

python tasks/emo_compass/precompute_targets.py
