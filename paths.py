"""Default file paths, all relative to the repo root.

Everything defaults to a location under the repo (``data/`` for datasets and
generated features, ``checkpoints/`` for weights, ``pretrained/`` for cached
frozen extractors). Override any root with the matching ``TRACE_*`` environment
variable, e.g. point ``TRACE_DATA_ROOT`` at a scratch disk::

    export TRACE_DATA_ROOT=/scratch/you/trace_data
    export TRACE_MEAD_RAW=/datasets/MEAD

External inputs you must provide (no default file ships with the repo): the raw
MEAD/ESD audio, the HuBERT k-means model (``km200.bin``), the speaker split JSON,
and the TRACE-Instruct prompt files. See ``data/README.md`` for the full layout.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _root(env: str, default) -> Path:
    return Path(os.environ.get(env, str(default)))


DATA_ROOT = _root("TRACE_DATA_ROOT", REPO_ROOT / "data")
CKPT_ROOT = _root("TRACE_CKPT_ROOT", REPO_ROOT / "checkpoints")
MODEL_ROOT = _root("TRACE_MODEL_ROOT", REPO_ROOT / "pretrained")

PROMPT_OUT = _root("TRACE_PROMPT_OUT", DATA_ROOT / "prompts")
ESD_PROMPTS = PROMPT_OUT / "esd_relative_textrol_qwen_prompts.jsonl"
ESD_PROMPT_EMB = PROMPT_OUT / "esd_relative_textrol_qwen_prompt_emb_e5.pt"
MEAD_PROMPTS = PROMPT_OUT / "mead_intra_qwen_prompts.jsonl"
MEAD_PROMPT_EMB = PROMPT_OUT / "mead_intra_qwen_prompt_emb_e5.pt"

ESD_EMO2VEC = MODEL_ROOT / "emo2vec_plus_large_1024.pt"
MEAD_EMO2VEC = MODEL_ROOT / "emo2vec_mead_1024.pt"

MEAD_SPLIT = _root("TRACE_MEAD_SPLIT", DATA_ROOT / "features" / "speaker_splits.json")
MEAD_VAD = _root("TRACE_MEAD_VAD", DATA_ROOT / "vad" / "vad.jsonl")
ESD_INDEX_SPLIT = _root("TRACE_ESD_INDEX_SPLIT", DATA_ROOT / "esd_data" / "esd_index_split_20_30_300.jsonl")
ESD_PAIRS = _root("TRACE_ESD_PAIRS", DATA_ROOT / "esd_data" / "esd_targetvad_emo2vec_pairs.jsonl")

def _model(env: str, local, hub_id: str) -> str:
    """HF model ref: TRACE_* env > local dir under pretrained/ (offline-safe) > Hub id."""
    v = os.environ.get(env)
    if v:
        return v
    local = Path(local)
    return str(local) if local.exists() else hub_id


E5_MODEL = _model("TRACE_E5_MODEL", MODEL_ROOT / "e5-base-v2", "intfloat/e5-base-v2")

EMOTION2VEC_MODEL = _root("TRACE_EMOTION2VEC_MODEL", "iic/emotion2vec_plus_large")
ODYSSEY_VAD_MODEL = _model("TRACE_ODYSSEY_VAD_MODEL", MODEL_ROOT / "odyssey_ser",
                           "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes")

EMO_COMPASS_CKPT = CKPT_ROOT / "emo_compass"
ESD_EMOTION_HEAD = EMO_COMPASS_CKPT / "emo2vec_emotion_head_esd.pt"
MEAD_EMOTION_HEAD = EMO_COMPASS_CKPT / "mead_emotion_head.pt"
MEAD_INTENSITY_HEAD = EMO_COMPASS_CKPT / "mead_intensity_head.pt"
ESD_FLOW_CKPT_DIR = EMO_COMPASS_CKPT / "trace_flow_esd"
MEAD_FLOW_CKPT_DIR = EMO_COMPASS_CKPT / "trace_flow_mead_intra"

MEAD_RAW_AUDIO = _root("TRACE_MEAD_RAW", DATA_ROOT / "raw" / "MEAD")
HUBERT_KM200 = _root("TRACE_HUBERT_KM200", MODEL_ROOT / "km200.bin")
MEAD_PROCESSED = DATA_ROOT / "features" / "processed"
MEAD_BINARY = DATA_ROOT / "features" / "binary"
EASE_ROOT = DATA_ROOT / "speaker"
EASE_XVECTORS = EASE_ROOT / "cache" / "xvectors.pt"
EASE_DIR = EASE_ROOT / "EASE_embeddings"
EASE_STATS = EASE_ROOT / "ease_stats.npz"
EASE_CKPT = EASE_ROOT / "ease.pt"

F0_UTT_STATS = DATA_ROOT / "features" / "f0_utt_stats.npz"
ENERGY_UTT_STATS = DATA_ROOT / "features" / "energy_raw_utt_stats.npz"
DIFFUSED_TARGET_EMB = DATA_ROOT / "features" / "diffused" / "diffused_target_emb.pt"
