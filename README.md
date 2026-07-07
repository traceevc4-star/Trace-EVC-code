# TRACE-EVC: Text-Guided Relative Affective Control for Zero-Shot Emotional Voice Conversion

Convert the emotion of a source utterance from a natural-language instruction
describing a **relative** change (e.g. *"make it slightly calmer"*), preserving
speaker identity and content. Three stages: **EASE** (speaker embedding),
**Emo-Compass** (instruction → target emotion embedding), **Synthesis** (renderer).

## Installation

```bash
git clone https://github.com/traceevc/TRACE-EVC.git
cd TRACE-EVC

conda create -n trace python=3.10 pip -y
conda activate trace
python -m pip install -r requirements.txt
```

Run everything from the repo root with `PYTHONPATH=.`. Dataset / checkpoint /
pretrained-model locations are set in `paths.py` (or via `TRACE_*` env vars).

## Data preparation

Provide the raw MEAD/ESD audio, the HuBERT `km200.bin`, the speaker split, and the
TRACE-Instruct prompts (see [`data/README.md`](data/README.md) for the exact
locations / `TRACE_*` overrides), then build every feature file from raw audio:

```bash
export PYTHONPATH=.
bash scripts/prepare_data.sh        # units, x-vectors, binary, F0, energy, VAD
```

## Training

```bash
export PYTHONPATH=.
bash scripts/train_ease.sh          # Stage 1 — EASE speaker encoder
bash scripts/train_emo_compass.sh   # Stage 2 — Emo-Compass flow + heads + target bank
bash scripts/train_synthesis.sh     # Stage 3 — DurFlex-EVC renderer
```

## Inference

```bash
export PYTHONPATH=.
python infer_end2end.py \
  --source-wav src.wav \
  --prompt-text "make it much calmer and far less intense" \
  --emotion angry \
  --flow-ckpt checkpoints/emo_compass/trace_flow_mead_intra/best_trace_mead.pt \
  --out-wav out.wav
```

`--emotion {angry,happy,sad,surprised}` is the source emotion axis. Source features
(HuBERT, EASE, emotion2vec, VAD) are extracted automatically; skip live extraction
with `--source-affect` / `--source-emo2vec` / `--source-vad`.
