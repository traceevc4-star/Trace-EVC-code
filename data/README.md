# Data layout & preparation

Everything here lives under the repo's `data/` folder by default (override the root
with `TRACE_DATA_ROOT`, or any individual path with the matching `TRACE_*` env var —
see `../paths.py`). **Nothing in this folder is committed** — you provide the raw
datasets and generate the rest with the scripts below.

## What you must provide (external inputs)

| Path (default) | What it is |
|---|---|
| `data/raw/MEAD` | raw MEAD audio, `…/<spk>/…/<emo>/level_<L>/<utt>.wav` (`TRACE_MEAD_RAW`) |
| `data/raw/ESD` | raw ESD audio (for the ESD inter-emotion flow) |
| `data/features/speaker_splits.json` | `{"val":[…],"test":[…]}` speaker split (`TRACE_MEAD_SPLIT`) |
| `<pretrained>/km200.bin` | HuBERT k-means (200) model (`TRACE_HUBERT_KM200`) |
| `data/prompts/*.jsonl` | relative-instruction prompts from **TRACE-Instruct** (external repo) |

Frozen models (HuBERT, ECAPA, E5, emotion2vec, Odyssey SER) are auto-downloaded
from the Hugging Face Hub on first use.

**Offline / air-gapped nodes:** the training scripts set `HF_HUB_OFFLINE=1`. For the
E5 text encoder and the Odyssey SER model, drop a local snapshot under `pretrained/`
(`pretrained/e5-base-v2`, `pretrained/odyssey_ser`) — `paths.py` prefers a local dir
automatically, which loads reliably offline (a bare Hub id can mis-resolve its commit
under `HF_HUB_OFFLINE` even when fully cached). Grab a snapshot with e.g.
`huggingface-cli download intfloat/e5-base-v2 --local-dir pretrained/e5-base-v2`.

## Build it (from raw audio)

```bash
export PYTHONPATH=.
bash scripts/prepare_data.sh        # runs every step below, in order
```

Each generated artifact and its producer:

| Artifact | Produced by |
|---|---|
| `features/processed/units/<spk>/<item>.pt` — HuBERT km200 units | `tasks/ease/extract_units_mead.py` |
| `speaker/cache/xvectors.pt` — ECAPA x-vectors | `tasks/ease/extract_xvectors_mead.py` |
| `features/binary[_44k]/{train,valid,test}` — mel+units binary | `tasks/ease/build_mead_binary.py` |
| `features/processed/f0_yaapt/<spk>/<item>.npy` — YAAPT F0 @50Hz | `tasks/preprocess/extract_f0.py` |
| `features/f0_spk_stats.npz`, `f0_utt_stats.npz` | ″ (aggregation pass) |
| `features/processed/energy_rms[_raw]/<spk>/<item>.npy` — RMS @50Hz | `tasks/preprocess/extract_energy.py [--norm/--raw]` |
| `features/energy[_raw]_spk_stats.npz`, `energy[_raw]_utt_stats.npz` | ″ (aggregation pass) |
| `vad/vad.jsonl` — Odyssey SER arousal/dominance/valence | `tasks/preprocess/extract_vad.py` |
| `prompts/*_prompt_emb_e5.pt` — E5 prompt embeddings | `tasks/emo_compass/encode_prompts.py` |

> **Ordering:** F0/energy read `wav_fn` from the binary, so `build_mead_binary.py`
> must run first. F0 and energy are extracted on the SAME `trim_long_silences()`
> output the binarizer used, so the contours stay frame-aligned to the mel/units.

## Produced by the training stages (not by prepare_data.sh)

These are outputs of `scripts/train_ease.sh` / `train_emo_compass.sh` — listed here
because the synthesis stage reads them:

```
speaker/EASE_embeddings/<item>.npy   per-utterance EASE speaker embeddings (train_mead_ease.py)
speaker/ease_stats.npz               EASE mean/std
features/diffused/                    Emo-Compass target-embedding bank (precompute_targets.py)
```
