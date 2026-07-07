#!/usr/bin/env python3
"""Odyssey WavLM SER arousal/dominance/valence for raw MEAD -> data/vad/vad.jsonl.

Same model and preprocessing as infer_end2end.extract_vad. Walks the MEAD tree for
<spk>/.../<emo>/level_<L>/<utt>.wav and writes one JSONL row per utterance with flat
arousal/dominance/valence fields (read by data.load_mead_vad_index).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

import paths

EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]
_KEY = re.compile(
    r"/(?P<spk>M\d+|W\d+)/.*?/(?P<emo>angry|happy|sad|surprised|neutral)/"
    r"level[_ ]?(?P<lv>[123])/(?P<utt>\d+)\.(?:wav|m4a)"
)


def load_model(model_id: str, device: str):
    from transformers import AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(
        model_id, trust_remote_code=True).to(device).eval()
    return model


@torch.no_grad()
def extract_vad(model, wav_path: str, device: str) -> np.ndarray:
    """Return [arousal, dominance, valence] (the model's first 3 outputs)."""
    import librosa

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    mean, std = float(model.config.mean), float(model.config.std)
    audio = (audio - mean) / (std + 1e-6)
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones_like(wav)
    pred = model(wav, mask)
    pred = (pred.logits if hasattr(pred, "logits") else pred).detach().cpu().float().view(-1)
    return pred[:3].numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default=str(paths.MEAD_RAW_AUDIO), help="raw MEAD audio root")
    p.add_argument("--out-jsonl", default=str(paths.MEAD_VAD))
    p.add_argument("--model", default=str(paths.ODYSSEY_VAD_MODEL))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from pathlib import Path

    raw = Path(args.raw_dir)
    wavs = []
    for emo in EMOTIONS:
        wavs += sorted(raw.rglob(f"*/{emo}/level_*/*.wav"))
        wavs += sorted(raw.rglob(f"*/{emo}/level_*/*.m4a"))
    wavs = [w for w in wavs if _KEY.search(str(w))]
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        raise SystemExit(f"no MEAD wavs found under {raw} (expected <spk>/.../<emo>/level_<L>/<utt>.wav)")
    print(f"found {len(wavs)} utterances; loading SER model {args.model} on {args.device}", flush=True)

    model = load_model(args.model, args.device)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = out_path.with_name(out_path.stem + "_failures.jsonl")
    n_ok = n_fail = 0
    with out_path.open("w", encoding="utf-8") as fout, fail_path.open("w", encoding="utf-8") as ferr:
        for i, w in enumerate(wavs):
            m = _KEY.search(str(w))
            try:
                a, d, v = (float(x) for x in extract_vad(model, str(w), args.device))
            except Exception as e:
                n_fail += 1
                ferr.write(json.dumps({"audio_path": str(w), "error": str(e)}) + "\n")
                continue
            fout.write(json.dumps({
                "audio_path": str(w),
                "speaker_id": m.group("spk"),
                "emotion": m.group("emo"),
                "intensity": f"level_{m.group('lv')}",
                "utterance_id": m.group("utt"),
                "arousal": a, "dominance": d, "valence": v,
                "model_name_or_path": args.model,
            }) + "\n")
            n_ok += 1
            if (i + 1) % 1000 == 0:
                print(f"progress {i+1}/{len(wavs)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"wrote {n_ok} rows -> {out_path}  ({n_fail} failures -> {fail_path})", flush=True)
    print("VAD_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
