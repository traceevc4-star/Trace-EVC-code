"""Data loading and affect-vector helpers shared by the Emo-Compass trainers."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

EMOTIONS_ESD = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
EMOTIONS_MEAD = ["angry", "happy", "sad", "surprised"]
EMO_ID_MEAD = {e: i for i, e in enumerate(EMOTIONS_MEAD)}


def vad_to_tensor(vad) -> torch.Tensor:
    """Accept a {arousal,dominance,valence} dict or a 3-list; return an ordered tensor."""
    if isinstance(vad, dict):
        return torch.tensor([vad["arousal"], vad["dominance"], vad["valence"]], dtype=torch.float32)
    if isinstance(vad, (list, tuple)) and len(vad) == 3:
        return torch.tensor(list(vad), dtype=torch.float32)
    raise ValueError(f"Unsupported VAD value: {vad}")


def affect_vector(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(z_src, z_tgt) = [emotion2vec | VAD | prosody]. The prosody tail [zF0, zEnergy]
    is appended only when the record carries it (MEAD); ESD stays [emotion2vec | VAD]."""
    parts_src = [row["source_emo"], row["source_vad"]]
    parts_tgt = [row["target_emo"], row["target_vad"]]
    if "source_prosody" in row:
        parts_src.append(row["source_prosody"])
        parts_tgt.append(row["target_prosody"])
    return torch.cat(parts_src, dim=0), torch.cat(parts_tgt, dim=0)


def compute_affect_stats(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    vals = []
    for row in records:
        z_src, z_tgt = affect_vector(row)
        vals.extend([z_src, z_tgt])
    z = torch.stack(vals)
    return z.mean(0), z.std(0).clamp_min(1e-6)


_KEY = re.compile(
    r"/(?P<spk>M\d+|W\d+)/.*?/(?P<emo>angry|happy|sad|surprised|neutral)/"
    r"level[_ ]?(?P<lv>[123])/(?P<utt>\d+)\.(?:wav|m4a)"
)


def parse_key(path: str):
    """MEAD wav path -> (speaker, emotion, level:int, content) or None."""
    m = _KEY.search(path)
    if not m:
        return None
    return (m.group("spk"), m.group("emo"), int(m.group("lv")), m.group("utt"))


def load_esd_records(prompts_jsonl: Path, prompt_emb_path: Path, emo_cache_path: Path, limit: int | None):
    """ESD pairs: emotion2vec cache keyed by exact wav path; 5-way categorical labels."""
    prompt_cache = torch.load(prompt_emb_path, map_location="cpu", weights_only=False)
    pair_emb = prompt_cache["pair_emb"]
    emo_cache = torch.load(emo_cache_path, map_location="cpu", weights_only=False)
    label_to_id = {e: i for i, e in enumerate(EMOTIONS_ESD)}
    records, missing = [], Counter()
    with Path(prompts_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["pair_id"]
            if pid not in pair_emb:
                missing["prompt_emb"] += 1
                continue
            if rec["source_wav"] not in emo_cache:
                missing["source_emo2vec"] += 1
                continue
            if rec["target_wav"] not in emo_cache:
                missing["target_emo2vec"] += 1
                continue
            if "source_vad" not in rec or "target_vad" not in rec:
                missing["vad"] += 1
                continue
            records.append({
                "pair_id": pid,
                "speaker_id": rec["speaker_id"],
                "text_idx": rec["text_idx"],
                "source_emotion": rec["source_emotion"],
                "target_emotion": rec["target_emotion"],
                "target_label": label_to_id[rec["target_emotion"]],
                "source_emo": emo_cache[rec["source_wav"]].float(),
                "target_emo": emo_cache[rec["target_wav"]].float(),
                "source_vad": vad_to_tensor(rec["source_vad"]),
                "target_vad": vad_to_tensor(rec["target_vad"]),
                "prompt_emb": pair_emb[pid].float(),
            })
            if limit is not None and len(records) >= limit:
                break
    print(f"ESD records loaded: {len(records)} missing={dict(missing)}")
    if not records:
        raise RuntimeError("No usable ESD records loaded.")
    return records, int(prompt_cache["dim"])


def _item_name_from_key(k) -> str:
    """(spk, emo, level:int, utt) -> 'SPK_emo_Llv_utt' (matches the *_utt_stats keys)."""
    spk, emo, lv, utt = k
    return f"{spk}_{emo}_L{lv}_{utt}"


def load_mead_prosody(f0_stats_path, energy_stats_path) -> dict:
    """item_name -> [z-mean F0, z-mean energy] from the utt-level prosody stats."""
    import numpy as np
    f0 = np.load(str(f0_stats_path), allow_pickle=True)
    en = np.load(str(energy_stats_path), allow_pickle=True)
    f0z = {str(n): float(s[0]) for n, s in zip(f0["names"], f0["stats"])}
    enz = {str(n): float(z) for n, z in zip(en["names"], en["z_mean"])}
    keys = set(f0z) | set(enz)
    return {k: torch.tensor([f0z.get(k, 0.0), enz.get(k, 0.0)], dtype=torch.float32) for k in keys}


def load_mead_records(prompts_jsonl: Path, prompt_emb_path: Path, emo_cache_path: Path, limit: int | None,
                      prosody_index: dict | None = None):
    """MEAD intra pairs joined by (spk,emo,level,content) key. When prosody_index is
    given, records also carry source/target mean prosody so z becomes [e; a; r]."""
    prompt_cache = torch.load(prompt_emb_path, map_location="cpu", weights_only=False)
    pair_emb = prompt_cache["pair_emb"]
    emo_cache = torch.load(emo_cache_path, map_location="cpu", weights_only=False)
    key_index = {}
    for path, emb in emo_cache.items():
        k = parse_key(path)
        if k is not None:
            key_index[k] = emb.float()

    _zero_pros = torch.zeros(2, dtype=torch.float32)
    records, missing = [], Counter()
    with Path(prompts_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["pair_id"]
            if pid not in pair_emb:
                missing["prompt_emb"] += 1
                continue
            sk, tk = parse_key(r["source_wav"]), parse_key(r["target_wav"])
            if sk not in key_index:
                missing["src_emo2vec"] += 1
                continue
            if tk not in key_index:
                missing["tgt_emo2vec"] += 1
                continue
            emo = r["emotion"]
            rec = {
                "pair_id": pid,
                "speaker_id": r["speaker_id"],
                "emotion": emo,
                "emo_id": EMO_ID_MEAD[emo],
                "target_label": EMO_ID_MEAD[r["target_emotion"]],
                "source_level": int(r["source_level"]),
                "target_level": int(r["target_level"]),
                "source_emo": key_index[sk],
                "target_emo": key_index[tk],
                "source_vad": vad_to_tensor(r["source_vad"]),
                "target_vad": vad_to_tensor(r["target_vad"]),
                "prompt_emb": pair_emb[pid].float(),
            }
            if prosody_index is not None:
                rec["source_prosody"] = prosody_index.get(_item_name_from_key(sk), _zero_pros)
                rec["target_prosody"] = prosody_index.get(_item_name_from_key(tk), _zero_pros)
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    print(f"MEAD records loaded: {len(records)} missing={dict(missing)}")
    if not records:
        raise RuntimeError("No usable MEAD records loaded.")
    return records, int(prompt_cache["dim"])


def load_esd_index_split(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(rec["index"])] = rec["split"]
    missing = sorted(set(range(1, 351)).difference(out))
    if missing:
        raise ValueError(f"Index split missing {len(missing)} indices, first few: {missing[:10]}")
    return out


def split_esd_by_index(records: list[dict], index_split: dict[int, str], split: str) -> list[dict]:
    return [r for r in records if index_split[int(r["text_idx"]) + 1] == split]


def split_by_speaker(records: list[dict], speakers) -> list[dict]:
    keep = set(speakers)
    return [r for r in records if r["speaker_id"] in keep]


def load_mead_split(path: Path):
    """Return (train, val, test) MEAD speaker lists from the standard split json."""
    d = json.loads(Path(path).read_text())
    return list(d["train"]), list(d["val"]), list(d["test"])


class TracePromptDataset(Dataset):
    """Standardized (z_src, z_tgt) pairs with one sampled prompt variant per item."""

    def __init__(self, records: list[dict], mean: torch.Tensor, std: torch.Tensor, split: str):
        self.records = records
        self.mean = mean
        self.std = std
        self.split = split
        self.rng = random.Random(1234)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        z_src, z_tgt = affect_vector(row)
        z_src = (z_src - self.mean) / self.std
        z_tgt = (z_tgt - self.mean) / self.std
        bank = row["prompt_emb"]
        j = self.rng.randrange(bank.shape[0]) if self.split == "train" else 0
        return {
            "z_src": z_src,
            "z_tgt": z_tgt,
            "prompt_emb": bank[j],
            "emo_id": torch.tensor(row.get("emo_id", 0), dtype=torch.long),
            "target_label": torch.tensor(row["target_label"], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "z_src": torch.stack([b["z_src"] for b in batch]),
        "z_tgt": torch.stack([b["z_tgt"] for b in batch]),
        "prompt_emb": torch.stack([b["prompt_emb"] for b in batch]),
        "emo_id": torch.stack([b["emo_id"] for b in batch]),
        "target_label": torch.stack([b["target_label"] for b in batch]),
    }


def load_mead_vad_index(path: Path) -> dict:
    """(spk,emo,level,content) -> VAD tensor [arousal, dominance, valence]."""
    idx = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = parse_key(r["audio_path"])
            if k is not None:
                idx[k] = torch.tensor([r["arousal"], r["dominance"], r["valence"]], dtype=torch.float32)
    return idx


def attach_vad(rows: list[dict], vad_path: Path) -> list[dict]:
    """Add row['vad'] (3-d) to each row; drop rows without a VAD entry."""
    vidx = load_mead_vad_index(vad_path)
    out = []
    for r in rows:
        k = (r["speaker"], r["emotion"], r["level"], r["content"])
        if k in vidx:
            r["vad"] = vidx[k]
            out.append(r)
    return out


def load_mead_emo2vec_rows(cache_path: Path, emotions=EMOTIONS_MEAD) -> list[dict]:
    """Rows of {emb[1024], emotion, emo_id, level(1..3), speaker, content}."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    keep = set(emotions)
    rows = []
    for path, emb in cache.items():
        k = parse_key(path)
        if k is None:
            continue
        spk, emo, lv, utt = k
        if emo not in keep:
            continue
        rows.append({
            "emb": emb.float(), "emotion": emo, "emo_id": EMO_ID_MEAD[emo],
            "level": lv, "speaker": spk, "content": utt,
        })
    return rows


def speaker_split_rows(rows: list[dict], val_speakers, test_speakers):
    val_s, test_s = set(val_speakers), set(test_speakers)
    tr = [r for r in rows if r["speaker"] not in val_s and r["speaker"] not in test_s]
    va = [r for r in rows if r["speaker"] in val_s]
    te = [r for r in rows if r["speaker"] in test_s]
    return tr, va, te


def vad_stats(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.stack([r["vad"] for r in rows])
    return v.mean(0), v.std(0).clamp_min(1e-6)


def emo_stats(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([r["emb"] for r in rows])
    return x.mean(0), x.std(0).clamp_min(1e-6)
