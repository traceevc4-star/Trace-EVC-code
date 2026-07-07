import os
import random
import time
import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset


def _load_diff_bank(path, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except RuntimeError as e:
            last_err = e
            if attempt + 1 == retries:
                break
            time.sleep(2.0 * (attempt + 1))
    raise last_err


def _level_id_from_item(item_name):
    """Return zero-based MEAD level id from names like M003_angry_L2_002."""
    for part in str(item_name).split("_"):
        if part.startswith("L") and part[1:].isdigit():
            return int(part[1:]) - 1
    raise ValueError(f"Cannot parse MEAD level from item_name={item_name!r}")


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.train = train
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [
                    x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]
                ]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None or getattr(self, "_indexed_ds_pid", None) != os.getpid():
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
            self._indexed_ds_pid = os.getpid()
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (
            len(item["mel"]),
            self.sizes[index],
        )
        wav_fn = item["wav_fn"]
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]

        sample = {
            "id": index,
            "wav_fn": wav_fn,
            "item_name": item["item_name"],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "spk_id": int(item["spk_id"]),
        }
        return sample

    def collater(self, samples):

        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        wav_fns = [s["wav_fn"] for s in samples]
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])
        spk_ids = torch.LongTensor([s["spk_id"] for s in samples])

        batch = {
            "id": id,
            "wav_fn": wav_fns,
            "item_name": item_names,
            "nsamples": len(samples),
            "mels": mels,
            "mel_lengths": mel_lengths,
            "spk_ids": spk_ids,
        }

        return batch


class DurFlexDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(prefix, shuffle, items, data_dir, train)
        emo_names = self.hparams.get(
            "emo_names", ["angry", "happy", "sad", "surprised"]
        )
        self.emo_names = [str(e) for e in emo_names]
        self.emo_dict = {e: i for i, e in enumerate(self.emo_names)}
        self.use_emb_cond = self.hparams.get("use_emb_cond", False)
        self._diff = None
        self.use_text_cond = self.hparams.get("use_text_cond", False)
        self._text = None
        self._text_mean = None
        self._cross_ratio_val = None
        self._cross_calls = 0
        self._unit_cache = {}
        self._ease_cache = {}
        self.use_f0_cond = self.hparams.get("use_f0_cond", False)
        self._f0_cache = {}
        if self.use_f0_cond:
            _fs = np.load(self.hparams["f0_spk_stats_path"], allow_pickle=True)
            self._f0_spk_mean = {s: float(m) for s, m in zip(_fs["speakers"], _fs["mean"])}
            self._f0_spk_std = {s: float(v) for s, v in zip(_fs["speakers"], _fs["std"])}
        self.use_energy_frame_cond = self.hparams.get("use_energy_frame_cond", False)
        self._energy_cache = {}
        if self.use_energy_frame_cond:
            _es2 = np.load(self.hparams["energy_spk_stats_path"], allow_pickle=True)
            self._en_spk_mean = {s: float(m) for s, m in zip(_es2["speakers"], _es2["mean"])}
            self._en_spk_std = {s: float(v) for s, v in zip(_es2["speakers"], _es2["std"])}
        self.use_mean_prosody = self.hparams.get("use_mean_prosody", False)
        if self.use_mean_prosody:
            _f = np.load(self.hparams["f0_utt_stats_path"], allow_pickle=True)
            self._f0_zmean = {n: float(s[0]) for n, s in zip(_f["names"], _f["stats"])}
            _e = np.load(self.hparams["energy_utt_stats_path"], allow_pickle=True)
            self._en_zmean = {n: float(z) for n, z in zip(_e["names"], _e["z_mean"])}
            _ers = np.load(self.hparams["energy_raw_spk_stats_path"], allow_pickle=True)
            self._en_raw_std = {s: float(v) for s, v in zip(_ers["speakers"], _ers["std"])}
            self._pred_mp = None
            _pp = self.hparams.get("mean_prosody_pred_path", "")
            if _pp:
                _d = np.load(_pp, allow_pickle=True)
                self._pred_mp = {str(n): (float(p[0]), float(p[1])) for n, p in zip(_d["names"], _d["pred"])}
                print(f"| mean_prosody source = PREDICTED ({len(self._pred_mp)} items)")
        self._filter_items()
        if self.hparams.get("preload_content", True):
            self._preload_content()
            if self.use_emb_cond:
                _ = self.diff

    def _filter_items(self):
        """Keep only MEAD items whose emotion and affect vectors are trainable.

        The MEAD binary contains neutral and some utterances without a complete
        L1/L2/L3 intensity set. TRACE precompute only emits real_by_item for
        utterances covered by the intra-emotion pair bank, so filter them once
        here and keep the large bank out of the pickled Dataset object.
        """
        if not self.hparams.get("filter_unpaired_items", True):
            return

        allowed = set(self.emo_dict)
        real_items = None
        if self.use_emb_cond:
            meta_path = self.hparams.get("diffused_item_names_path", "")
            if meta_path and os.path.exists(meta_path):
                with open(meta_path) as f:
                    real_items = {line.strip() for line in f if line.strip()}
            else:
                d = _load_diff_bank(self.hparams["diffused_emb_path"])
                real_items = set(d["real_by_item"])
                del d

        ds = self.indexed_ds
        close_ds = False
        if ds is None:
            ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
            close_ds = True

        spk_subset = self.hparams.get("spk_subset", "")
        spk_subset = {s.strip() for s in str(spk_subset).split(",") if s.strip()}

        keep_idxs, keep_sizes = [], []
        skipped_emo, skipped_bank = {}, 0
        for raw_idx, size in zip(self.avail_idxs, self.sizes):
            item = ds[raw_idx]
            emo = str(item.get("emo", ""))
            item_name = item["item_name"]
            if spk_subset and item_name.split("_")[0] not in spk_subset:
                continue
            _cs = int(self.hparams.get("content_split_utt", 0))
            if _cs:
                _utt = int(item_name.split("_")[3])
                if (_utt > _cs) != bool(self.hparams.get("content_split_eval", False)):
                    continue
            if emo not in allowed:
                skipped_emo[emo] = skipped_emo.get(emo, 0) + 1
                continue
            if real_items is not None and item_name not in real_items:
                skipped_bank += 1
                continue
            keep_idxs.append(raw_idx)
            keep_sizes.append(size)

        if close_ds:
            ds.data_file.close()

        before = len(self.sizes)
        self.avail_idxs = keep_idxs
        self.sizes = keep_sizes
        print(
            f"| {self.prefix}: kept {len(self.sizes)}/{before} MEAD EVC items "
            f"(skip_emo={skipped_emo}, skip_unpaired={skipped_bank})",
            flush=True,
        )

    @property
    def diff(self):
        if self._diff is None:
            self._diff = _load_diff_bank(self.hparams["diffused_emb_path"])
        return self._diff

    @property
    def text(self):
        if self._text is None:
            bank = torch.load(self.hparams["text_emb_path"], map_location="cpu", weights_only=False)
            self._text = {pid: e.float() for pid, e in bank["pair_emb"].items()}
            stacked = torch.stack([e[0] for e in self._text.values()])
            self._text_mean = stacked.mean(0)
        return self._text

    def _cross_ratio(self):
        """Current cross-level probability, read from the task-written sentinel.
        0 until val unit_acc crosses the curriculum threshold; ramps up after."""
        if not self.hparams.get("cross_curriculum", False):
            return 0.0
        self._cross_calls += 1
        if self._cross_ratio_val is None or self._cross_calls % 64 == 0:
            try:
                with open(os.path.join(self.hparams["work_dir"], "cross_curriculum.txt")) as f:
                    self._cross_ratio_val = float(f.read().strip() or 0.0)
            except Exception:
                self._cross_ratio_val = 0.0
        return self._cross_ratio_val

    def _pick_conversion(self, item_name, index, do_cross):
        """Return (content_item, src_vec, tgt_vec) for an intensity edit (option B).

        tgt = a diffused source->item embedding (restyle). For SELF-recon content and
        destyle use the item itself. For CROSS-level content comes from the lower-level
        SOURCE utterance and destyle uses the source's real affect, while the target mel
        stays the item (e.g. L1 content -> reconstruct L3 mel) -> breaks energy leakage.
        Train: random pair/prompt. Val/test: deterministic (seeded by index)."""
        d = self.diff
        pin = self.hparams.get("pin_pid_map", None)
        if pin and item_name in pin:
            _pid = pin[item_name]; _pr = d["pairs"][_pid]; _src = _pr["source_item"]
            return _src, d["real_by_item"][_src], _pr["emb"][0]
        pids = d["by_target_item"].get(item_name, [])
        if self.use_text_cond:
            _ = self.text
            if not pids:
                v = self._text_mean
                return item_name, v, v.clone()
            if self.train:
                pid = random.choice(pids)
                te = self.text[pid]
                tgt_vec = te[random.randrange(te.shape[0])]
            else:
                rng = random.Random(index)
                pid = pids[rng.randrange(len(pids))]
                tgt_vec = self.text[pid][0]
            content = d["pairs"][pid]["source_item"] if do_cross else item_name
            return content, tgt_vec.clone(), tgt_vec
        if not pids:
            v = d["real_by_item"][item_name]
            return item_name, v, v.clone()
        if self.train:
            pid = random.choice(pids)
            emb = d["pairs"][pid]["emb"]
            tgt_vec = emb[random.randrange(emb.shape[0])]
        else:
            rng = random.Random(index)
            pid = pids[rng.randrange(len(pids))]
            tgt_vec = d["pairs"][pid]["emb"][0]
        if do_cross:
            source_item = d["pairs"][pid]["source_item"]
            return source_item, d["real_by_item"][source_item], tgt_vec
        return item_name, d["real_by_item"][item_name], tgt_vec

    def _unit_from_disk(self, content_item):
        spk = content_item.split("_")[0]
        return torch.load(
            os.path.join(self.hparams["processed_data_dir"], "units", spk, content_item + ".pt")
        )

    def _load_unit_content(self, content_item):
        c = self._unit_cache.get(content_item)
        if c is None:
            c = self._unit_from_disk(content_item)
            self._unit_cache[content_item] = c
        return c

    def _load_ease(self, content_item):
        c = self._ease_cache.get(content_item)
        if c is None:
            c = (
                np.load(os.path.join(self.hparams["ease_dir"], content_item + ".npy"))
                .astype(np.float32)
                .ravel()
            )
            self._ease_cache[content_item] = c
        return c

    def _load_f0_cond(self, item_name, n_mel_full, n_mel_keep):
        """GT YAAPT contour -> (z-log-F0, uv) resampled to THIS item's mel frames.

        Per-sample resampling (50Hz -> hop256) keeps the collated batch exactly
        mel-aligned; a batch-level interpolate in the model would blur shorter
        samples. Resample to the FULL mel length first, then apply the same
        truncation the mel got (max_frames/frames_multiple)."""
        c = self._contour(item_name, self._f0_cache, self.hparams["f0_dir"],
                          self._f0_spk_mean, self._f0_spk_std)
        T = c.shape[0]
        pos = np.linspace(0, T - 1, n_mel_full)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (pos - i0)[:, None].astype(np.float32)
        out = c[i0] * (1 - w) + c[i1] * w
        return torch.from_numpy(out[:n_mel_keep])

    def _contour(self, item_name, cache, cdir, spk_mean, spk_std):
        """Shared (value, mask) contour loader: positive=active, 0=masked,
        log-domain per-speaker z. Works for F0 (Hz) and energy (linear RMS)."""
        c = cache.get(item_name)
        if c is None:
            spk = item_name.split("_")[0]
            raw = np.load(os.path.join(cdir, spk, item_name + ".npy"))
            v = raw > 0
            z = np.zeros_like(raw, dtype=np.float32)
            if v.any():
                z[v] = (np.log(raw[v]) - spk_mean[spk]) / (spk_std[spk] + 1e-8)
            c = np.stack([z, v.astype(np.float32)], -1)
            cache[item_name] = c
        return c

    def _load_energy_cond(self, item_name, n_mel_full, n_mel_keep):
        c = self._contour(item_name, self._energy_cache, self.hparams["energy_dir"],
                          self._en_spk_mean, self._en_spk_std)
        T = c.shape[0]
        pos = np.linspace(0, T - 1, n_mel_full)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (pos - i0)[:, None].astype(np.float32)
        out = c[i0] * (1 - w) + c[i1] * w
        return torch.from_numpy(out[:n_mel_keep])

    def _preload_content(self):
        """Read every unit/ease file this split will use into RAM up-front so the
        training loop never touches the Weka disk per-sample. ~17 GB units total /
        946 GB RAM, single read shared via OS page cache across DDP ranks."""
        import time
        t0 = time.time()
        for i in range(len(self.avail_idxs)):
            name = self._get_item(i)["item_name"]
            try:
                self._unit_cache[name] = self._unit_from_disk(name)
            except Exception as e:
                print(f"| preload skip units {name}: {e}", flush=True)
            try:
                self._load_ease(name)
            except Exception as e:
                print(f"| preload skip ease {name}: {e}", flush=True)
        print(
            f"| {self.prefix}: preloaded {len(self._unit_cache)} units / "
            f"{len(self._ease_cache)} ease into RAM in {time.time()-t0:.0f}s",
            flush=True,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        item_name = item["item_name"]

        content_item = item_name
        src_vec = tgt_vec = None
        if self.hparams.get("use_energy_cond", False):
            if self.use_emb_cond:
                v = self.diff["real_by_item"][item_name].float()
                src_vec = tgt_vec = v
            if self.train and float(self.hparams.get("energy_gain", 0.0)) > 0:
                c = (random.random() * 2.0 - 1.0) * float(self.hparams["energy_gain"])
                sample["mel"] = sample["mel"] + c
            sample["energy_target"] = float(sample["mel"].mean())
        elif self.use_emb_cond:
            do_cross = self.hparams.get("force_cross_eval", False) or (self.train and (random.random() < self._cross_ratio()))
            content_item, src_vec, tgt_vec = self._pick_conversion(item_name, index, do_cross)

        u = self._load_unit_content(content_item)
        sample["unit"] = torch.IntTensor(u["units"])
        sample["unit_frame"] = torch.IntTensor(u["units_frame"])
        sample["hubert_feature"] = torch.FloatTensor(u["features"])
        mel2unit = u["mel2unit"]
        sample["unit_l"] = mel2unit[-1]
        sample["dur_unit"] = torch.IntTensor(u["count"])
        sample["mel2unit"] = mel2unit
        sample["spk_id"] = int(item["spk_id"])
        sample["emotion_id"] = int(self.emo_dict[item["emo"]])
        if self.hparams.get("use_int_disentangle", False):
            _lv = int(item_name.split("_")[2][1:])
            sample["intensity_label"] = float((_lv - 1) / 2.0)
        if self.use_mean_prosody:
            f0_gt = self._f0_zmean.get(item_name, 0.0)
            en_gt = self._en_zmean.get(item_name, 0.0)
            if self._pred_mp is not None and item_name in self._pred_mp:
                f0m, enm = self._pred_mp[item_name]
            else:
                f0m, enm = f0_gt, en_gt
            g = float(self.hparams.get("energy_gain_db", 0.0))
            if self.train and g > 0:
                c_db = (random.random() * 2.0 - 1.0) * g
                sample["mel"] = sample["mel"] + c_db * float(np.log(10.0) / 20.0)
                spk = item_name.split("_")[0]
                _sc = c_db * (np.log(10.0) / 20.0) / (self._en_raw_std.get(spk, 1.0) + 1e-8)
                enm = enm + _sc; en_gt = en_gt + _sc
            sample["mean_prosody"] = torch.tensor([f0m, enm], dtype=torch.float32)
            sample["mean_prosody_gt"] = torch.tensor([f0_gt, en_gt], dtype=torch.float32)
        sample["ease"] = torch.from_numpy(self._load_ease(content_item).copy())
        if self.use_f0_cond:
            sample["f0_cond"] = self._load_f0_cond(
                item_name, int(self.sizes[index]), sample["mel"].shape[0]
            )
        if self.use_energy_frame_cond:
            sample["energy_cond"] = self._load_energy_cond(
                item_name, int(self.sizes[index]), sample["mel"].shape[0]
            )
            g = float(self.hparams.get("energy_gain_db", 0.0))
            if self.train and g > 0:
                c_db = (random.random() * 2.0 - 1.0) * g
                sample["mel"] = sample["mel"] + c_db * float(np.log(10.0) / 20.0)
                spk = item_name.split("_")[0]
                vm = sample["energy_cond"][:, 1] > 0
                sample["energy_cond"][vm, 0] += c_db * (np.log(10.0) / 20.0) / (self._en_spk_std[spk] + 1e-8)
        if self.use_emb_cond:
            sample["src_emo_vec"] = src_vec.float()
            sample["tgt_emo_vec"] = tgt_vec.float()
        return sample

    def collater(self, samples):
        batch = super().collater(samples)

        units = collate_1d_or_2d([s["unit"] for s in samples], 0.0)
        unit_frames = collate_1d_or_2d([s["unit_frame"] for s in samples], 0.0)
        dur_unit = collate_1d_or_2d([s["dur_unit"] for s in samples], 0.0)
        unit_l = torch.LongTensor([s["unit_l"] for s in samples])
        mel2unit = collate_1d_or_2d([s["mel2unit"] for s in samples], 0.0)
        hubert_features = collate_1d_or_2d([s["hubert_feature"] for s in samples], 0.0)
        batch["units"] = units
        batch["unit_frames"] = unit_frames
        batch["unit_l"] = unit_l
        batch["dur_unit"] = dur_unit
        batch["mel2unit"] = mel2unit
        batch["hubert_features"] = hubert_features
        hubert_lengths = torch.LongTensor(
            [s["hubert_feature"].shape[0] for s in samples]
        )
        batch["hubert_lengths"] = hubert_lengths

        spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
        batch["spk_ids"] = spk_ids
        emos = torch.LongTensor([s["emotion_id"] for s in samples])
        batch["emotion_ids"] = emos
        batch["ease"] = torch.stack([s["ease"] for s in samples])
        if self.use_emb_cond and "src_emo_vec" in samples[0]:
            batch["src_emo_vec"] = torch.stack([s["src_emo_vec"] for s in samples])
            batch["tgt_emo_vec"] = torch.stack([s["tgt_emo_vec"] for s in samples])
        if "energy_target" in samples[0]:
            batch["energy_target"] = torch.FloatTensor([s["energy_target"] for s in samples])
        if "f0_cond" in samples[0]:
            batch["f0_cond"] = collate_1d_or_2d([s["f0_cond"] for s in samples], 0.0)
        if "energy_cond" in samples[0]:
            batch["energy_cond"] = collate_1d_or_2d([s["energy_cond"] for s in samples], 0.0)
        if "intensity_label" in samples[0]:
            batch["intensity_label"] = torch.FloatTensor([s["intensity_label"] for s in samples])
        if "mean_prosody" in samples[0]:
            batch["mean_prosody"] = torch.stack([s["mean_prosody"] for s in samples])
        if "mean_prosody_gt" in samples[0]:
            batch["mean_prosody_gt"] = torch.stack([s["mean_prosody_gt"] for s in samples])
        return batch
