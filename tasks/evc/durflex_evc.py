import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.evc.durflex import DurFlexEVC
from tasks.evc.speech_base import SpeechBaseTask
from tasks.evc.evc_utils import VocoderInfer
from tasks.evc.sb_vocoder import build_vocoder
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars, move_to_cuda
from utils.nn.model_utils import print_arch, num_params

DEFAULT_EMO_NAMES = ["angry", "happy", "sad", "surprised"]
DEFAULT_TARGET_LEVELS = [1, 2, 3]


def emo_names():
    return list(hparams.get("emo_names", DEFAULT_EMO_NAMES))


def target_levels():
    return [int(x) for x in hparams.get("target_levels", DEFAULT_TARGET_LEVELS)]


def level_id_from_item(item_name):
    for part in str(item_name).split("_"):
        if part.startswith("L") and part[1:].isdigit():
            return int(part[1:]) - 1
    raise ValueError(f"Cannot parse MEAD level from item_name={item_name!r}")


class DurFlexEVCTask(SpeechBaseTask):
    def __init__(self):
        super(DurFlexEVCTask, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self._wandb_voc = None
        self._wandb_samples = None
        self._gt_logged = False
        self.use_emb_cond = hparams.get("use_emb_cond", False)
        self._diff = None

    def _diffbank(self):
        if self._diff is None:
            self._diff = torch.load(
                hparams["diffused_emb_path"], map_location="cpu", weights_only=False
            )
        return self._diff

    def _conv_vecs(self, item_names, tgt_level, device):
        """Per-sample (src real, tgt diffused) affect vecs for MEAD intensity edits.

        The MEAD bank is keyed by source item and target intensity level. Same-level
        or missing pairs fall back to the source's real affect, i.e. reconstruction.
        """
        d = self._diffbank()
        src, tgt = [], []
        for it in item_names:
            src_v = d["real_by_item"][it]
            pid = d["by_src_item_tgt"].get(f"{it}|||{int(tgt_level)}")
            tgt_v = d["pairs"][pid]["emb"][0] if pid is not None else src_v
            src.append(src_v.float())
            tgt.append(tgt_v.float())
        return torch.stack(src).to(device), torch.stack(tgt).to(device)

    def on_train_start(self):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        if hparams.get("use_wandb"):
            import wandb

            name = os.path.basename(hparams["work_dir"].rstrip("/"))
            wandb.init(
                project=hparams.get("wandb_project", "durflex-evc"),
                entity=hparams.get("wandb_entity") or None,
                name=name,
                dir=hparams["work_dir"],
                config={k: v for k, v in hparams.items()},
                resume="allow",
            )

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        interval = hparams.get("audio_log_interval", 1000)
        if (
            hparams.get("use_wandb")
            and interval > 0
            and (self.global_step + 1) % interval == 0
        ):
            try:
                self.log_audio_to_wandb()
            except Exception as e:
                import traceback

                print("| wandb audio logging failed (training continues):", e)
                traceback.print_exc()

    @torch.no_grad()
    def log_audio_to_wandb(self):
        """Per-epoch wandb audio: one validation utterance per speaker converted to
        one random target intensity level. Wandb is normally disabled on c015, but
        keep this path usable for offline runs.
        """
        import wandb

        if wandb.run is None:
            return
        if self._wandb_voc is None:
            self._wandb_voc = build_vocoder(self.hparams)
        if self._wandb_samples is None:
            by_spk = {}
            for b in self.val_dataloader():
                if b is None:
                    continue
                spk = b["item_name"][0].split("_")[0]
                if spk not in by_spk:
                    by_spk[spk] = b
            self._wandb_samples = [by_spk[s] for s in sorted(by_spk)]
            print(
                f"| wandb audio: cached {len(self._wandb_samples)} speakers' "
                f"source utts",
                flush=True,
            )

        sr = hparams["audio_sample_rate"]
        diff_steps = hparams.get("audio_diffusion_step", 100)
        was_training = self.model.training
        self.model.eval()
        logs = {}
        for batch in self._wandb_samples:
            batch = move_to_cuda(batch, 0)
            x = batch["hubert_features"]
            y = batch["mels"]
            y_lengths = batch["mel_lengths"]
            spk_id = batch.get("spk_ids")
            ease = batch.get("ease")
            src_emo = batch.get("emotion_ids")
            spk = batch["item_name"][0].split("_")[0]
            if not self._gt_logged:
                logs[f"audio/{spk}_source"] = wandb.Audio(
                    self._wandb_voc.spec2wav(y[0].cpu()), sample_rate=sr,
                    caption=batch["item_name"][0],
                )
            tgt_level = random.choice(target_levels())
            src_vec = tgt_vec = None
            if self.use_emb_cond:
                src_vec, tgt_vec = self._conv_vecs(batch["item_name"], tgt_level, x.device)
            try:
                out = self.model(
                    x,
                    spk_id=spk_id,
                    ease=ease,
                    emotion_id=src_emo,
                    tgt_emotion_id=src_emo,
                    src_emo_vec=src_vec,
                    tgt_emo_vec=tgt_vec,
                    infer=True,
                    y=y,
                    y_lengths=y_lengths,
                    diffusion_step=diff_steps,
                )
                wav = self._wandb_voc.spec2wav(out["mel_out"][0].cpu())
            except Exception as e:
                print(
                    f"| wandb audio skipped ({spk} -> L{tgt_level}) "
                    f"@step{self.global_step}: {e}",
                    flush=True,
                )
                continue
            logs[f"audio/{spk}_to_L{tgt_level}"] = wandb.Audio(
                wav, sample_rate=sr,
                caption=f"step{self.global_step}: {batch['item_name'][0]} -> L{tgt_level}",
            )
        if was_training:
            self.model.train()
        self._gt_logged = True
        wandb.log(logs, step=self.global_step)

    def build_model(self):
        self.model = DurFlexEVC(hparams["n_units"], hparams)
        load_ckpt = hparams.get("load_ckpt", "")
        if load_ckpt:
            ckpt = torch.load(load_ckpt, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"]["model"] if "state_dict" in ckpt else ckpt
            msd = self.model.state_dict()
            keep = {k: v for k, v in sd.items()
                    if k in msd and v.shape == msd[k].shape}
            skipped = [k for k in sd if k not in keep]
            self.model.load_state_dict(keep, strict=False)
            print(f"| warm-start from {load_ckpt}: loaded {len(keep)}/{len(sd)} "
                  f"tensors; skipped (shape mismatch / new) = {skipped}", flush=True)
        freeze = hparams.get("freeze_modules", []) or []
        if freeze:
            n_frozen = 0
            for name in freeze:
                m = getattr(self.model, name, None)
                if m is None:
                    print(f"| freeze_modules: '{name}' not found on model, skipped", flush=True)
                    continue
                if isinstance(m, nn.Parameter):
                    m.requires_grad_(False)
                    n_frozen += m.numel()
                else:
                    for p in m.parameters():
                        p.requires_grad_(False)
                        n_frozen += p.numel()
            print(f"| froze {len(freeze)} modules / {n_frozen/1e6:.2f}M params: {freeze}", flush=True)
        print_arch(self.model)
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        return self.model

    def forward(self, sample, infer=False, *args, **kwargs):
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        ease = sample.get("ease")

        emotion_id = sample.get("emotion_ids")
        x = sample["hubert_features"]

        y = sample["mels"]
        y_lengths = sample["mel_lengths"]

        src_emo_vec = sample.get("src_emo_vec")
        tgt_emo_vec = sample.get("tgt_emo_vec")
        energy_target = sample.get("energy_target")
        f0_cond = sample.get("f0_cond")
        energy_cond = sample.get("energy_cond")
        mean_prosody = sample.get("mean_prosody")

        if not infer:
            mel2unit = sample["mel2unit"]

            output = self.model(
                x,
                mel2unit=mel2unit,
                spk_embed=spk_embed,
                spk_id=spk_id,
                ease=ease,
                emotion_id=emotion_id,
                tgt_emotion_id=emotion_id,
                src_emo_vec=src_emo_vec,
                tgt_emo_vec=tgt_emo_vec,
                energy_target=energy_target,
                f0_cond=f0_cond,
                energy_cond=energy_cond,
                mean_prosody=mean_prosody,
                y=y,
                y_lengths=y_lengths,
                infer=False,
            )
            losses = {}

            self.add_ce_loss(
                output["unit_pred"],
                output["unit_logits"],
                sample["unit_frames"].float().unsqueeze(1),
                output["unit_nonpadding"],
                losses,
            )
            if "dur" in output:
                self.add_dur_loss(output["dur"], losses=losses)
            losses["diff_loss"] = output["diff_loss"]

            if "style_int_pred" in output and "intensity_label" in sample:
                tgt = sample["intensity_label"].float()
                lam_i = hparams.get("lambda_int", 1.0)
                losses["int_style"] = lam_i * F.mse_loss(output["style_int_pred"], tgt)
                with torch.no_grad():
                    losses["int_style_mae"] = (output["style_int_pred"] - tgt).abs().mean()
                if "content_int_pred" in output:
                    lam_a = hparams.get("lambda_adv", 1.0)
                    losses["int_adv"] = lam_a * F.mse_loss(output["content_int_pred"], tgt)
                    with torch.no_grad():
                        losses["int_content_mae"] = (output["content_int_pred"] - tgt).abs().mean()

            if "mp_pred" in output and "mean_prosody_gt" in sample:
                lam_mp = hparams.get("lambda_mp", 1.0)
                tgt_mp = sample["mean_prosody_gt"].float()
                losses["mp_sup"] = lam_mp * F.mse_loss(output["mp_pred"], tgt_mp)
                with torch.no_grad():
                    _d = (output["mp_pred"] - tgt_mp).abs().mean(0)
                    losses["mp_f0_mae"] = _d[0]
                    losses["mp_en_mae"] = _d[1]

            if hparams["use_spk_encoder"]:
                self.add_emo_loss(output["emo_logits"], emotion_id, losses)
            return losses, output
        else:
            mel2unit = None
            output = self.model(
                x,
                spk_embed=spk_embed,
                spk_id=spk_id,
                ease=ease,
                emotion_id=emotion_id,
                tgt_emotion_id=emotion_id,
                src_emo_vec=src_emo_vec,
                tgt_emo_vec=tgt_emo_vec,
                energy_target=energy_target,
                f0_cond=f0_cond,
                energy_cond=energy_cond,
                mean_prosody=mean_prosody,
                infer=True,
                y=y,
                y_lengths=y_lengths,
            )
            return output

    def add_ce_loss(self, pred, logits, target, nonpadding, losses=None):
        _, L, N = logits.shape
        logits = torch.log(logits + 1e-9)
        unit_logits = logits.view(-1, N)
        targets = F.interpolate(
            target,
            size=L,
            mode="nearest",
        ).squeeze(1)
        unit_loss = self.ce_loss(unit_logits, targets.view(-1).long())
        unit_loss = unit_loss * nonpadding.view(-1)
        unit_loss = unit_loss.sum() / nonpadding.sum()

        unit_accuracy = pred == targets
        unit_accuracy = unit_accuracy.view(-1) * nonpadding.view(-1)
        unit_accuracy = torch.sum(unit_accuracy) / nonpadding.sum()
        losses["unit_loss"] = unit_loss * 0.1
        losses["unit_accuracy"] = unit_accuracy

    def add_emo_loss(self, logits, target, losses=None):
        emo_loss = self.ce_loss(logits, target.long())
        emo_loss = emo_loss.sum() * hparams["lambda_grl"]
        pred = torch.argmax(logits, dim=-1)
        emo_accuracy = pred == target
        emo_accuracy = torch.mean(emo_accuracy.float())
        losses["emo_loss"] = emo_loss
        losses["emo_accuracy"] = emo_accuracy

    def add_dur_loss(self, l_length, losses=None):
        loss_dur = torch.sum(l_length)
        losses["pdur"] = loss_dur * hparams["lambda_ph_dur"]

    def validation_step(self, sample, batch_idx):
        outputs = {}
        val_losses, _ = self(sample, infer=False)
        loss_terms = {k: v for k, v in val_losses.items()
                      if "accuracy" not in k and "_mae" not in k}
        log_terms = dict(loss_terms)
        if "unit_accuracy" in val_losses:
            log_terms["unit_accuracy"] = val_losses["unit_accuracy"]
        outputs["losses"] = log_terms
        outputs["total_loss"] = sum(loss_terms.values())
        outputs["nsamples"] = sample["nsamples"]

        import torch.distributed as dist
        _is_rank0 = not (dist.is_available() and dist.is_initialized()
                         and dist.get_rank() != 0)
        if (
            _is_rank0
            and self.global_step % hparams["valid_infer_interval"] == 0
            and batch_idx < hparams["num_valid_plots"]
        ):
            emo_mels = []
            spk_embed = sample.get("spk_embed")
            spk_id = sample.get("spk_ids")
            ease = sample.get("ease")
            src_emotion_id = sample.get("emotion_ids")
            x = sample["hubert_features"]
            y = sample["mels"]
            y_lengths = sample["mel_lengths"]
            f0s = None
            for tgt_level in target_levels():
                src_vec = tgt_vec = None
                if self.use_emb_cond:
                    src_vec, tgt_vec = self._conv_vecs(
                        sample["item_name"], tgt_level, x.device
                    )
                try:
                    output = self.model(
                        x,
                        spk_embed=spk_embed,
                        spk_id=spk_id,
                        ease=ease,
                        emotion_id=src_emotion_id,
                        tgt_emotion_id=src_emotion_id,
                        src_emo_vec=src_vec,
                        tgt_emo_vec=tgt_vec,
                        f0_cond=sample.get("f0_cond"),
                        energy_cond=sample.get("energy_cond"),
                        mean_prosody=sample.get("mean_prosody"),
                        infer=True,
                        y=y,
                        y_lengths=y_lengths,
                        diffusion_step=hparams.get("audio_diffusion_step", 50),
                    )
                    mel_pred = output["mel_out"]
                except Exception as e:
                    print(
                        f"| valid infer skipped (L{tgt_level}) "
                        f"@step{self.global_step}: {e}",
                        flush=True,
                    )
                    mel_pred = y
                emo_mels.append(mel_pred)
            gt_mel = sample["mels"]
            self.save_valid_result(sample, batch_idx, [gt_mel, emo_mels], f0s=f0s)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        res = super().validation_end(outputs)
        try:
            va = (res or {}).get("tb_log", {}).get("val/unit_accuracy")
        except Exception:
            va = None
        self._update_cross_curriculum(va)
        return res

    def _update_cross_curriculum(self, val_unit_acc):
        """Cross-level curriculum. Once val unit_accuracy crosses the threshold (or a
        force step) the dataset's cross_ratio ramps 0 -> max. Communicated to the data
        workers via sentinel files in work_dir (resumable across restarts)."""
        if not hparams.get("cross_curriculum", False):
            return
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        wdir = hparams["work_dir"]
        trig_path = os.path.join(wdir, "cross_trigger_step.txt")
        ratio_path = os.path.join(wdir, "cross_curriculum.txt")
        thr = float(hparams.get("cross_trigger_unit_acc", 0.97))
        force = int(hparams.get("cross_force_step", 12000))
        ramp = max(1, int(hparams.get("cross_ramp_steps", 6000)))
        rmax = float(hparams.get("cross_ratio_max", 0.3))
        step = self.global_step

        trigger = None
        if os.path.exists(trig_path):
            try:
                trigger = int(open(trig_path).read().strip())
            except Exception:
                trigger = None
        if trigger is None and ((val_unit_acc is not None and val_unit_acc >= thr) or step >= force):
            trigger = step
            with open(trig_path, "w") as f:
                f.write(str(step))
            print(f"| cross-curriculum TRIGGERED @step{step} "
                  f"(val_unit_acc={val_unit_acc}, thr={thr}, force={force})", flush=True)

        ratio = min(rmax, rmax * (step - trigger) / ramp) if trigger is not None else 0.0
        with open(ratio_path, "w") as f:
            f.write(f"{ratio:.4f}")
        if getattr(self, "logger", None) is not None:
            self.logger.add_scalar("cross/ratio", ratio, step)
            if val_unit_acc is not None:
                self.logger.add_scalar("cross/val_unit_acc", val_unit_acc, step)

    def save_valid_result(self, sample, batch_idx, model_out, f0s):
        sr = hparams["audio_sample_rate"]
        gt = model_out[0]
        pred = model_out[1]

        wav_title_gt = "Wav_gt_{}".format(batch_idx)
        wav_gt = self.vocoder.spec2wav(gt[0].cpu())
        self.logger.add_audio(wav_title_gt, wav_gt, self.global_step, sr)

        levels = target_levels()
        for idx, level in enumerate(levels):
            wav_title_pred = "wav_pred_{}/L{}".format(batch_idx, level)
            wav_pred = self.vocoder.spec2wav(pred[idx][0].cpu())
            self.logger.add_audio(wav_title_pred, wav_pred, self.global_step, sr)

        mel_title = "mel_{}".format(batch_idx)
        self.plot_mel(
            batch_idx,
            [gt[0]] + [p[0] for p in pred],
            title=mel_title,
            f0s=f0s,
        )

    def test_step(self, sample, batch_idx):
        sr = hparams["audio_sample_rate"]
        x = sample["hubert_features"]
        y = sample["mels"]
        y_lengths = sample["mel_lengths"]
        spk_id = sample.get("spk_ids")
        ease = sample.get("ease")
        emotion_id = sample.get("emotion_ids")

        outputs = self.model(
            x,
            spk_id=spk_id,
            ease=ease,
            emotion_id=emotion_id,
            tgt_emotion_id=emotion_id,
            src_emo_vec=sample.get("src_emo_vec"),
            tgt_emo_vec=sample.get("tgt_emo_vec"),
            f0_cond=sample.get("f0_cond"),
            energy_cond=sample.get("energy_cond"),
            mean_prosody=sample.get("mean_prosody"),
            infer=True,
            y=y,
            y_lengths=y_lengths,
            diffusion_step=hparams["diffusion_step"],
        )
        item_name = sample["item_name"][0]
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = outputs["mel_out"][0].cpu().numpy()

        base_fn = item_name
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(
            self.save_result,
            args=[
                wav_pred,
                mel_pred,
                base_fn,
                gen_dir,
                None,
                None,
                None,
                sr,
            ],
        )
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "wav_fn_pred": base_fn,
        }
