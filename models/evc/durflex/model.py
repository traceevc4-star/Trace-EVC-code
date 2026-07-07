import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

from .transformer import FFTBlocks, MultiheadAttention
from .utils import dedup_seq, fix_len_compatibility
from .diffusion import Diffusion, Mish
from .duration_predictor import StochasticDurationPredictor

from .style_encoder import StyleEncoder
from .layers import Embedding
from models.evc.durflex.utils import (
    clip_mel2token_to_multiple,
    expand_states,
    LengthRegulator,
)
from utils.audio.align import mel2token_to_dur
from utils.nn.seq_utils import group_hidden_by_segs, sequence_mask


class DurFlexEVC(nn.Module):
    def __init__(self, dict_size, hparams):
        super().__init__()
        self.hparams = hparams
        self.hidden_size = hparams["hidden_size"]
        self.spk_proj = nn.Linear(hparams["ease_dim"], self.hidden_size)
        self.use_emb_cond = hparams.get("use_emb_cond", False)
        self.split_vad_proj = hparams.get("split_vad_proj", False)
        self.vad_dim = int(hparams.get("vad_dim", 3))
        self.use_int_disentangle = hparams.get("use_int_disentangle", False)
        self.use_energy_cond = hparams.get("use_energy_cond", False)
        self.hard_content = hparams.get("hard_content", False)
        self.use_f0_cond = hparams.get("use_f0_cond", False)
        if self.use_f0_cond:
            self.f0_proj = nn.Linear(2, self.hidden_size)
        self.use_energy_frame_cond = hparams.get("use_energy_frame_cond", False)
        if self.use_energy_frame_cond:
            self.energy_proj = nn.Linear(2, self.hidden_size)
        self.use_mean_prosody = hparams.get("use_mean_prosody", False)
        if self.use_mean_prosody:
            self.f0m_proj = nn.Linear(1, self.hidden_size)
            self.enm_proj = nn.Linear(1, self.hidden_size)
        self.use_mp_supervision = hparams.get("use_mp_supervision", False)
        if self.use_mp_supervision:
            self.mp_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                nn.Linear(self.hidden_size, 2))
        emo_dim = hparams["emo_cond_dim"]
        self.emo_use_dim = int(hparams.get("emo_use_dim", emo_dim))
        if self.use_energy_cond:
            self.emo_proj = nn.Linear(self.emo_use_dim, self.hidden_size)
            self.energy_proj = nn.Linear(1, self.hidden_size)
        elif self.split_vad_proj:
            self.emo_proj = nn.Linear(emo_dim - self.vad_dim, self.hidden_size)
            self.vad_proj = nn.Linear(self.vad_dim, self.hidden_size)
            self.vad_scale = nn.Parameter(torch.tensor(1.0))
        elif self.use_int_disentangle:
            _vmid = int(hparams.get("vad_mlp_dim", 64))
            self.emo_proj = nn.Linear(emo_dim - self.vad_dim, self.hidden_size)
            self.vad_mlp = nn.Sequential(
                nn.Linear(self.vad_dim, _vmid), nn.ReLU(), nn.Linear(_vmid, _vmid))
            self.vad_proj = nn.Linear(_vmid, self.hidden_size)
        else:
            self.emo_proj = nn.Linear(emo_dim, self.hidden_size)
        _es = np.load(hparams["ease_stats_path"])
        self.register_buffer("ease_mean", torch.tensor(_es["mean"], dtype=torch.float32))
        self.register_buffer("ease_std", torch.tensor(_es["std"], dtype=torch.float32))
        self.n_feats = hparams["audio_num_mel_bins"]

        if hparams["use_spk_encoder"]:
            self.spk_encoder = StyleEncoder()
            self.emo_clf = torch.nn.Sequential(
                RevGrad(),
                nn.Linear(256, 1024),
                Mish(),
                nn.Linear(1024, 256),
                nn.Linear(256, 5),
            )

        self.feat_proj = nn.Linear(hparams["feature_dims"], self.hidden_size)
        self.destyle_enc = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="mixstyle",
        )
        self.style_enc = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="saln",
        )
        self.embed = nn.Parameter(torch.FloatTensor(dict_size, self.hidden_size))
        nn.init.normal_(self.embed, mean=0, std=0.5)

        self.unit_aligner = MultiheadAttention(
            self.hidden_size,
            hparams["unit_attn_num_heads"],
            dropout=hparams["unit_attn_dropout"],
        )
        self.unit_level_encoder = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="saln",
        )
        self.frame_level_encoder = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="saln",
        )
        hubert_fps = 16000 / 320
        self.scale = (hparams["audio_sample_rate"] / hparams["hop_size"]) / hubert_fps

        self.num_downsamplings_in_unet = len(hparams["decoder"]["dim_mults"]) - 1
        self.segment_size = hparams["segment_size"]
        self.diffusion = Diffusion(
            n_feats=hparams["audio_num_mel_bins"],
            dim=hparams["decoder"]["dim"],
            dim_mults=hparams["decoder"]["dim_mults"],
            pe_scale=hparams["decoder"]["pe_scale"],
            beta_min=hparams["decoder"]["beta_min"],
            beta_max=hparams["decoder"]["beta_max"],
            spk_emb_dim=hparams["decoder"]["spk_emb_dim"],
        )

        self.proj_m = nn.Linear(hparams["hidden_size"], hparams["audio_num_mel_bins"])

        self.dur_predictor = StochasticDurationPredictor(
            hparams["hidden_size"],
            hparams["hidden_size"],
            3,
            0.5,
            4,
            gin_channels=hparams["hidden_size"],
        )
        self.length_regulator = LengthRegulator()

        if self.use_int_disentangle:
            self.style_int_head = nn.Linear(self.hidden_size, 1)
            self.use_content_grl = hparams.get("use_content_grl", True)
            if self.use_content_grl:
                self.content_int_grl = RevGrad(alpha=float(hparams.get("grl_alpha", 1.0)))
                self.content_int_head = nn.Linear(self.hidden_size, 1)
        self.use_dur_predictor = hparams.get("use_dur_predictor", True)

    def forward(
        self,
        x=None,
        mel2unit=None,
        spk_embed=None,
        spk_id=None,
        ease=None,
        emotion_id=None,
        tgt_emotion_id=None,
        src_emo_vec=None,
        tgt_emo_vec=None,
        energy_target=None,
        f0_cond=None,
        energy_cond=None,
        mean_prosody=None,
        infer=False,
        y=None,
        y_lengths=None,
        diffusion_step=4,
        **kwargs,
    ):
        ret = {}
        src_spk_embed, src_emo_embed, _ = self.forward_style_embed(
            y, y_lengths, emotion_id, spk_embed, spk_id, ease,
            emo_vec=src_emo_vec,
        )
        tgt_spk_embed, tgt_emo_embed, tgt_vad_embed = self.forward_style_embed(
            y, y_lengths, tgt_emotion_id, spk_embed, spk_id, ease,
            emo_vec=tgt_emo_vec,
        )
        if self.hparams["use_spk_encoder"]:
            emo_logits = self.emo_clf(tgt_spk_embed)
            ret["emo_logits"] = emo_logits
        src_meta_embed = src_spk_embed + src_emo_embed
        tgt_meta_embed = tgt_spk_embed + tgt_emo_embed
        if self.use_energy_cond and energy_target is not None:
            energy_embed = self.energy_proj(energy_target.view(-1, 1).float())
            tgt_meta_embed = tgt_meta_embed + energy_embed
        if self.use_mean_prosody and mean_prosody is not None:
            mp = mean_prosody.float()
            tgt_meta_embed = tgt_meta_embed + self.f0m_proj(mp[:, 0:1]) + self.enm_proj(mp[:, 1:2])

        x = self.feat_proj(x)
        N, L, _ = x.shape
        x = self.destyle_enc(
            x,
            style_vector=src_meta_embed.unsqueeze(1).transpose(0, 1),
        )
        if self.use_int_disentangle:
            ret["style_int_pred"] = torch.sigmoid(
                self.style_int_head(tgt_vad_embed)).squeeze(-1)
            if self.use_content_grl:
                content_pool = x.mean(dim=1)
                ret["content_int_pred"] = torch.sigmoid(self.content_int_head(
                    self.content_int_grl(content_pool))).squeeze(-1)
        x = self.style_enc(
            x,
            style_vector=tgt_meta_embed.unsqueeze(1).transpose(0, 1),
        )

        embed_ = self.embed.unsqueeze(0).expand(N, -1, -1)
        keys = embed_

        x, unit_logits = self.unit_aligner(
            x.transpose(0, 1),
            keys.transpose(0, 1),
            keys.transpose(0, 1),
            need_weights=True,
            before_softmax=True,
        )

        unit_pred = torch.argmax(unit_logits, dim=-1)
        _fuf = kwargs.get("force_unit_frames", None)
        if _fuf is not None:
            L = unit_pred.shape[1]
            fu = _fuf[:, :L].long()
            if fu.shape[1] < L:
                fu = F.pad(fu, (0, L - fu.shape[1]))
            unit_pred = fu
        ret["unit_logits"] = unit_logits
        ret["unit_pred"] = unit_pred

        _, count = dedup_seq(unit_pred)
        count = count.to(unit_pred.device)
        mel2unit = self.length_regulator(count)
        unit_len = mel2unit.max()
        unit_pred = (
            group_hidden_by_segs(unit_pred.unsqueeze(-1), mel2unit, unit_len)[0]
            .squeeze(-1)
            .long()
        )
        ret["mel2unit"] = mel2unit
        x_grp = group_hidden_by_segs(x.transpose(0, 1), mel2unit, unit_len)[0]

        x = self.unit_level_encoder(
            x_grp, style_vector=tgt_meta_embed.unsqueeze(1).transpose(0, 1)
        )
        src_nonpadding = (unit_pred > 0).float()[:, :, None]
        dur_inp = x * src_nonpadding
        if not self.use_dur_predictor:
            pass
        elif infer:
            mel2unit = self.forward_dur(
                dur_inp, None, unit_pred, tgt_meta_embed, ret, infer
            )
        else:
            mel2unit = self.forward_dur(
                dur_inp, mel2unit, unit_pred, tgt_meta_embed, ret, infer
            )
        tgt_nonpadding = (mel2unit > 0).float()[:, :, None]
        if self.hard_content:
            x_hard = F.embedding(unit_pred.clamp(min=0), self.embed)
            x_hard_st = x_grp + (x_hard - x_grp).detach()
            x = self.unit_level_encoder(
                x_hard_st, style_vector=tgt_meta_embed.unsqueeze(1).transpose(0, 1)
            )
        x = expand_states(x, mel2unit)

        ret["unit_nonpadding"] = tgt_nonpadding.squeeze(-1)

        if not infer:
            _, l, _ = y.shape
        else:
            l = round(x.shape[1] * self.scale)

        x = F.interpolate(
            x.transpose(1, 2),
            size=l,
            mode="linear",
        ).transpose(1, 2)
        tgt_nonpadding = F.interpolate(
            tgt_nonpadding.float().transpose(1, 2),
            size=l,
            mode="linear",
        ).transpose(1, 2)
        mel2unit = (
            F.interpolate(
                mel2unit.float().unsqueeze(1),
                size=l,
                mode="linear",
            )
            .long()
            .squeeze(1)
        )

        if self.use_f0_cond and f0_cond is not None:
            if f0_cond.shape[1] != l:
                f0_cond = F.interpolate(
                    f0_cond.transpose(1, 2).float(), size=l, mode="linear"
                ).transpose(1, 2)
            x = x + self.f0_proj(f0_cond.float())
        if self.use_energy_frame_cond and energy_cond is not None:
            if energy_cond.shape[1] != l:
                energy_cond = F.interpolate(
                    energy_cond.transpose(1, 2).float(), size=l, mode="linear"
                ).transpose(1, 2)
            x = x + self.energy_proj(energy_cond.float())

        style_embed = tgt_meta_embed.unsqueeze(1)
        if style_embed.shape[1] == 1:
            style_embed = style_embed.expand(x.shape[0], x.shape[1], -1)

        x = self.frame_level_encoder(
            x,
            style_vector=style_embed.transpose(0, 1),
        )

        if self.use_mp_supervision:
            _m = tgt_nonpadding
            _pooled = (x * _m).sum(1) / _m.sum(1).clamp(min=1.0)
            ret["mp_pred"] = self.mp_head(_pooled)

        x = self.proj_m(x)
        x = x * tgt_nonpadding
        ret["tgt_nonpadding"] = tgt_nonpadding.squeeze(1)

        cond_y = x.transpose(1, 2)
        y_max_length = cond_y.shape[-1]
        y_mask = tgt_nonpadding
        style_embed = style_embed.transpose(1, 2)

        if not infer:
            if y_max_length < self.segment_size:
                pad_size = self.segment_size - y_max_length
                y = torch.cat([y, torch.zeros_like(y)[:, :, :pad_size]], dim=-1)
                y_mask = torch.cat(
                    [y_mask, torch.zeros_like(y_mask)[:, :, :pad_size]], dim=-1
                )
                cond_y = torch.cat(
                    [cond_y, torch.zeros_like(cond_y)[:, :, :pad_size]], dim=-1
                )

            max_offset = (y_lengths - self.segment_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)
            cond_y_cut = torch.zeros(
                cond_y.shape[0],
                cond_y.shape[1],
                self.segment_size,
                dtype=cond_y.dtype,
                device=cond_y.device,
            )
            y_cut = torch.zeros(
                y.shape[0],
                self.n_feats,
                self.segment_size,
                dtype=y.dtype,
                device=y.device,
            )
            style_embed_cut = torch.zeros(
                style_embed.shape[0],
                style_embed.shape[1],
                self.segment_size,
                dtype=cond_y.dtype,
                device=cond_y.device,
            )

            y_cut_lengths = []
            y = y.transpose(1, 2)
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = self.segment_size + (
                    y_lengths[i] - self.segment_size
                ).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                cond_y_cut[i, :, :y_cut_length] = cond_y[i, :, cut_lower:cut_upper]
                style_embed_cut[i, :, :y_cut_length] = style_embed[
                    i, :, cut_lower:cut_upper
                ]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            if y_cut_mask.shape[-1] < self.segment_size:
                y_cut_mask = torch.nn.functional.pad(
                    y_cut_mask, (0, self.segment_size - y_cut_mask.shape[-1])
                )
            cond_y_cut = cond_y_cut * y_cut_mask

            diff_loss, xt = self.diffusion.compute_loss(
                y_cut,
                y_cut_mask,
                cond_y_cut,
                spk_emb=style_embed_cut,
            )
            ret["diff_loss"] = diff_loss
        else:
            y_max_length_ = fix_len_compatibility(
                y_max_length, self.num_downsamplings_in_unet
            )
            y_mask = (
                sequence_mask(
                    torch.LongTensor([y_max_length]).to(cond_y.device),
                    y_max_length_,
                )
                .unsqueeze(1)
                .to(y_mask.dtype)
            )
            cond_y = F.pad(cond_y, (0, y_max_length_ - y_max_length))
            style_embed = F.pad(style_embed, (0, y_max_length_ - y_max_length))
            z = torch.randn_like(cond_y, device=cond_y.device)
            decoder_outputs = self.diffusion(
                z,
                y_mask,
                cond_y,
                spk_emb=style_embed,
                n_timesteps=diffusion_step,
            )
            decoder_outputs = decoder_outputs[:, :, :y_max_length]
            ret["mel_out"] = decoder_outputs.transpose(1, 2)
        return ret

    def forward_style_embed(
        self, y=None, y_length=None, emotion_id=None, spk_embed=None, spk_id=None,
        ease=None, emo_vec=None,
    ):
        if self.hparams["use_spk_encoder"]:
            y_mask = sequence_mask(y_length).unsqueeze(1)
            spk_embed = self.spk_encoder(y.transpose(1, 2), y_mask)
        else:
            spk_embed = self.spk_proj((ease - self.ease_mean) / self.ease_std)
        vad_embed = None
        if self.split_vad_proj:
            n = self.emo_proj.in_features
            e2v, vad = emo_vec[..., :n], emo_vec[..., n : n + self.vad_dim]
            emo_embed = self.emo_proj(e2v) + self.vad_scale * self.vad_proj(vad)
        elif self.use_int_disentangle:
            n = self.emo_proj.in_features
            vad_embed = self.vad_proj(self.vad_mlp(emo_vec[..., n : n + self.vad_dim]))
            emo_embed = self.emo_proj(emo_vec[..., :n]) + vad_embed
        else:
            emo_embed = self.emo_proj(emo_vec[..., : self.emo_proj.in_features])
        return spk_embed, emo_embed, vad_embed

    def forward_dur(
        self,
        dur_input,
        mel2ph,
        txt_tokens,
        style_embed,
        ret,
        infer,
        length_scale=1,
        noise_scale_w=1.0,
    ):
        src_padding = txt_tokens == 0
        _, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_input = dur_input.detach()
        if infer:
            logw = self.dur_predictor(
                dur_input.transpose(1, 2),
                nonpadding.unsqueeze(1),
                g=style_embed.unsqueeze(-1),
                reverse=True,
                noise_scale=noise_scale_w,
            )
            dur = torch.exp(logw) * nonpadding * length_scale
            dur = torch.ceil(dur).squeeze(1)
            mel2ph = self.length_regulator(dur, src_padding).detach()
        else:
            dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
            dur = self.dur_predictor(
                dur_input.transpose(1, 2),
                nonpadding.unsqueeze(1),
                dur_gt.unsqueeze(1),
                g=style_embed.unsqueeze(-1),
            )
            dur = dur / torch.sum(nonpadding)

        ret["dur"] = dur
        mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams["frames_multiple"])
        return mel2ph
