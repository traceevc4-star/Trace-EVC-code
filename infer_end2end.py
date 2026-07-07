#!/usr/bin/env python3
"""End-to-end inference: source audio + instruction -> target audio."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths
from tasks.emo_compass.data import EMOTIONS_MEAD, EMO_ID_MEAD
from tasks.emo_compass.infer import encode_instruction_e5, load_flow


def extract_hubert(wav_path: str, device: str) -> torch.Tensor:
    """Continuous HuBERT last-hidden features [T, 768] (the synthesis content input)."""
    from transformers import HubertModel

    from utils.audio.vad import trim_long_silences

    model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).to(device).eval()
    wav, _, _ = trim_long_silences(wav_path, 16000)
    wav = F.pad(torch.from_numpy(wav).float().unsqueeze(0).to(device), (40, 40), "reflect")
    with torch.no_grad():
        return model(wav).hidden_states[-1].squeeze(0).cpu().float()


def extract_mel(wav_path: str, hparams) -> torch.Tensor:
    """Log-mel [T, n_mels] matching the synthesis training domain (length reference)."""
    from utils.audio import wav2spec

    out = wav2spec(
        wav_path, fft_size=hparams["fft_size"], hop_size=hparams["hop_size"],
        win_length=hparams["win_size"], num_mels=hparams["audio_num_mel_bins"],
        fmin=hparams["fmin"], fmax=hparams["fmax"], sample_rate=hparams["audio_sample_rate"],
    )
    return torch.FloatTensor(out["mel"])


def extract_ease(wav_path: str, ease_ckpt: str, device: str) -> torch.Tensor:
    """EASE speaker embedding [ease_dim]: ECAPA x-vector -> trained EASE encoder."""
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier

    from tasks.ease.train_mead_ease import SpeakerModel

    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    sig, sr = torchaudio.load(wav_path)
    if sr != 16000:
        sig = torchaudio.functional.resample(sig, sr, 16000)
    with torch.no_grad():
        xvec = clf.encode_batch(sig.to(device))[0, 0, :].float()

    ck = torch.load(ease_ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state", ck))
    n_spk = ck.get("n_spk") or len(ck.get("spk2id", {})) or 1
    n_emo = ck.get("n_emo") or len(ck.get("emotions", EMOTIONS_MEAD))
    net = SpeakerModel(n_spk, n_emo, in_dim=ck.get("in_dim", xvec.shape[-1]), dim=ck.get("dim", 128))
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        _, _, feat = net(xvec.unsqueeze(0).cpu())
    return feat.squeeze(0)


def extract_emotion2vec(wav_path: str, model_id: str) -> torch.Tensor:
    """emotion2vec utterance embedding [1024] via funasr."""
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "emotion2vec needs funasr (`pip install funasr`), or pass --source-emo2vec / --source-affect."
        ) from exc
    m = AutoModel(model=model_id, disable_update=True)
    res = m.generate(wav_path, granularity="utterance", extract_embedding=True)
    return torch.tensor(np.asarray(res[0]["feats"], dtype=np.float32))


def extract_vad(wav_path: str, model_id: str, device: str) -> torch.Tensor:
    """VAD [arousal, dominance, valence] via the Odyssey-2024 WavLM SER baseline.

    Same model + preprocessing used to build the training VAD (`vad.jsonl`), so
    the predicted coordinates match the distribution Emo-Compass was trained on.
    """
    import librosa
    from transformers import AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    mean, std = float(model.config.mean), float(model.config.std)
    audio = (audio - mean) / (std + 1e-6)
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones_like(wav)
    with torch.no_grad():
        pred = model(wav, mask)
    pred = (pred.logits if hasattr(pred, "logits") else pred).detach().cpu().float().view(-1)
    return pred[:3]


def build_source_affect(args) -> torch.Tensor:
    """Return the raw source affect vector z_src = [emotion2vec(1024) | VAD(3)] = [1027]."""
    if args.source_affect:
        z = torch.load(args.source_affect, map_location="cpu", weights_only=False).float().view(-1)
        if z.numel() != 1027:
            raise ValueError(f"--source-affect must be length 1027, got {z.numel()}")
        return z
    if args.source_emo2vec:
        e = torch.load(args.source_emo2vec, map_location="cpu", weights_only=False).float().view(-1)
    else:
        e = extract_emotion2vec(args.source_wav, args.emotion2vec_model).view(-1)
    if args.source_vad is not None:
        vad = torch.tensor([float(x) for x in args.source_vad.split()], dtype=torch.float32)
        if vad.numel() != 3:
            raise ValueError("--source-vad must be three numbers: arousal dominance valence")
    else:
        vad = extract_vad(args.source_wav, args.vad_model, args.device).view(-1)
    return torch.cat([e, vad], dim=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-wav", required=True)
    p.add_argument("--prompt-text", default=None)
    p.add_argument("--prompt-emb", default=None, help="Precomputed E5 embedding [768] (.pt).")
    p.add_argument("--emotion", choices=EMOTIONS_MEAD, required=True,
                   help="Source emotion axis (for the mean-prosody one-hot).")
    p.add_argument("--out-wav", default="trace_evc_out.wav")
    p.add_argument("--source-affect", default=None, help="Precomputed [1027] = [emotion2vec|VAD].")
    p.add_argument("--source-emo2vec", default=None, help="Precomputed emotion2vec [1024] (.pt).")
    p.add_argument("--source-vad", default=None, help='"arousal dominance valence" (skip live VAD).')
    p.add_argument("--emotion2vec-model", default=str(paths.EMOTION2VEC_MODEL), help="funasr emotion2vec model id.")
    p.add_argument("--vad-model", default=str(paths.ODYSSEY_VAD_MODEL), help="Odyssey WavLM SER model id/dir.")
    p.add_argument("--source-meanpros", default="0 0", help='Source "zF0 zEnergy" prosody (predictor input).')
    p.add_argument("--flow-ckpt", default=str(paths.MEAD_FLOW_CKPT_DIR / "best_trace_mead.pt"))
    p.add_argument("--ease-ckpt", default=str(paths.EASE_CKPT))
    p.add_argument("--e5-model", default=str(paths.E5_MODEL))
    p.add_argument("--cfg", default="trace_evc")
    p.add_argument("--exp", default="DurFlex_ours_mead_meanpros_gtmp")
    p.add_argument("--step", default="455000")
    p.add_argument("--flow-steps", type=int, default=50)
    p.add_argument("--diff-step", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if (args.prompt_emb is None) == (args.prompt_text is None):
        raise SystemExit("Provide exactly one of --prompt-text or --prompt-emb.")
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"

    from utils.commons.hparams import set_hparams, hparams
    set_hparams(f"configs/exp/{args.cfg}.yaml", exp_name=args.exp, print_hparams=False)
    from models.evc.durflex import DurFlexEVC

    model = DurFlexEVC(hparams["n_units"], hparams).to(device).eval()
    model.load_state_dict(torch.load(f"{hparams['work_dir']}/model_ckpt_steps_{args.step}.ckpt",
                                     map_location="cpu", weights_only=False)["state_dict"]["model"])
    try:
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError:
        from speechbrain.pretrained import HIFIGAN
    voc = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                               savedir=os.environ.get("SB_HIFIGAN", "pretrained_hifigan"),
                               run_opts={"device": device})

    den, flow, aff_mean, aff_std, _ = load_flow(args.flow_ckpt, device)
    if args.prompt_emb:
        prompt = torch.load(args.prompt_emb, map_location="cpu", weights_only=False).float().view(1, -1).to(device)
    else:
        prompt = encode_instruction_e5([args.prompt_text], args.e5_model, device)

    z_src = build_source_affect(args).to(device)
    if aff_mean.numel() > z_src.numel():
        src_mp = torch.tensor([float(x) for x in args.source_meanpros.split()], dtype=torch.float32).to(device)
        z_src = torch.cat([z_src, src_mp[: aff_mean.numel() - z_src.numel()]])
    z_src_std = ((z_src - aff_mean) / aff_std).unsqueeze(0)
    z_tgt_std = flow.sample(den, z_src_std, prompt, num_steps=args.flow_steps)

    z_tgt_raw = z_tgt_std.squeeze(0) * aff_std + aff_mean
    mean_prosody = (z_tgt_raw[1027:].unsqueeze(0) if aff_mean.numel() > 1027
                    else torch.zeros(1, 2, device=device))

    hubert = extract_hubert(args.source_wav, device).unsqueeze(0).to(device)
    mel = extract_mel(args.source_wav, hparams).unsqueeze(0).to(device)
    mel_len = torch.LongTensor([mel.shape[1]]).to(device)
    ease = extract_ease(args.source_wav, args.ease_ckpt, device).unsqueeze(0).to(device)
    emo_id = torch.LongTensor([EMO_ID_MEAD[args.emotion]]).to(device)

    out = model(hubert, spk_id=None, ease=ease,
                emotion_id=emo_id, tgt_emotion_id=emo_id,
                src_emo_vec=z_src_std, tgt_emo_vec=z_tgt_std,
                mean_prosody=mean_prosody, infer=True,
                y=mel, y_lengths=mel_len, diffusion_step=args.diff_step)
    wav = voc.decode_batch(out["mel_out"][0].T.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    sf.write(args.out_wav, wav, 16000)
    print(f"DONE -> {args.out_wav}  (mel {tuple(out['mel_out'][0].shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
