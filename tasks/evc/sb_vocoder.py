"""Vocoder factory. Default = SpeechBrain HiFiGAN (libritts-16kHz).

BigVGAN homogenized speaker identity; SpeechBrain HiFiGAN preserves it, so all
mel->wav in this `final` setup goes through SpeechBrain. Set hparams
`audio_vocoder: speechbrain_hifigan` (default here) to use it; anything else
falls back to the original BigVGAN VocoderInfer.
"""
import torch


class SpeechBrainHifiGanInfer:
    """SpeechBrain HiFiGAN with the same .spec2wav(mel) interface as VocoderInfer."""

    def __init__(self, source, savedir):
        try:
            from speechbrain.inference.vocoders import HIFIGAN
        except ImportError:
            from speechbrain.pretrained import HIFIGAN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HIFIGAN.from_hparams(
            source=source, savedir=savedir, run_opts={"device": self.device}
        )

    def spec2wav(self, mel):
        if not torch.is_tensor(mel):
            mel = torch.FloatTensor(mel)
        if mel.dim() != 2:
            raise ValueError(f"Expected 2D mel, got {tuple(mel.shape)}")
        if mel.shape[0] != 80:
            mel = mel.transpose(0, 1)
        mel = mel.unsqueeze(0).to(self.device)
        with torch.no_grad():
            wav = self.model.decode_batch(mel)
        return wav.squeeze().detach().cpu().numpy()


class BigVGANv2Infer:
    """NVIDIA BigVGAN-v2 (44.1kHz / 128-band) with the same .spec2wav(mel) interface.

    Expects mel in BigVGAN's log-mel domain (log(clamp(x,1e-5))), which is exactly
    what utils/audio.mel_spectrogram produces -> this renderer's predicted mel is
    directly decodable here at 44.1k. ckpt_dir holds bigvgan.py + config.json +
    bigvgan_generator.pt (HF snapshot copied locally for offline nodes).
    """

    def __init__(self, ckpt_dir):
        import sys
        _clash = ("utils", "env", "bigvgan", "activations", "meldataset")
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k in _clash or k.startswith("utils.")}
        sys.path.insert(0, ckpt_dir)
        try:
            import bigvgan as _bv
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = _bv.BigVGAN.from_pretrained(ckpt_dir, use_cuda_kernel=False)
        finally:
            if ckpt_dir in sys.path:
                sys.path.remove(ckpt_dir)
            for k in list(sys.modules):
                if k in _clash or k.startswith("utils."):
                    del sys.modules[k]
            sys.modules.update(saved)
        self.model.remove_weight_norm()
        self.model = self.model.to(self.device).eval()
        self.n_mels = self.model.h.num_mels

    def spec2wav(self, mel):
        if not torch.is_tensor(mel):
            mel = torch.FloatTensor(mel)
        if mel.dim() != 2:
            raise ValueError(f"Expected 2D mel, got {tuple(mel.shape)}")
        if mel.shape[0] != self.n_mels:
            mel = mel.transpose(0, 1)
        mel = mel.unsqueeze(0).to(self.device)
        with torch.no_grad():
            wav = self.model(mel)
        return wav.squeeze().detach().cpu().numpy()


def build_vocoder(hparams):
    try:
        audio_vocoder = hparams["audio_vocoder"]
    except KeyError:
        audio_vocoder = None
    if audio_vocoder == "bigvgan_v2":
        return BigVGANv2Infer(hparams["bigvgan_v2_ckpt_dir"])
    if audio_vocoder == "speechbrain_hifigan":
        return SpeechBrainHifiGanInfer(
            source=hparams["sb_vocoder_source"],
            savedir=hparams["sb_vocoder_savedir"],
        )
    from tasks.evc.evc_utils import VocoderInfer
    return VocoderInfer(hparams)
