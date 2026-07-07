from skimage.transform import resize
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import librosa
import numpy as np
import pyloudnorm as pyln
import warnings

warnings.filterwarnings("ignore", message="Possible clipped samples in output")

int16_max = (2**15) - 1


def trim_long_silences(
    path, sr=None, return_raw_wav=False, norm=True, vad_max_silence_length=12
):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :param vad_max_silence_length: Maximum number of consecutive silent frames a segment can have.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(
        y=wav_raw, orig_sr=sr, target_sr=sampling_rate, res_type="kaiser_best"
    )

    vad_window_length = 30
    vad_moving_average_width = 8

    samples_per_window = (vad_window_length * sampling_rate) // 1000

    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    pcm_wave = struct.pack(
        "%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16)
    )

    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(
                pcm_wave[window_start * 2 : window_end * 2], sample_rate=sampling_rate
            )
        )
    voice_flags = np.array(voice_flags)

    def moving_average(array, width):
        array_padded = np.concatenate(
            (np.zeros((width - 1) // 2), array, np.zeros(width // 2))
        )
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1 :] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask, sr
    return wav_raw[audio_mask], audio_mask, sr
