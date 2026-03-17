"""
Vocal source separation using Demucs.

Extracts only the vocals stem from a mixed audio file, discarding drums,
bass, and other instrumentation. Use this before transcription when the
input contains music (e.g. concert recordings, songs with lyrics).

Interface: separate_vocals(audio_path) -> path (temp WAV, caller cleans up)
"""

import os
import tempfile

import numpy as np

import config

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise ImportError(
            "Demucs is required for vocal separation.\n"
            "Install it with: pip install demucs"
        ) from exc

    _model = get_model(config.demucs_model)
    _model.eval()
    return _model


def _load_audio_tensor(audio_path: str, target_sr: int, target_channels: int):
    """
    Load audio into a float32 torch tensor [channels, samples] at target_sr.
    Uses soundfile + librosa instead of torchaudio.load to avoid the
    torchcodec dependency introduced in torchaudio 2.9.
    """
    import torch
    import librosa

    # mono=False preserves stereo; librosa returns (channels, samples) or (samples,)
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=(target_channels == 1))

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]          # (1, samples)
    if audio.shape[0] < target_channels:
        audio = np.tile(audio, (target_channels, 1))  # mono → stereo duplicate

    return torch.from_numpy(audio.astype(np.float32))


def separate_vocals(audio_path: str) -> str:
    """
    Run Demucs on audio_path, extract the vocals stem, and write it to a
    temporary WAV file. Returns the path; caller is responsible for cleanup.

    The output sample rate matches the Demucs model's native rate (typically
    44.1 kHz). The preprocessor resamples it to 16 kHz afterward.
    """
    import torch
    import soundfile as sf
    from demucs.audio import convert_audio
    from demucs.apply import apply_model

    model = _get_model()

    wav = _load_audio_tensor(audio_path, model.samplerate, model.audio_channels)
    # convert_audio handles any remaining channel/rate mismatch
    wav = convert_audio(wav, model.samplerate, model.samplerate, model.audio_channels)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav.unsqueeze(0),
            device="cpu",
            num_workers=0,
            progress=False,
        )[0]  # [num_sources, channels, samples]

    vocal_idx = model.sources.index("vocals")
    vocals = sources[vocal_idx].cpu().numpy()  # [channels, samples]

    tmp = tempfile.NamedTemporaryFile(suffix="_vocals.wav", delete=False)
    # soundfile expects (samples, channels)
    sf.write(tmp.name, vocals.T, model.samplerate)
    return tmp.name
