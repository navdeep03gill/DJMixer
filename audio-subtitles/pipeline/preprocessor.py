"""
Audio preprocessing pipeline:
  1. (Optional) Vocal separation via Demucs  — isolates lyrics from music
  2. Resample to 16 kHz mono                 — Whisper's required format
  3. Loudness normalization (EBU R128)        — consistent input levels
  4. (Optional) Stationary noise reduction   — cleans background hiss/hum

Interface: preprocess(audio_path) -> path (temp WAV, caller cleans up)
"""

import os
import tempfile

import librosa
import numpy as np
import soundfile as sf

import config


def preprocess(audio_path: str) -> str:
    """
    Run the full preprocessing pipeline on audio_path.
    Returns the path of a temp WAV file ready for Whisper.
    Caller is responsible for deleting the file when done.
    """
    working_path = audio_path
    tmp_vocals = None

    # Step 1: Vocal separation
    if config.separate_vocals:
        from pipeline.vocal_separator import separate_vocals
        tmp_vocals = separate_vocals(working_path)
        working_path = tmp_vocals

    # Step 2: Resample to 16 kHz mono
    audio, _ = librosa.load(working_path, sr=config.sample_rate, mono=True)

    if tmp_vocals is not None:
        os.unlink(tmp_vocals)

    # Step 3: Loudness normalization
    audio = _normalize_loudness(audio, config.sample_rate, config.loudness_target_lufs)

    # Step 4: Noise reduction (optional)
    if config.noise_reduction:
        audio = _reduce_noise(audio, config.sample_rate)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, config.sample_rate)
    return tmp.name


def _normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """
    EBU R128 integrated loudness normalization via pyloudnorm.
    Falls back to peak normalization if the signal is too short, silent,
    or if pyloudnorm is not installed.
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness) and loudness > -70.0:
            audio = pyln.normalize.loudness(audio, loudness, target_lufs)
            return audio
    except Exception:
        pass

    # Fallback: peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Stationary noise reduction via noisereduce.
    Conservative prop_decrease (75%) avoids distorting speech while still
    attenuating constant background hiss and room noise.
    """
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.75,
        )
    except Exception:
        pass
    return audio
