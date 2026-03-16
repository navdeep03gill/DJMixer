"""
Audio preprocessing: normalize, resample to 16 kHz mono.
Interface: preprocess(audio_path) -> path (temp WAV).
"""

import tempfile

import librosa
import numpy as np
import soundfile as sf

import config


def preprocess(audio_path: str) -> str:
    """
    Load audio, convert to 16 kHz mono, normalize loudness, write to a temp
    WAV file, and return its path. The caller is responsible for cleanup if needed.
    """
    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, config.sample_rate)
    return tmp.name
