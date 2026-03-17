"""
Shared pytest fixtures for the audio-subtitles test suite.
"""
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Put audio-subtitles root on sys.path so `import config` and `pipeline.*` work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# WER threshold for clean TTS speech transcribed by the small model
WER_THRESHOLD = 0.10


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def whisper_transcriber():
    """Session-scoped: model is loaded once and reused across all slow tests."""
    from pipeline.transcriber import WhisperTranscriber
    return WhisperTranscriber(
        model_size="small",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        vad_filter=False,
    )


# ── Synthetic audio helpers ────────────────────────────────────────────────

def _write_sine(path: str, freq: float = 440.0, duration: float = 2.0,
                sample_rate: int = 44100, channels: int = 1) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    sf.write(path, wave, sample_rate)


@pytest.fixture
def sine_wav_44k(tmp_path) -> Path:
    """Mono 44.1 kHz sine — verifies preprocessor resamples to 16 kHz."""
    p = tmp_path / "sine_44k.wav"
    _write_sine(str(p), sample_rate=44100, channels=1)
    return p


@pytest.fixture
def sine_wav_stereo(tmp_path) -> Path:
    """Stereo 44.1 kHz sine — verifies preprocessor converts to mono."""
    p = tmp_path / "sine_stereo.wav"
    _write_sine(str(p), sample_rate=44100, channels=2)
    return p


@pytest.fixture
def sine_wav_16k(tmp_path) -> Path:
    """Already 16 kHz mono sine — verifies preprocessor handles correct input."""
    p = tmp_path / "sine_16k.wav"
    _write_sine(str(p), sample_rate=16000, channels=1)
    return p


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """Silent 16 kHz mono — verifies zero-peak edge case doesn't crash."""
    p = tmp_path / "silent.wav"
    sf.write(str(p), np.zeros(16000, dtype=np.float32), 16000)
    return p
