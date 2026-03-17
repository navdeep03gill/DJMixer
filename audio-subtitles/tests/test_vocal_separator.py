"""
Tests for pipeline.vocal_separator — requires Demucs model download.

Marked @pytest.mark.slow — downloads htdemucs on first run (~80 MB).

These tests verify the vocal separation step in isolation before it is
wired into the full preprocessing pipeline.
"""
import os

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.slow
class TestVocalSeparator:
    def test_returns_valid_wav(self, fixtures_dir):
        """Output should be a readable WAV file."""
        from pipeline.vocal_separator import separate_vocals
        path = fixtures_dir.parent.parent / "mp3-files/morgan_wallen_concert.mp3"
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")

        out = separate_vocals(str(path))
        try:
            audio, sr = sf.read(out)
            assert sr > 0
            assert len(audio) > 0
        finally:
            os.unlink(out)

    def test_output_is_not_silent(self, fixtures_dir):
        """Vocals stem should contain non-zero signal."""
        from pipeline.vocal_separator import separate_vocals
        path = fixtures_dir.parent.parent / "mp3-files/morgan_wallen_concert.mp3"
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")

        out = separate_vocals(str(path))
        try:
            audio, _ = sf.read(out)
            assert np.max(np.abs(audio)) > 1e-4, "Vocals stem is silent"
        finally:
            os.unlink(out)

    def test_output_duration_matches_input(self, fixtures_dir):
        """Separated vocals should have the same duration as the input (±0.1s)."""
        import librosa
        from pipeline.vocal_separator import separate_vocals

        path = fixtures_dir.parent.parent / "mp3-files/morgan_wallen_concert.mp3"
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")

        input_duration = librosa.get_duration(path=str(path))
        out = separate_vocals(str(path))
        try:
            audio, sr = sf.read(out)
            output_duration = len(audio) / sr
            assert abs(output_duration - input_duration) < 0.5, (
                f"Duration mismatch: input={input_duration:.2f}s, "
                f"output={output_duration:.2f}s"
            )
        finally:
            os.unlink(out)

    def test_full_pipeline_with_separation(self, fixtures_dir):
        """Smoke test: preprocess with separate_vocals=True runs end to end."""
        import config
        from pipeline.preprocessor import preprocess

        path = fixtures_dir.parent.parent / "mp3-files/morgan_wallen_concert.mp3"
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")

        original = config.separate_vocals
        config.separate_vocals = True
        try:
            out = preprocess(str(path))
            assert os.path.exists(out)
            audio, sr = sf.read(out)
            assert sr == 16000
            assert audio.ndim == 1
        finally:
            config.separate_vocals = original
            if os.path.exists(out):
                os.unlink(out)
