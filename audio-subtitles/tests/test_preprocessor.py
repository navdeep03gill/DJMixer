"""
Tests for pipeline.preprocessor — uses synthetic WAV fixtures, no model needed.
"""
import os
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from pipeline.preprocessor import preprocess, _normalize_loudness, _reduce_noise


class TestResample:
    def test_output_is_16k(self, sine_wav_44k):
        out = preprocess(str(sine_wav_44k))
        try:
            _, sr = sf.read(out)
            assert sr == 16000
        finally:
            os.unlink(out)

    def test_already_16k_stays_16k(self, sine_wav_16k):
        out = preprocess(str(sine_wav_16k))
        try:
            _, sr = sf.read(out)
            assert sr == 16000
        finally:
            os.unlink(out)

    def test_stereo_converted_to_mono(self, sine_wav_stereo):
        out = preprocess(str(sine_wav_stereo))
        try:
            audio, _ = sf.read(out)
            assert audio.ndim == 1, "Expected 1-D (mono) array"
        finally:
            os.unlink(out)

    def test_mono_input_stays_mono(self, sine_wav_44k):
        out = preprocess(str(sine_wav_44k))
        try:
            audio, _ = sf.read(out)
            assert audio.ndim == 1
        finally:
            os.unlink(out)


class TestLoudnessNormalization:
    def test_normalized_loudness_near_target(self, sine_wav_44k):
        """Output integrated loudness should be close to the configured target."""
        import pyloudnorm as pyln
        import config

        out = preprocess(str(sine_wav_44k))
        try:
            audio, sr = sf.read(out)
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            assert abs(loudness - config.loudness_target_lufs) < 1.0, (
                f"Expected loudness near {config.loudness_target_lufs} LUFS, "
                f"got {loudness:.1f} LUFS"
            )
        finally:
            os.unlink(out)

    def test_silent_audio_does_not_crash(self, silent_wav):
        """Silent input falls back to peak normalization without crashing."""
        out = preprocess(str(silent_wav))
        try:
            audio, _ = sf.read(out)
            assert np.all(audio == 0.0)
        finally:
            os.unlink(out)

    def test_fallback_peak_norm_when_pyloudnorm_absent(self, sine_wav_44k):
        """When pyloudnorm raises, output should still be peak-normalized."""
        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "pyloudnorm":
                raise ImportError("simulated missing pyloudnorm")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            audio = np.array([0.1, 0.3, 0.5, 0.2], dtype=np.float32)
            result = _normalize_loudness(audio, sr=16000, target_lufs=-16.0)
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=0.01)


class TestNoiseLoudnessUnit:
    """Unit tests for the private helpers — no file I/O."""

    def test_normalize_loudness_short_signal_fallback(self):
        """Signal too short for EBU R128 should fall back to peak norm."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = _normalize_loudness(audio, sr=16000, target_lufs=-16.0)
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=0.01)

    def test_normalize_loudness_zero_signal(self):
        """Silent signal should be returned as-is (no division by zero)."""
        audio = np.zeros(1600, dtype=np.float32)
        result = _normalize_loudness(audio, sr=16000, target_lufs=-16.0)
        assert np.all(result == 0.0)

    def test_reduce_noise_returns_same_shape(self):
        """Noise reduction must not change the number of samples."""
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = _reduce_noise(audio, sr)
        assert result.shape == audio.shape

    def test_reduce_noise_does_not_crash_on_silence(self):
        audio = np.zeros(16000, dtype=np.float32)
        result = _reduce_noise(audio, sr=16000)
        assert result.shape == audio.shape


class TestNoiseReductionFlag:
    def test_noise_reduction_enabled_runs_without_crash(self, sine_wav_16k):
        import config
        original = config.noise_reduction
        config.noise_reduction = True
        try:
            out = preprocess(str(sine_wav_16k))
            assert os.path.exists(out)
            os.unlink(out)
        finally:
            config.noise_reduction = original

    def test_noise_reduction_disabled_by_default(self, sine_wav_16k):
        import config
        assert config.noise_reduction is False  # default off


class TestOutputPath:
    def test_returns_wav(self, sine_wav_44k):
        out = preprocess(str(sine_wav_44k))
        try:
            assert out.endswith(".wav")
        finally:
            os.unlink(out)

    def test_output_file_exists(self, sine_wav_44k):
        out = preprocess(str(sine_wav_44k))
        try:
            assert os.path.exists(out)
        finally:
            os.unlink(out)

    def test_output_differs_from_input(self, sine_wav_44k):
        out = preprocess(str(sine_wav_44k))
        try:
            assert out != str(sine_wav_44k)
        finally:
            os.unlink(out)

    def test_repeated_calls_produce_different_paths(self, sine_wav_44k):
        out1 = preprocess(str(sine_wav_44k))
        out2 = preprocess(str(sine_wav_44k))
        try:
            assert out1 != out2
        finally:
            os.unlink(out1)
            os.unlink(out2)
