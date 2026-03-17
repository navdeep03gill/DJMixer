"""
Unit tests for pipeline.formatter — no model, no audio, always fast.
"""
import pytest

from pipeline.formatter import _ts_srt, _ts_vtt, format_segments
from pipeline.transcriber import Segment

SEGMENTS = [
    Segment(start=0.0, end=2.5, text="Hello world"),
    Segment(start=2.5, end=5.0, text="This is a test"),
]


# ── Timestamp helpers ──────────────────────────────────────────────────────

class TestTimestampSRT:
    def test_zero(self):
        assert _ts_srt(0.0) == "00:00:00,000"

    def test_half_second(self):
        assert _ts_srt(0.5) == "00:00:00,500"

    def test_full_second(self):
        assert _ts_srt(1.0) == "00:00:01,000"

    def test_minutes(self):
        assert _ts_srt(90.0) == "00:01:30,000"

    def test_hours(self):
        assert _ts_srt(3661.25) == "01:01:01,250"

    def test_uses_comma_separator(self):
        ts = _ts_srt(1.1)
        assert "," in ts
        assert "." not in ts


class TestTimestampVTT:
    def test_zero(self):
        assert _ts_vtt(0.0) == "00:00:00.000"

    def test_half_second(self):
        assert _ts_vtt(0.5) == "00:00:00.500"

    def test_minutes(self):
        assert _ts_vtt(90.0) == "00:01:30.000"

    def test_uses_dot_separator(self):
        ts = _ts_vtt(1.1)
        assert "." in ts
        assert "," not in ts


# ── SRT format ─────────────────────────────────────────────────────────────

class TestFormatSRT:
    def test_contains_sequential_indices(self):
        out = format_segments(SEGMENTS, "srt")
        lines = out.splitlines()
        assert lines[0] == "1"

    def test_timestamp_line(self):
        out = format_segments(SEGMENTS, "srt")
        assert "00:00:00,000 --> 00:00:02,500" in out

    def test_second_block_index(self):
        out = format_segments(SEGMENTS, "srt")
        assert "\n2\n" in out

    def test_text_present(self):
        out = format_segments(SEGMENTS, "srt")
        assert "Hello world" in out
        assert "This is a test" in out

    def test_empty_input(self):
        assert format_segments([], "srt") == ""

    def test_single_segment(self):
        segs = [Segment(start=0.0, end=1.0, text="Only one")]
        out = format_segments(segs, "srt")
        assert out.startswith("1\n")
        assert "Only one" in out


# ── VTT format ─────────────────────────────────────────────────────────────

class TestFormatVTT:
    def test_webvtt_header(self):
        out = format_segments(SEGMENTS, "vtt")
        assert out.startswith("WEBVTT")

    def test_timestamp_line(self):
        out = format_segments(SEGMENTS, "vtt")
        assert "00:00:00.000 --> 00:00:02.500" in out

    def test_text_present(self):
        out = format_segments(SEGMENTS, "vtt")
        assert "Hello world" in out

    def test_empty_input_still_has_header(self):
        out = format_segments([], "vtt")
        assert out.startswith("WEBVTT")


# ── TXT format ─────────────────────────────────────────────────────────────

class TestFormatTXT:
    def test_lines_match_segments(self):
        out = format_segments(SEGMENTS, "txt")
        assert out.splitlines() == ["Hello world", "This is a test"]

    def test_empty_input(self):
        assert format_segments([], "txt") == ""

    def test_no_timestamps(self):
        out = format_segments(SEGMENTS, "txt")
        assert "-->" not in out


# ── Dispatch ───────────────────────────────────────────────────────────────

class TestFormatDispatch:
    def test_unknown_format_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown format"):
            format_segments(SEGMENTS, "xml")

    @pytest.mark.parametrize("fmt", ["srt", "vtt", "txt"])
    def test_all_formats_return_string(self, fmt):
        result = format_segments(SEGMENTS, fmt)
        assert isinstance(result, str)
