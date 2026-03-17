"""
End-to-end pipeline tests: preprocess → transcribe → format.

Marked @pytest.mark.slow — load the Whisper model.

Also contains a baseline test for the music file (morgan_wallen_concert.mp3)
that measures WER *without* vocal separation, establishing a before/after
benchmark for when Milestone 3 vocal separation is added.
"""
import re

import pytest

from pipeline.formatter import format_segments
from pipeline.preprocessor import preprocess
from tests.helpers import wer, segments_to_text

SRT_BLOCK_RE = re.compile(
    r"^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+",
    re.MULTILINE,
)
VTT_TIMESTAMP_RE = re.compile(
    r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}"
)

SPEECH_FILES = [
    "mp3-files/10_Second_Pep_Talk.mp3",
    "mp3-files/wednesday_dialogue.mp3",
]


def _root(fixtures_dir):
    return fixtures_dir.parent.parent  # audio-subtitles/


def _pipeline(whisper_transcriber, audio_path: str, fmt: str) -> tuple:
    processed = preprocess(audio_path)
    segments = whisper_transcriber.transcribe(processed)
    return format_segments(segments, fmt), segments


# ── Output format validation ───────────────────────────────────────────────

@pytest.mark.slow
class TestOutputFormats:
    @pytest.mark.parametrize("audio_rel", SPEECH_FILES)
    def test_srt_structure(self, whisper_transcriber, fixtures_dir, audio_rel):
        path = _root(fixtures_dir) / audio_rel
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")
        out, _ = _pipeline(whisper_transcriber, str(path), "srt")
        assert SRT_BLOCK_RE.search(out), f"No valid SRT block found:\n{out[:300]}"

    @pytest.mark.parametrize("audio_rel", SPEECH_FILES)
    def test_vtt_structure(self, whisper_transcriber, fixtures_dir, audio_rel):
        path = _root(fixtures_dir) / audio_rel
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")
        out, _ = _pipeline(whisper_transcriber, str(path), "vtt")
        assert out.startswith("WEBVTT"), "Missing WEBVTT header"
        assert VTT_TIMESTAMP_RE.search(out), f"No valid VTT timestamp:\n{out[:300]}"

    @pytest.mark.parametrize("audio_rel", SPEECH_FILES)
    def test_txt_nonempty(self, whisper_transcriber, fixtures_dir, audio_rel):
        path = _root(fixtures_dir) / audio_rel
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")
        out, _ = _pipeline(whisper_transcriber, str(path), "txt")
        assert out.strip(), "TXT output was empty"


# ── WER regression for speech files ───────────────────────────────────────

@pytest.mark.slow
class TestSpeechWER:
    @pytest.mark.parametrize("audio_rel,truth_rel", [
        ("mp3-files/10_Second_Pep_Talk.mp3", "tests/fixtures/pep_talk.txt"),
        ("mp3-files/wednesday_dialogue.mp3", "tests/fixtures/wednesday_dialogue.txt"),
    ])
    def test_e2e_wer(self, whisper_transcriber, fixtures_dir, audio_rel, truth_rel):
        root = _root(fixtures_dir)
        audio_path = root / audio_rel
        truth_path = root / truth_rel

        if not audio_path.exists():
            pytest.skip(f"Audio not found: {audio_path}")
        if not truth_path.exists():
            pytest.skip(
                f"Ground truth missing: {truth_path}\n"
                "  Fill in the correct transcript to enable this test."
            )

        reference = truth_path.read_text(encoding="utf-8").strip()
        if reference.startswith("TODO"):
            pytest.skip(
                f"Ground truth not filled in: {truth_rel}\n"
                "  Replace the TODO placeholder with the real transcript."
            )

        out, _ = _pipeline(whisper_transcriber, str(audio_path), "txt")
        score = wer(reference, out)

        assert score < 0.25, (
            f"E2E WER {score:.1%} on {audio_rel}\n"
            f"  Reference:  {reference[:120]}\n"
            f"  Hypothesis: {out[:120]}"
        )


# ── Music baseline (no vocal separation) ──────────────────────────────────
# This test records the WER *before* Milestone 3 vocal separation is added.
# It is NOT expected to pass a tight threshold — it establishes the baseline.

@pytest.mark.slow
class TestMusicBaseline:
    def test_concert_transcription_produces_output(self, whisper_transcriber, fixtures_dir):
        """Smoke test: pipeline runs on music without crashing."""
        path = _root(fixtures_dir) / "mp3-files/morgan_wallen_concert.mp3"
        if not path.exists():
            pytest.skip(f"Audio not found: {path}")
        out, segments = _pipeline(whisper_transcriber, str(path), "txt")
        # We don't assert WER here — just that output is produced
        assert isinstance(out, str)
        assert len(segments) >= 0  # zero segments is allowed on music without VAD

    def test_concert_wer_baseline(self, whisper_transcriber, fixtures_dir):
        """
        Measures WER against known lyrics and RECORDS it — does not enforce a
        threshold. Used to compare before/after vocal separation is added.
        """
        root = _root(fixtures_dir)
        audio_path = root / "mp3-files/morgan_wallen_concert.mp3"
        truth_path = root / "tests/fixtures/morgan_wallen_lyrics.txt"

        if not audio_path.exists():
            pytest.skip("Concert audio not found")
        if not truth_path.exists():
            pytest.skip(
                "Lyrics ground truth missing: tests/fixtures/morgan_wallen_lyrics.txt\n"
                "  Add the correct lyrics to enable this baseline measurement."
            )

        reference = truth_path.read_text(encoding="utf-8").strip()
        out, _ = _pipeline(whisper_transcriber, str(audio_path), "txt")
        score = wer(reference, out)

        # Print baseline — visible with pytest -s
        print(f"\n[BASELINE] Concert WER without vocal separation: {score:.1%}")
        print(f"  Reference:  {reference[:100]}")
        print(f"  Hypothesis: {out[:100]}")

        # No assertion — this is a measurement, not a pass/fail
