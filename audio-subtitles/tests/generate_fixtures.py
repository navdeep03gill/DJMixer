"""
Generate TTS audio fixtures using macOS 'say' command.

Run once from the audio-subtitles directory:
    python tests/generate_fixtures.py

Produces:
    tests/fixtures/clean_speech.wav
    tests/fixtures/numbers.wav

These WAV files are used by the deterministic transcriber accuracy tests.
The matching .txt ground truth files are already committed to the repo.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TARGET_SR = 16000
SPEAK_RATE = 140  # words per minute — slower is clearer for Whisper

PHRASES = {
    "clean_speech": "The quick brown fox jumps over the lazy dog.",
    "numbers": "One two three four five six seven eight nine ten.",
}


def say_to_wav(text: str, out_path: str) -> None:
    """
    Use macOS 'say' (text-to-speech engine) to produce AIFF, then resample to 16 kHz mono WAV.
    Requires: macOS (say), librosa, soundfile.
    """
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff_path = tmp.name

    try:
        subprocess.run(
            ["say", "-r", str(SPEAK_RATE), "-o", aiff_path, text],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("ERROR: 'say' command not found. This script requires macOS.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: 'say' failed: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    import librosa
    audio, _ = librosa.load(aiff_path, sr=TARGET_SR, mono=True)
    os.unlink(aiff_path)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    sf.write(out_path, audio, TARGET_SR)


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)
    print(f"Generating TTS fixtures in {FIXTURES_DIR} ...")

    for name, phrase in PHRASES.items():
        wav_path = str(FIXTURES_DIR / f"{name}.wav")
        print(f"  {name}.wav  \"{phrase}\"")
        say_to_wav(phrase, wav_path)
        print(f"    → {wav_path}")

    print()
    print("Done. Next steps:")
    print("  pytest -m 'not slow'   # fast unit tests")
    print("  pytest -m slow         # accuracy tests (requires model download on first run)")


if __name__ == "__main__":
    main()
