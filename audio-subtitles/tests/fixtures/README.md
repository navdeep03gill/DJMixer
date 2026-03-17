# Test Fixtures

## TTS fixtures (generated, deterministic)

These are produced by `generate_fixtures.py` using macOS `say`.
Run once, then re-run whenever the TTS phrases change.

```bash
cd audio-subtitles
python tests/generate_fixtures.py
```

| Audio file        | Ground truth       | Content                                         |
|-------------------|--------------------|-------------------------------------------------|
| clean_speech.wav  | clean_speech.txt   | "The quick brown fox jumps over the lazy dog."  |
| numbers.wav       | numbers.txt        | "One two three four five six seven eight nine ten." |

The `.wav` files are git-ignored. The `.txt` files are committed.

## User-provided MP3 fixtures

Ground truth `.txt` files for the project MP3s must be filled in manually.
Tests that reference these files are **skipped** until the TODO is replaced
with the real transcript.

| Audio file                          | Ground truth file               |
|-------------------------------------|---------------------------------|
| mp3-files/10_Second_Pep_Talk.mp3    | fixtures/pep_talk.txt           |
| mp3-files/wednesday_dialogue.mp3    | fixtures/wednesday_dialogue.txt |
| mp3-files/morgan_wallen_concert.mp3 | fixtures/morgan_wallen_lyrics.txt |

## WER thresholds

| Test class            | Threshold | Rationale                              |
|-----------------------|-----------|----------------------------------------|
| TTS clean speech      | 10%       | TTS is clean; small model should nail it |
| User MP3 speech       | 25%       | Real-world audio, no vocal separation  |
| Music (concert)       | none      | Baseline measurement only, not enforced |
