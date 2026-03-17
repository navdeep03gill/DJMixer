# audio-subtitles

Transcribe audio to text with timestamps (SRT, VTT, or plain text) using faster-whisper. Includes a full preprocessing pipeline with vocal isolation, loudness normalization, and optional noise reduction.

## Setup

```bash
cd audio-subtitles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage

```bash
python cli.py path/to/audio.mp3 -o output.srt -f srt
python cli.py path/to/audio.mp3 -o output.txt -f txt --no-preprocess
```

| Option | Default | Description |
|---|---|---|
| `--output` / `-o` | stdout | Output file path |
| `--format` / `-f` | `srt` | Output format: `srt`, `vtt`, or `txt` |
| `--no-preprocess` | off | Skip all preprocessing, pass audio directly to Whisper |

## Preprocessing pipeline

By default, every audio file goes through a 4-step preprocessing pipeline before transcription:

```
Input audio
    │
    ▼
[1] Vocal separation (Demucs)     ← off by default, enable for music
    │  Strips drums, bass, and instrumentation — keeps only vocals
    ▼
[2] Resample to 16 kHz mono       ← always on
    │  Whisper requires 16 kHz mono input
    ▼
[3] Loudness normalization         ← always on
    │  EBU R128 integrated loudness, target -16 LUFS
    │  Falls back to peak normalization if signal is too short/quiet
    ▼
[4] Noise reduction                ← off by default
    │  Stationary noise gate (75% reduction of estimated background noise)
    ▼
Preprocessed WAV → Whisper
```

### Configuring preprocessing

All preprocessing flags live in `config.py`:

```python
# Preprocessing
separate_vocals: bool = False      # enable Demucs vocal separation (for music)
noise_reduction: bool = False      # enable stationary noise gate
loudness_target_lufs: float = -16.0  # EBU R128 loudness target

# Demucs model (only used when separate_vocals = True)
demucs_model: str = "htdemucs"    # htdemucs | htdemucs_ft (fine-tuned)
```

### When to enable each step

**`separate_vocals = True`** — use for music files, concert recordings, or any audio where speech is mixed with instruments. Demucs (`htdemucs`) isolates the vocals stem before passing it to Whisper. First run downloads the model (~80 MB).

**`noise_reduction = True`** — use for audio with constant background noise (HVAC hum, crowd ambience, room reverb). Has no effect on clean speech and adds processing time, so leave off unless needed.

**`--no-preprocess`** — skip the entire pipeline. Useful if the audio is already a clean 16 kHz mono WAV, or for debugging to isolate whether a transcription issue is in preprocessing or the model.

### Skipping preprocessing entirely

```bash
python cli.py audio.wav --no-preprocess -o output.srt
```

### Example: transcribing a song

```python
# config.py
separate_vocals = True    # isolate lyrics from instruments
noise_reduction = False   # not needed after vocal separation
```

```bash
python cli.py song.mp3 -o lyrics.srt -f srt
```

## Project layout

```
audio-subtitles/
├── cli.py                        # Typer CLI entry point
├── config.py                     # All tunable defaults
├── pipeline/
│   ├── preprocessor.py           # 4-step preprocessing pipeline
│   ├── vocal_separator.py        # Demucs vocal isolation
│   ├── transcriber.py            # Transcriber protocol + WhisperTranscriber
│   └── formatter.py              # Segments → SRT / VTT / TXT
└── tests/
    ├── conftest.py               # Shared fixtures (synthetic audio, model singleton)
    ├── generate_fixtures.py      # Generates TTS audio fixtures via macOS 'say'
    ├── fixtures/                 # Ground truth .txt files for accuracy tests
    ├── test_formatter.py         # Unit tests — no model needed
    ├── test_preprocessor.py      # Unit + integration tests — no model needed
    ├── test_transcriber.py       # WER accuracy tests (slow)
    ├── test_vocal_separator.py   # Demucs smoke tests (slow)
    └── test_pipeline_e2e.py      # Full pipeline tests (slow)
```

## Running tests

```bash
# Fast unit tests only (~3s, no model)
pytest -m "not slow"

# Full suite including accuracy and vocal separation tests (~3 min)
pytest -s

# Generate TTS audio fixtures (run once before slow tests)
python tests/generate_fixtures.py
```

Slow tests measure Word Error Rate (WER) against ground truth transcripts in `tests/fixtures/`. Tests for user-provided audio are skipped until the corresponding `.txt` file is filled in.

## Apple Silicon note

faster-whisper uses CTranslate2, which has no Metal/MPS backend. Always use `device="cpu"` with `compute_type="int8"`. The defaults in `config.py` are already set correctly for Apple Silicon.

See **../BUILD_PLAN.md** for the full milestone roadmap.
