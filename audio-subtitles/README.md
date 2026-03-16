# audio-subtitles

Transcribe audio to text with timestamps (SRT, VTT, or plain text) using faster-whisper behind a simple transcriber abstraction.

## Setup

```bash
cd audio-subtitles
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Quick start (Milestone 1)

```bash
python demo.py path/to/your/audio.mp3
```

Prints segments with start/end times and text.

## CLI (Milestone 2+)

```bash
python cli.py path/to/audio.mp3 -o output.srt -f srt
python cli.py path/to/audio.mp3 -o output.txt -f txt
```

Options: `--output` / `-o`, `--format` / `-f` (srt | vtt | txt), `--no-preprocess`.

## Project layout

- `cli.py` — Typer CLI
- `demo.py` — Minimal transcribe-and-print for testing
- `config.py` — Model size, device, sample rate, output format
- `pipeline/preprocessor.py` — Audio normalization, resampling (path in → path out)
- `pipeline/transcriber.py` — `Transcriber` protocol + `WhisperTranscriber`
- `pipeline/formatter.py` — Segments → SRT / VTT / txt

See **../BUILD_PLAN.md** for the full milestone breakdown and actionable steps.
