# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo contains `audio-subtitles/`: a Python pipeline that transcribes audio files to SRT, VTT, or plain text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It is structured around a `BUILD_PLAN.md` with four milestones (demo → CLI+SRT → medium model/VAD → optional Streamlit frontend).

## Setup

```bash
cd audio-subtitles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# CLI (Milestone 2+)
python cli.py path/to/audio.mp3 -o output.srt -f srt
python cli.py path/to/audio.mp3 -o output.txt -f txt --no-preprocess

# Available format options: srt | vtt | txt
```

## Architecture

The pipeline is: **preprocess → transcribe → format → output**

- `config.py` — Global defaults (model size, device, compute type, sample rate, output format, VAD). Wire changes here to affect CLI and transcriber behavior.
- `pipeline/transcriber.py` — Defines `Segment(start, end, text)`, `Transcriber` protocol, and `WhisperTranscriber`. The model is lazy-loaded on first call. New backends (e.g. AssemblyAI) should implement the `Transcriber` protocol.
- `pipeline/preprocessor.py` — `preprocess(path) -> path`. Currently a passthrough; intended for 16 kHz resampling and loudness normalization using `librosa` or `pydub` (commented out in requirements).
- `pipeline/formatter.py` — `format_segments(segments, fmt) -> str`. Converts `list[Segment]` to SRT, VTT, or plain text. SRT uses comma for milliseconds (`HH:MM:SS,mmm`); VTT uses dot (`HH:MM:SS.mmm`).
- `cli.py` — Typer app wiring the above together. Takes `audio_path` as argument; `--output`, `--format`, `--no-preprocess` as options.

## Apple Silicon note

faster-whisper uses CTranslate2, which has no Metal/MPS backend. Always use `device="cpu"` with `compute_type="int8"` and set `cpu_threads` to your core count. This is already the default in `config.py`.

## Current milestone status

- **Milestone 1** (transcriber + demo): implemented
- **Milestone 2** (CLI, formatter, config): implemented; `preprocessor.py` is still a passthrough (TODO to add resampling)
- **Milestone 3** (medium model, VAD): config supports it via `model_size` and `vad_filter`; not yet tested on messy audio
- **Milestone 4** (Streamlit frontend): not started; `streamlit` dependency is commented out in `requirements.txt`
