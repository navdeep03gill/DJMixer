# Audio-Subtitles: Build Plan & Actionable Steps

This document breaks your transcription pipeline into milestones with concrete steps and the context you need at each stage.

---

## Tech note: Apple Silicon

**faster-whisper does not support MPS.** It uses CTranslate2, which has no Metal backend yet. On Apple Silicon, use:

- `device="cpu"` with `compute_type="int8"` (faster than float32)
- `cpu_threads=8` (or your core count) to use all cores

You still get a big win over vanilla `openai-whisper` from CTranslate2’s optimized CPU kernels. For GPU-style acceleration on Mac, alternatives are whisper.cpp + CoreML or a different stack; the transcriber abstraction lets you swap that in later.

---

## Milestone 1: Working demo (single file → text)

**Goal:** Install faster-whisper, load a small model, transcribe one MP3, print segments. ~15–30 minutes.

### Steps

| # | Action | Context needed |
|---|--------|----------------|
| 1 | Create `audio-subtitles/` and a venv (or use existing Python env) | Python 3.10+ |
| 2 | Add `faster-whisper` to `requirements.txt` and install | — |
| 3 | In `pipeline/transcriber.py`: define `Segment` (start, end, text) and a `Transcriber` protocol with `transcribe(audio_path) -> list[Segment]` | — |
| 4 | Implement `WhisperTranscriber`: load model (e.g. `small`), call `model.transcribe(audio_path)`, map result to `list[Segment]` | [faster-whisper API](https://github.com/SYSTRAN/faster-whisper): `WhisperModel.transcribe()` returns segments with `.start`, `.end`, `.text` |
| 5 | In `cli.py` (or a minimal `demo.py`): instantiate `WhisperTranscriber`, call `transcribe("path/to/sample.mp3")`, print segments | One short clean MP3 (e.g. 30–60 s) |

### Definition of done

- Run one script with one audio path → printed list of (start, end, text) segments.

### Optional

- Read `config.py` for model size (e.g. `small` / `base`) so you can switch later without touching transcriber logic.

---

## Milestone 2: Pipeline + CLI + SRT

**Goal:** Preprocess audio, run through transcriber, output SRT (and optionally plain text). CLI via Typer.

### Steps

| # | Action | Context needed |
|---|--------|----------------|
| 1 | **config.py:** Add defaults: `model_size`, `sample_rate` (16000 for Whisper), `output_format` (`srt` / `txt` / `vtt`), `device`, `compute_type`, `cpu_threads` | — |
| 2 | **pipeline/preprocessor.py:** Define `preprocess(audio_path) -> path`. Optional: normalize loudness, resample to 16 kHz, optional noise reduction. Return path to temp file or original; keep interface “path in → path out”. | `librosa` or `pydub` for resample/load; Whisper expects 16 kHz mono. |
| 3 | **pipeline/formatter.py:** `format_segments(segments: list[Segment], fmt: "srt" \| "vtt" \| "txt") -> str`. Implement SRT (index, timestamps, text), VTT header + timestamps, and plain text (concat). | SRT: `1\n00:00:00,000 --> 00:00:02,500\nText\n\n` |
| 4 | **cli.py:** Typer app. Argument: `audio_path`. Options: `--output`, `--format` (srt/txt/vtt), `--no-preprocess`. Flow: preprocess → transcribe → format → write to file or stdout. | Typer: `@app.command()`, `typer.Argument()`, `typer.Option()`. |
| 5 | Wire `config.py` into transcriber and CLI (model size, device, compute_type, cpu_threads). | — |
| 6 | Test: `python cli.py sample.mp3 -o out.srt` produces valid SRT. | — |

### Definition of done

- Single CLI command: input audio → preprocess → transcribe → SRT (or txt/vtt) to file.

---

## Milestone 3: Medium model + messy audio

**Goal:** Use `medium` model, tune for real-world (noisy, accented, long) audio.

### Steps

| # | Action | Context needed |
|---|--------|----------------|
| 1 | In config, set default model to `medium` (or make it CLI option `--model small/medium/large-v2`). | Medium is ~1.5 GB; first run will download. |
| 2 | Add optional **VAD** (voice activity detection): skip or shorten silence. faster-whisper can use `vad_filter=True` in `transcribe()` to reduce hallucinations. | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `transcribe(..., vad_filter=True)`. |
| 3 | Test on “messy” samples: background noise, accents, 5–10 min files. Compare with/without preprocessing and with/without VAD. | — |
| 4 | Optional: **chunking** for long files (e.g. 10+ min). Split on silence or fixed duration, transcribe chunks, merge segments (adjust timestamps). | Only if you hit OOM or need lower latency; often unnecessary for medium model. |
| 5 | Document in README: recommended settings for “clean” vs “noisy” and for Apple Silicon (cpu, int8, cpu_threads). | — |

### Definition of done

- Default or optional `medium` model; VAD option; tested on noisy/longer audio; README updated.

---

## Milestone 4: Streamlit frontend (optional)

**Goal:** Upload audio in browser, run pipeline, show transcript and download SRT/VTT/txt.

### Steps

| # | Action | Context needed |
|---|--------|----------------|
| 1 | Add `streamlit run app.py` (or `frontend/app.py`). Dependencies: `streamlit` in requirements. | — |
| 2 | Page: file upload (audio), optional format selector, “Transcribe” button. On submit: save upload to temp file, call same pipeline (preprocess → transcribe → format), show text in text area and offer download link for SRT/VTT/txt. | Reuse `pipeline` from Milestone 2; no new business logic. |
| 3 | Optional: progress indicator (e.g. “Transcribing…”) and display of segment count / duration. | — |

### Definition of done

- Upload → transcribe → view + download without using CLI.

---

## Context cheat sheet

| Need | Where / what |
|------|----------------|
| **Segment type** | `start`, `end` (seconds, float), `text` (str). Use a `dataclass` or `NamedTuple`. |
| **faster-whisper** | `WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)`. `transcribe(audio_path)` returns an object with `.segments`. Each segment has `.start`, `.end`, `.text`. |
| **Whisper sample rate** | 16 kHz. Preprocessor should resample if needed. |
| **SRT timestamp** | `HH:MM:SS,mmm` (comma for ms). VTT uses `HH:MM:SS.mmm` (dot). |
| **Typer** | `import typer; app = typer.Typer(); @app.command(); def main(path: str = typer.Argument(...)): ...` |

---

## Suggested order

1. Do **Milestone 1** first (transcriber + minimal script).  
2. Then **Milestone 2** (preprocessor, formatter, CLI, config).  
3. Then **Milestone 3** (medium, VAD, messy audio).  
4. Add **Milestone 4** only if you want a web UI.

This keeps each step testable and avoids building the CLI before the core “audio → segments” path works.
