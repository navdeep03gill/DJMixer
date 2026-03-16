"""
CLI entry point. Usage:
  python cli.py <audio_path> [--output out.srt] [--format srt|vtt|txt] [--no-preprocess]
"""

import sys
from pathlib import Path

import typer

# Add project root so pipeline and config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.formatter import format_segments
from pipeline.preprocessor import preprocess
from pipeline.transcriber import WhisperTranscriber

app = typer.Typer(help="Transcribe audio to SRT, VTT, or plain text.")


@app.command()
def main(
    audio_path: Path = typer.Argument(..., help="Input audio file (e.g. MP3, WAV)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    format_name: str = typer.Option("srt", "--format", "-f", help="Output format: srt, vtt, txt"),
    no_preprocess: bool = typer.Option(False, "--no-preprocess", help="Skip preprocessing"),
) -> None:
    """Transcribe audio and write segments in the chosen format."""
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        typer.echo(f"Error: file not found: {audio_path}", err=True)
        raise typer.Exit(1)

    path_to_use = preprocess(str(audio_path)) if not no_preprocess else str(audio_path)

    transcriber = WhisperTranscriber(
        model_size=config.model_size,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
        vad_filter=config.vad_filter,
    )
    segments = transcriber.transcribe(path_to_use)
    text = format_segments(segments, format_name)

    if output is not None:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {len(segments)} segments to {output}")
    else:
        print(text)


if __name__ == "__main__":
    app()
