"""
Transcriber abstraction: transcribe(audio_path) -> list[Segment].
Implement WhisperTranscriber first; swap in AssemblyAI or custom model later.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Segment:
    """One transcribed segment with time bounds and text."""
    start: float  # seconds
    end: float
    text: str


class Transcriber(Protocol):
    """Interface for any transcription backend."""

    def transcribe(self, audio_path: str) -> list[Segment]:
        """Return ordered segments for the given audio file."""
        ...


class WhisperTranscriber:
    """faster-whisper implementation. Use CPU + int8 on Apple Silicon."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 8,
        vad_filter: bool = False,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.vad_filter = vad_filter
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe(self, audio_path: str) -> list[Segment]:
        model = self._get_model()
        segments_raw, _ = model.transcribe(
            audio_path,
            vad_filter=self.vad_filter,
        )
        return [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_raw
        ]
