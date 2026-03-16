"""
Default configuration for the transcription pipeline.
Override via CLI options or environment if needed.
"""

# Model
model_size: str = "small"  # small | medium | large-v2
device: str = "cpu"
compute_type: str = "int8"  # int8 recommended on Apple Silicon
cpu_threads: int = 8

# Audio
sample_rate: int = 16000  # Whisper expects 16 kHz

# Output
output_format: str = "srt"  # srt | vtt | txt

# Inference options (Milestone 3)
vad_filter: bool = False  # set True to reduce hallucinations on noisy audio
