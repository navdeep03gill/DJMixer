"""
Default configuration for the transcription pipeline.
Override via CLI options or environment if needed.
"""

# Model
model_size: str = "small"   # small | medium | large-v3
device: str = "cpu"
compute_type: str = "int8"  # int8 recommended on Apple Silicon
cpu_threads: int = 8

# Audio
sample_rate: int = 16000    # Whisper expects 16 kHz

# Output
output_format: str = "srt"  # srt | vtt | txt

# Inference
vad_filter: bool = False    # Silero VAD — reduces hallucinations on noisy audio

# Preprocessing
separate_vocals: bool = False     # Demucs vocal separation (enable for music)
noise_reduction: bool = False     # Stationary noise gate via noisereduce
loudness_target_lufs: float = -16.0  # EBU R128 target; -16 suits speech/Whisper

# Demucs
demucs_model: str = "htdemucs"   # htdemucs | htdemucs_ft (fine-tuned variant)
