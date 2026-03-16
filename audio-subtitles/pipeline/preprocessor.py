"""
Audio preprocessing: normalize, resample to 16 kHz, optional noise reduction.
Interface: preprocess(audio_path) -> path (temp or original).
"""


def preprocess(audio_path: str) -> str:
    """
    Prepare audio for Whisper. For Milestone 1 you can return the path unchanged.
    Later: resample to 16 kHz mono, normalize loudness, optional noise reduction.
    """
    # TODO Milestone 2: load with librosa/pydub, resample to 16k, save to temp, return temp path
    return audio_path
