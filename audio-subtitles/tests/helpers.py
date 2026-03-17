"""
Shared test utilities — WER/CER normalization, fixture loading.
"""
import re

import jiwer


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate after normalization."""
    return jiwer.wer(normalize(reference), normalize(hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate after normalization."""
    return jiwer.cer(normalize(reference), normalize(hypothesis))


def segments_to_text(segments) -> str:
    return " ".join(s.text.strip() for s in segments)
