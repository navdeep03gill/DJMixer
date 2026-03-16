"""
Convert list[Segment] to SRT, VTT, or plain text.
"""

from .transcriber import Segment


def _ts_srt(seconds: float) -> str:
    """SRT: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    """VTT: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_segments(segments: list[Segment], fmt: str) -> str:
    """fmt: 'srt' | 'vtt' | 'txt'."""
    if fmt == "srt":
        return _to_srt(segments)
    if fmt == "vtt":
        return _to_vtt(segments)
    if fmt == "txt":
        return _to_txt(segments)
    raise ValueError(f"Unknown format: {fmt}")


def _to_srt(segments: list[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n{seg.text}\n"
        )
    return "\n".join(blocks)


def _to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _to_txt(segments: list[Segment]) -> str:
    return "\n".join(seg.text for seg in segments)
