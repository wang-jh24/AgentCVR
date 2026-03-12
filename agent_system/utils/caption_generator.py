# On-demand Whisper caption generation for video segments (same model/config as reference script).

import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AgentWorkflowLogger")

MODEL_NAME = "large-v3"
WHISPER_TRANSCRIBE_KWARGS = {
    "beam_size": 5,
    "language": "en",
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "initial_prompt": "This is a movie clip with dialogue and background sound.",
}

_cached_whisper_model = None
_cached_whisper_device = None


def _get_whisper_model(device: Optional[str] = None):
    """Lazy-load Whisper model (cached)."""
    global _cached_whisper_model, _cached_whisper_device
    if _cached_whisper_model is not None and (device is None or _cached_whisper_device == device):
        return _cached_whisper_model
    import torch
    import whisper
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"    [Caption] Loading Whisper '{MODEL_NAME}' @ {device} ...")
    _cached_whisper_model = whisper.load_model(MODEL_NAME, device=device)
    _cached_whisper_device = device
    return _cached_whisper_model


def _extract_segment_ffmpeg(video_path: str, start_time: float, end_time: float, out_path: str) -> bool:
    """Extract video segment with ffmpeg; start/end in seconds."""
    duration = end_time - start_time
    if duration <= 0:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        "-loglevel", "error",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"    [Caption] ffmpeg segment failed: {e}")
        return False


def generate_caption_for_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate captions for [start_time, end_time] via Whisper. Returns list of {start, end, text} in absolute time."""
    if not os.path.exists(video_path) and not (video_path.startswith("http://") or video_path.startswith("https://")):
        logger.error(f"    [Caption] Video path missing or not URL: {video_path}")
        return []

    duration = end_time - start_time
    if duration <= 0:
        return []

    tmp_segment = None
    try:
        suffix = ".mp4"
        if video_path.startswith("http://") or video_path.startswith("https://"):
            segment_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
            if not _extract_segment_ffmpeg(video_path, start_time, end_time, segment_path):
                return []
            tmp_segment = segment_path
        else:
            segment_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
            tmp_segment = segment_path
            if not _extract_segment_ffmpeg(video_path, start_time, end_time, segment_path):
                return []

        model = _get_whisper_model(device)
        result = model.transcribe(segment_path, **WHISPER_TRANSCRIBE_KWARGS)

        segments_raw = result.get("segments") or []
        segments = []
        for s in segments_raw:
            seg_start = float(s.get("start", 0))
            seg_end = float(s.get("end", 0))
            text = (s.get("text") or "").strip()
            if not text:
                continue
            segments.append({
                "start": start_time + seg_start,
                "end": start_time + seg_end,
                "text": text,
            })
        return segments
    except Exception as e:
        logger.error(f"    [Caption] Whisper failed: {e}", exc_info=True)
        return []
    finally:
        if tmp_segment and os.path.exists(tmp_segment):
            try:
                os.unlink(tmp_segment)
            except OSError:
                pass
