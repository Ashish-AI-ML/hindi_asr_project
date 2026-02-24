"""
Disfluency Audio Clipper
=========================
Clips audio segments from full recordings based on detected disfluencies.
Each clip corresponds to a segment with timestamps where disfluency was found.
"""

import os
import logging
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.audio_utils import clip_audio
from config import DISFLUENCY_CLIPS_DIR, CLIP_BUFFER_SEC

logger = logging.getLogger(__name__)


def clip_disfluency_segment(
    audio_path: str,
    segment: Dict[str, Any],
    recording_id: str,
    output_dir: str = DISFLUENCY_CLIPS_DIR,
    buffer_sec: float = CLIP_BUFFER_SEC,
) -> Optional[str]:
    """
    Clip the audio for a segment where disfluency was detected.
    
    Args:
        audio_path: path to the full recording audio
        segment: dict with 'start' and 'end' keys (in seconds)
        recording_id: identifier for the recording
        output_dir: directory to save clips
        buffer_sec: padding before/after clip
    
    Returns:
        Path to saved clip, or None on failure.
    
    File naming: {recording_id}_seg_{start:.2f}_{end:.2f}.wav
    """
    start = segment.get("start")
    end = segment.get("end")

    if start is None or end is None:
        logger.error(f"Segment missing timestamps: {segment}")
        return None

    if end <= start:
        logger.error(f"Invalid segment: end ({end}) <= start ({start})")
        return None

    # Create output filename
    clip_filename = f"{recording_id}_seg_{start:.2f}_{end:.2f}.wav"
    output_path = os.path.join(output_dir, clip_filename)

    # Skip if already clipped
    if os.path.exists(output_path):
        logger.info(f"Clip already exists: {output_path}")
        return output_path

    # Clip audio
    result = clip_audio(
        audio_path=audio_path,
        start_sec=start,
        end_sec=end,
        output_path=output_path,
        buffer_sec=buffer_sec,
    )

    if result:
        logger.info(f"Clipped disfluency segment: {output_path}")
    else:
        logger.error(f"Failed to clip: {audio_path} [{start}-{end}]")

    return result


def batch_clip_disfluencies(
    disfluencies: list,
    output_dir: str = DISFLUENCY_CLIPS_DIR,
) -> list:
    """
    Clip audio for all detected disfluencies.
    
    Args:
        disfluencies: list of dicts with 'audio_path', 'recording_id',
                      'segment_start', 'segment_end'
        output_dir: base directory for clips
    
    Returns:
        List of successful clip paths
    """
    os.makedirs(output_dir, exist_ok=True)
    clips = []

    for i, disf in enumerate(disfluencies):
        audio_path = disf.get("audio_path")
        recording_id = disf.get("recording_id", f"unknown_{i}")
        
        segment = {
            "start": disf.get("segment_start"),
            "end": disf.get("segment_end"),
        }

        clip_path = clip_disfluency_segment(
            audio_path=audio_path,
            segment=segment,
            recording_id=recording_id,
            output_dir=output_dir,
        )

        if clip_path:
            disf["clip_path"] = clip_path
            clips.append(clip_path)

    logger.info(f"Successfully clipped {len(clips)}/{len(disfluencies)} segments")
    return clips
