"""
Data Download & Validation
============================
Handles downloading audio files and transcription JSONs from GCS
with retry logic, validation, and systematic failure logging.
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Tuple, Dict, List, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    AUDIO_DIR, TRANSCRIPTION_DIR, METADATA_DIR,
    DOWNLOAD_RETRIES, DOWNLOAD_TIMEOUT, DOWNLOAD_BACKOFF,
    GCS_BASE_URL, build_gcs_url,
)

logger = logging.getLogger(__name__)


# ─── Core Download ────────────────────────────────────────────────────

def download_file(
    url: str,
    dest_path: str,
    retries: int = DOWNLOAD_RETRIES,
    timeout: int = DOWNLOAD_TIMEOUT,
    backoff: float = DOWNLOAD_BACKOFF,
) -> bool:
    """
    Download a file from URL to local path with retry + exponential backoff.
    
    Returns True on success, False on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file is non-empty
            if os.path.getsize(dest_path) == 0:
                logger.warning(f"Empty file downloaded from {url}")
                os.remove(dest_path)
                return False

            logger.info(f"Downloaded: {url} -> {dest_path}")
            return True

        except requests.exceptions.RequestException as e:
            wait_time = backoff ** attempt
            logger.warning(
                f"Download attempt {attempt}/{retries} failed for {url}: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            if attempt < retries:
                time.sleep(wait_time)

    logger.error(f"Failed to download after {retries} attempts: {url}")
    return False


# ─── URL Construction ─────────────────────────────────────────────────

def construct_urls(user_id: int, recording_id: int) -> Dict[str, str]:
    """
    Construct all GCS URLs for a given user_id and recording_id.
    
    Returns dict with keys: 'audio', 'transcription', 'metadata'
    """
    return {
        "audio": build_gcs_url(user_id, recording_id, "recording"),
        "transcription": build_gcs_url(user_id, recording_id, "transcription"),
        "metadata": build_gcs_url(user_id, recording_id, "metadata"),
    }


# ─── Transcription Parsing ────────────────────────────────────────────

def parse_transcription(json_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a transcription JSON file.
    
    Expected format: list of dicts with keys:
        - start (float): segment start in seconds
        - end (float): segment end in seconds
        - speaker_id (int): speaker identifier
        - text (str): transcribed text
    
    Returns list of segment dicts, or None if parsing fails.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        if not isinstance(segments, list):
            logger.error(f"Transcription is not a list: {json_path}")
            return None

        valid_segments = []
        for i, seg in enumerate(segments):
            # Validate required keys
            required_keys = {"start", "end", "text"}
            if not required_keys.issubset(seg.keys()):
                logger.warning(
                    f"Segment {i} in {json_path} missing keys: "
                    f"{required_keys - seg.keys()}"
                )
                continue

            # Ensure numeric timestamps
            try:
                seg["start"] = float(seg["start"])
                seg["end"] = float(seg["end"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamps in segment {i} of {json_path}")
                continue

            # Ensure text is a string
            if not isinstance(seg.get("text"), str):
                logger.warning(f"Non-string text in segment {i} of {json_path}")
                continue

            valid_segments.append(seg)

        if not valid_segments:
            logger.warning(f"No valid segments in {json_path}")
            return None

        return valid_segments

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to parse transcription {json_path}: {e}")
        return None


# ─── Download & Validate Pipeline ─────────────────────────────────────

def download_and_validate(
    metadata_row: Dict[str, Any],
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    Download and validate audio + transcription for one recording.
    
    Args:
        metadata_row: dict with keys 'user_id', 'recording_id', 
                       and optionally 'rec_url_gcp', 'transcription_url', 'metadata_url'
    
    Returns:
        Tuple of (audio_path, transcription_segments) on success, None on failure.
    """
    user_id = metadata_row.get("user_id")
    recording_id = metadata_row.get("recording_id")

    if not user_id or not recording_id:
        logger.error(f"Missing user_id or recording_id: {metadata_row}")
        return None

    # Build URLs — use provided URLs or construct from template
    audio_url = metadata_row.get(
        "rec_url_gcp",
        build_gcs_url(user_id, recording_id, "recording"),
    )
    transcription_url = metadata_row.get(
        "transcription_url",
        build_gcs_url(user_id, recording_id, "transcription"),
    )

    # Local paths
    audio_path = os.path.join(AUDIO_DIR, f"{recording_id}.wav")
    transcription_path = os.path.join(
        TRANSCRIPTION_DIR, f"{recording_id}_transcription.json"
    )

    # Download audio (skip if already exists)
    if not os.path.exists(audio_path):
        if not download_file(audio_url, audio_path):
            logger.error(f"Audio download failed for recording {recording_id}")
            return None
    else:
        logger.info(f"Audio already exists: {audio_path}")

    # Download transcription (skip if already exists)
    if not os.path.exists(transcription_path):
        if not download_file(transcription_url, transcription_path):
            logger.error(
                f"Transcription download failed for recording {recording_id}"
            )
            return None
    else:
        logger.info(f"Transcription already exists: {transcription_path}")

    # Parse and validate transcription
    segments = parse_transcription(transcription_path)
    if segments is None:
        return None

    return audio_path, segments


# ─── Batch Download ───────────────────────────────────────────────────

def download_dataset(
    metadata_list: List[Dict[str, Any]],
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Download and validate all recordings in the dataset.
    
    Returns a summary dict with:
        - 'success': list of (audio_path, segments) tuples
        - 'failures': list of recording_ids that failed
        - 'total': total attempted
    """
    results = {"success": [], "failures": [], "total": len(metadata_list)}

    for i, row in enumerate(metadata_list):
        recording_id = row.get("recording_id", f"unknown_{i}")
        logger.info(
            f"Processing {i+1}/{len(metadata_list)}: recording {recording_id}"
        )

        result = download_and_validate(row)
        if result is not None:
            results["success"].append(
                {"recording_id": recording_id, "audio_path": result[0], "segments": result[1]}
            )
        else:
            results["failures"].append(recording_id)

    logger.info(
        f"Download complete: {len(results['success'])}/{results['total']} succeeded, "
        f"{len(results['failures'])} failed."
    )
    return results
