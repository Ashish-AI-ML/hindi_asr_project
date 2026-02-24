"""
Dataset Manifest Builder
=========================
Validates segments, builds training manifest, and generates data health reports.
The manifest is the "table of contents" for the entire dataset.
"""

import os
import json
import logging
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION, MIN_TEXT_LENGTH,
    MANIFEST_DIR, SAMPLE_RATE,
)
from data.audio_utils import get_duration
from data.text_utils import clean_text

logger = logging.getLogger(__name__)


# ─── Segment Validation ──────────────────────────────────────────────

def validate_segment(
    segment: Dict[str, Any],
    audio_duration: float,
) -> Tuple[bool, str]:
    """
    Validate a single transcription segment.
    
    Checks:
    - start < end
    - start >= 0
    - end <= audio_duration (with 1s tolerance)
    - segment duration within [MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION]
    - text is non-empty and meets minimum length
    
    Returns (is_valid, reason).
    """
    start = segment.get("start", -1)
    end = segment.get("end", -1)
    text = segment.get("text", "")

    # Timestamp checks
    if start < 0:
        return False, f"Negative start time: {start}"

    if end <= start:
        return False, f"end ({end}) <= start ({start})"

    if end > audio_duration + 1.0:  # 1s tolerance for rounding
        return False, f"end ({end}) > audio_duration ({audio_duration})"

    # Duration checks
    seg_duration = end - start
    if seg_duration < MIN_SEGMENT_DURATION:
        return False, f"Too short: {seg_duration:.2f}s < {MIN_SEGMENT_DURATION}s"

    if seg_duration > MAX_SEGMENT_DURATION:
        return False, f"Too long: {seg_duration:.2f}s > {MAX_SEGMENT_DURATION}s"

    # Text checks
    cleaned = clean_text(text)
    if len(cleaned) < MIN_TEXT_LENGTH:
        return False, f"Text too short after cleaning: '{cleaned}'"

    return True, "OK"


# ─── Manifest Builder ────────────────────────────────────────────────

def build_dataset_manifest(
    download_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a clean dataset manifest from downloaded recordings.
    
    Args:
        download_results: list of dicts with keys:
            'recording_id', 'audio_path', 'segments'
    
    Returns:
        (valid_entries, rejected_entries) where each entry is:
        {
            "audio_path": str,
            "text": str,
            "duration": float,
            "start": float,
            "end": float,
            "recording_id": str/int,
            "segment_index": int,
        }
    """
    valid_entries = []
    rejected_entries = []

    for record in download_results:
        recording_id = record["recording_id"]
        audio_path = record["audio_path"]
        segments = record["segments"]

        # Get audio duration for validation
        audio_dur = get_duration(audio_path)
        if audio_dur <= 0:
            logger.warning(f"Cannot get duration for {audio_path}, skipping")
            for i, seg in enumerate(segments):
                rejected_entries.append({
                    "recording_id": recording_id,
                    "segment_index": i,
                    "reason": "Cannot determine audio duration",
                })
            continue

        for i, seg in enumerate(segments):
            is_valid, reason = validate_segment(seg, audio_dur)

            if is_valid:
                valid_entries.append({
                    "audio_path": audio_path,
                    "text": clean_text(seg["text"]),
                    "duration": seg["end"] - seg["start"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "recording_id": recording_id,
                    "segment_index": i,
                })
            else:
                rejected_entries.append({
                    "recording_id": recording_id,
                    "segment_index": i,
                    "reason": reason,
                    "original_text": seg.get("text", ""),
                })

    logger.info(
        f"Manifest: {len(valid_entries)} valid, {len(rejected_entries)} rejected "
        f"out of {len(valid_entries) + len(rejected_entries)} total segments"
    )

    return valid_entries, rejected_entries


def save_manifest(manifest: List[Dict[str, Any]], filename: str = "train_manifest.json"):
    """Save manifest to JSON file."""
    path = os.path.join(MANIFEST_DIR, filename)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved to {path} ({len(manifest)} entries)")
    return path


def load_manifest(filename: str = "train_manifest.json") -> List[Dict[str, Any]]:
    """Load manifest from JSON file."""
    path = os.path.join(MANIFEST_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Data Health Report ──────────────────────────────────────────────

def generate_data_health_report(
    valid_entries: List[Dict[str, Any]],
    rejected_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a comprehensive data health report.
    
    Returns a dict with statistics about the dataset quality.
    """
    total = len(valid_entries) + len(rejected_entries)
    durations = [e["duration"] for e in valid_entries]
    total_hours = sum(durations) / 3600 if durations else 0

    # Duration distribution buckets
    duration_buckets = Counter()
    for d in durations:
        if d < 1:
            duration_buckets["<1s"] += 1
        elif d < 5:
            duration_buckets["1-5s"] += 1
        elif d < 10:
            duration_buckets["5-10s"] += 1
        elif d < 20:
            duration_buckets["10-20s"] += 1
        elif d < 30:
            duration_buckets["20-30s"] += 1
        else:
            duration_buckets[">30s"] += 1

    # Rejection reasons
    rejection_reasons = Counter(e.get("reason", "unknown") for e in rejected_entries)

    # Text statistics
    word_counts = [len(e["text"].split()) for e in valid_entries]
    
    report = {
        "total_segments_processed": total,
        "valid_segments": len(valid_entries),
        "rejected_segments": len(rejected_entries),
        "acceptance_rate": f"{len(valid_entries)/total*100:.1f}%" if total > 0 else "N/A",
        "total_audio_hours": f"{total_hours:.2f}",
        "duration_stats": {
            "min_seconds": f"{min(durations):.2f}" if durations else "N/A",
            "max_seconds": f"{max(durations):.2f}" if durations else "N/A",
            "mean_seconds": f"{sum(durations)/len(durations):.2f}" if durations else "N/A",
            "median_seconds": f"{sorted(durations)[len(durations)//2]:.2f}" if durations else "N/A",
        },
        "duration_distribution": dict(duration_buckets),
        "text_stats": {
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "mean_words": f"{sum(word_counts)/len(word_counts):.1f}" if word_counts else "N/A",
        },
        "rejection_reasons": dict(rejection_reasons),
        "unique_recordings": len(set(e["recording_id"] for e in valid_entries)),
    }

    return report


def print_health_report(report: Dict[str, Any]):
    """Pretty-print the data health report."""
    print("\n" + "=" * 60)
    print("       📊 DATA HEALTH REPORT")
    print("=" * 60)
    print(f"  Total segments processed:  {report['total_segments_processed']}")
    print(f"  ✅ Valid segments:          {report['valid_segments']}")
    print(f"  ❌ Rejected segments:       {report['rejected_segments']}")
    print(f"  Acceptance rate:           {report['acceptance_rate']}")
    print(f"  Total audio (hours):       {report['total_audio_hours']}")
    print(f"  Unique recordings:         {report['unique_recordings']}")
    print()
    print("  📐 Duration Statistics:")
    for k, v in report["duration_stats"].items():
        print(f"    {k}: {v}")
    print()
    print("  📊 Duration Distribution:")
    for bucket, count in sorted(report["duration_distribution"].items()):
        bar = "█" * min(count, 50)
        print(f"    {bucket:>8s}: {count:>5d}  {bar}")
    print()
    print("  📝 Text Statistics:")
    for k, v in report["text_stats"].items():
        print(f"    {k}: {v}")
    print()
    if report["rejection_reasons"]:
        print("  ⚠️  Rejection Reasons:")
        for reason, count in sorted(
            report["rejection_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"    {count:>5d}  {reason}")
    print("=" * 60 + "\n")
