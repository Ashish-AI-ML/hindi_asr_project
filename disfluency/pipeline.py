"""
Disfluency Detection Pipeline
===============================
End-to-end pipeline: load transcriptions → detect disfluencies → clip audio → generate CSV.

Usage:
    python -m disfluency.pipeline --manifest processed_data/manifests/train_manifest.json

Deliverables (Q2):
- CSV with columns: recording_id, segment_id, start_time, end_time, disfluency_type, transcript_text, audio_clip_path
- Segmented audio clips
- Methodology summary
"""

import os
import sys
import csv
import json
import logging
import argparse
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DISFLUENCY_OUTPUT_DIR, DISFLUENCY_CLIPS_DIR, MANIFEST_DIR,
)
from data.manifest import load_manifest
from disfluency.lexicon import build_disfluency_lexicon
from disfluency.detector import classify_segment_disfluency
from disfluency.clipper import clip_disfluency_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_disfluency_pipeline(
    manifest: List[Dict[str, Any]],
    output_dir: str = DISFLUENCY_OUTPUT_DIR,
    clip_dir: str = DISFLUENCY_CLIPS_DIR,
    do_clip: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the full disfluency detection pipeline.
    
    Args:
        manifest: dataset manifest (list of segment dicts)
        output_dir: directory for CSV and summary output
        clip_dir: directory for audio clips
        do_clip: whether to actually clip audio (set False for text-only analysis)
    
    Returns:
        List of disfluency detection results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    lexicon = build_disfluency_lexicon()
    all_detections = []
    segments_with_disfluency = 0
    total_segments = len(manifest)

    # ── Process each segment ──────────────────────────────────────
    for i, entry in enumerate(manifest):
        segment = {
            "text": entry.get("text", ""),
            "start": entry.get("start"),
            "end": entry.get("end"),
        }

        detections = classify_segment_disfluency(segment, lexicon)

        if detections:
            segments_with_disfluency += 1

            for det in detections:
                recording_id = entry.get("recording_id", f"unknown_{i}")
                segment_idx = entry.get("segment_index", i)

                result = {
                    "recording_id": recording_id,
                    "segment_id": f"{recording_id}_seg{segment_idx}",
                    "start_time": entry.get("start"),
                    "end_time": entry.get("end"),
                    "disfluency_type": det.get("type", "unknown"),
                    "disfluency_category": det.get("category", "unknown"),
                    "detected_text": det.get("word", det.get("phrase", det.get("text", ""))),
                    "transcript_text": entry.get("text", ""),
                    "audio_clip_path": "",
                }

                # Clip audio if requested
                if do_clip and entry.get("audio_path"):
                    clip_path = clip_disfluency_segment(
                        audio_path=entry["audio_path"],
                        segment=segment,
                        recording_id=recording_id,
                        output_dir=clip_dir,
                    )
                    if clip_path:
                        result["audio_clip_path"] = clip_path

                all_detections.append(result)

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i+1}/{total_segments} segments...")

    logger.info(
        f"\nDisfluency detection complete:\n"
        f"  Total segments: {total_segments}\n"
        f"  Segments with disfluency: {segments_with_disfluency}\n"
        f"  Total disfluencies found: {len(all_detections)}"
    )

    return all_detections


def save_results_csv(
    detections: List[Dict[str, Any]],
    output_path: str,
):
    """Save disfluency detections as CSV (deliverable for Q2)."""
    if not detections:
        logger.warning("No detections to save")
        return

    fieldnames = [
        "recording_id", "segment_id", "start_time", "end_time",
        "disfluency_type", "disfluency_category", "detected_text",
        "transcript_text", "audio_clip_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)

    logger.info(f"Saved {len(detections)} detections to {output_path}")


def generate_methodology_summary(
    detections: List[Dict[str, Any]],
    total_segments: int,
) -> str:
    """Generate the methodology summary (deliverable for Q2)."""
    from collections import Counter

    type_counts = Counter(d["disfluency_type"] for d in detections)
    category_counts = Counter(d["disfluency_category"] for d in detections)

    summary = f"""
# Disfluency Detection — Methodology Summary

## Detection Approach
We use a **text-based multi-method approach** to identify speech disfluencies
from timestamped Hindi transcriptions.

### Method 1: Lexicon-Based Filler Detection
- Curated dictionary of Hindi fillers (उम, उह, हम्म), hesitation markers
  (मतलब, actually), and interjection-fillers (अच्छा, ठीक)
- Both Devanagari and Hinglish forms included
- Simple dictionary lookup with case-insensitive matching

### Method 2: Repetition Detection
- **Word repetition**: consecutive identical words (e.g., "मैं मैं")
- **Phrase repetition**: consecutive identical bigrams (e.g., "मैं सोच मैं सोच")
- Uses n-gram comparison with punctuation stripping

### Method 3: Prolongation Detection
- Regex-based detection of characters repeated 3+ times (e.g., "सोोोो")
- Covers Devanagari consonants and vowel signs

### Method 4: False Start Detection
- Heuristic: interruption punctuation ("...", "—")
- Very short utterances (1–2 words) flagged as potential fragments

## Audio Clipping
- Segments are clipped from full recordings using the timestamp boundaries
- A 0.1s buffer is added before and after to avoid sharp cuts
- Output: 16kHz mono WAV files

## Preprocessing Applied
- Unicode NFC normalization on all text
- Punctuation stripping for comparison (not for display)
- Case-insensitive matching for Hinglish terms

## Results Summary
- **Total segments analyzed**: {total_segments}
- **Total disfluencies detected**: {len(detections)}
- **Detection rate**: {len(detections)/max(total_segments,1)*100:.1f}% of segments

### By Type:
"""
    for dtype, count in type_counts.most_common():
        summary += f"- {dtype}: {count}\n"

    summary += "\n### By Category:\n"
    for cat, count in category_counts.most_common():
        summary += f"- {cat}: {count}\n"

    summary += """
## Limitations
- Context-dependent fillers (e.g., "अच्छा" as real word vs filler) may have false positives
- False start detection is heuristic and may miss subtle cases
- No audio-level analysis (e.g., VAD or acoustic features) — text-only approach
- Prolongation detection may flag legitimate words with repeated characters
"""
    return summary


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Disfluency detection pipeline")
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(MANIFEST_DIR, "train_manifest.json"),
    )
    parser.add_argument("--no_clip", action="store_true", help="Skip audio clipping")
    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest(os.path.basename(args.manifest))
    logger.info(f"Loaded {len(manifest)} segments from manifest")

    # Run pipeline
    detections = run_disfluency_pipeline(
        manifest=manifest,
        do_clip=not args.no_clip,
    )

    # Save CSV
    csv_path = os.path.join(DISFLUENCY_OUTPUT_DIR, "disfluency_detections.csv")
    save_results_csv(detections, csv_path)

    # Save methodology
    summary = generate_methodology_summary(detections, len(manifest))
    summary_path = os.path.join(DISFLUENCY_OUTPUT_DIR, "methodology_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Methodology summary saved to {summary_path}")


if __name__ == "__main__":
    main()
