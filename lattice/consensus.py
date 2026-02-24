"""
Consensus & Reference Correction
===================================
Uses model agreement to identify and correct potentially wrong reference words.

Core insight: if ≥ K out of N models agree on a word that differs from the
reference, the reference might be wrong. This is ROVER-inspired reference
correction for fairer WER evaluation.
"""

import logging
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONSENSUS_THRESHOLD, NUM_ASR_MODELS
from lattice.alignment import EPSILON

logger = logging.getLogger(__name__)


def compute_consensus(
    model_words: List[str],
    threshold: int = CONSENSUS_THRESHOLD,
) -> Tuple[Optional[str], int, str]:
    """
    Compute consensus word at a single lattice position.
    
    Args:
        model_words: list of words from each model at this position
        threshold: minimum number of models that must agree
    
    Returns:
        (consensus_word, agreement_count, decision_reason)
        
        consensus_word is None if no word meets the threshold.
    """
    # Filter out EPSILON (model had nothing at this position)
    non_epsilon = [w for w in model_words if w != EPSILON]

    if not non_epsilon:
        return None, 0, "All models have insertions/deletions at this position"

    # Count votes
    word_counts = Counter(non_epsilon)
    most_common_word, most_common_count = word_counts.most_common(1)[0]

    if most_common_count >= threshold:
        return (
            most_common_word,
            most_common_count,
            f"Consensus: {most_common_count}/{len(model_words)} models agree on '{most_common_word}'",
        )

    return (
        None,
        most_common_count,
        f"No consensus: highest agreement is {most_common_count}/{len(model_words)} "
        f"(threshold={threshold})",
    )


def correct_reference(
    reference: List[str],
    lattice: Dict[int, Dict[str, Any]],
    threshold: int = CONSENSUS_THRESHOLD,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build a corrected reference using model consensus.
    
    At each reference position:
    - If consensus agrees WITH reference → keep reference word
    - If consensus disagrees with reference AND meets threshold → replace with consensus
    - If no consensus → keep reference word (benefit of the doubt)
    
    Args:
        reference: original reference as list of words
        lattice: position lattice from build_position_lattice()
        threshold: consensus threshold
    
    Returns:
        (corrected_reference, correction_log)
        
        correction_log: list of dicts documenting each correction
    """
    corrected = list(reference)
    correction_log = []

    for pos in sorted(
        [k for k in lattice.keys() if isinstance(k, int)],
    ):
        data = lattice.get(pos)
        if data is None:
            continue

        ref_word = data["ref_word"]
        model_words = data["model_words"]

        if ref_word == EPSILON:
            continue  # Skip insertion positions

        consensus_word, agreement_count, reason = compute_consensus(
            model_words, threshold
        )

        if consensus_word is not None and consensus_word != ref_word:
            # Models agree on something different from reference
            correction_log.append({
                "position": pos,
                "original_ref": ref_word,
                "corrected_to": consensus_word,
                "agreement": f"{agreement_count}/{len(model_words)}",
                "reason": reason,
                "model_outputs": model_words,
            })

            if pos < len(corrected):
                corrected[pos] = consensus_word

            logger.info(
                f"Position {pos}: '{ref_word}' → '{consensus_word}' "
                f"({agreement_count}/{len(model_words)} models agree)"
            )

    logger.info(
        f"Reference correction: {len(correction_log)} words corrected "
        f"out of {len(reference)} total"
    )

    return corrected, correction_log


def analyze_corrections(
    correction_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze the corrections made to understand patterns.
    
    Returns summary statistics about the correction process.
    """
    if not correction_log:
        return {
            "total_corrections": 0,
            "correction_rate": "0%",
        }

    agreement_levels = Counter()
    for entry in correction_log:
        agreement_levels[entry["agreement"]] += 1

    return {
        "total_corrections": len(correction_log),
        "agreement_distribution": dict(agreement_levels),
        "examples": correction_log[:10],  # first 10 for inspection
    }
