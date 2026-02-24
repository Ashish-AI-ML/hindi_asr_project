"""
Position Lattice Builder
==========================
Constructs a lattice that captures all valid transcription alternatives
from multiple ASR model outputs, aligned to the reference.

A lattice is a graph where:
- Each node = a position in the transcript
- Each edge = a word that could appear at that position
- Multiple parallel edges = different models' hypotheses

This is essentially ROVER (Recognizer Output Voting Error Reduction).
"""

import logging
from typing import Dict, List, Any, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lattice.alignment import (
    align_hypothesis_to_reference,
    EPSILON, CORRECT, SUBSTITUTION, INSERTION, DELETION,
)

logger = logging.getLogger(__name__)


def build_position_lattice(
    reference: List[str],
    all_hypotheses: List[List[str]],
) -> Dict[int, Dict[str, Any]]:
    """
    Build a position lattice from reference + multiple model hypotheses.
    
    Args:
        reference: reference sentence as list of words
        all_hypotheses: list of hypothesis sentences (one per model),
                        each as a list of words
    
    Returns:
        Dict mapping position → {
            "ref_word": str,
            "model_words": [word_from_model_1, word_from_model_2, ...],
            "operations": [operation_type_1, ...],
        }
    
    Positions are defined by the reference alignment backbone.
    Insertions create intermediate positions.
    """
    num_models = len(all_hypotheses)
    lattice = {}

    # Align each hypothesis to reference
    alignments = []
    for i, hyp in enumerate(all_hypotheses):
        alignment = align_hypothesis_to_reference(reference, hyp)
        alignments.append(alignment)
        logger.debug(f"Model {i+1} alignment length: {len(alignment)}")

    # Build unified lattice
    # Strategy: use the reference as the backbone, then overlay each model's output
    # at the corresponding position.

    # First, number positions based on reference words
    # For each alignment, map ref positions to model outputs
    for model_idx, alignment in enumerate(alignments):
        ref_pos = 0
        for ref_word, hyp_word, operation in alignment:
            if operation in (CORRECT, SUBSTITUTION, DELETION):
                # This corresponds to a reference position
                if ref_pos not in lattice:
                    lattice[ref_pos] = {
                        "ref_word": ref_word,
                        "model_words": [None] * num_models,
                        "operations": [None] * num_models,
                    }

                if operation == CORRECT:
                    lattice[ref_pos]["model_words"][model_idx] = hyp_word
                    lattice[ref_pos]["operations"][model_idx] = CORRECT
                elif operation == SUBSTITUTION:
                    lattice[ref_pos]["model_words"][model_idx] = hyp_word
                    lattice[ref_pos]["operations"][model_idx] = SUBSTITUTION
                elif operation == DELETION:
                    lattice[ref_pos]["model_words"][model_idx] = EPSILON
                    lattice[ref_pos]["operations"][model_idx] = DELETION

                ref_pos += 1

            elif operation == INSERTION:
                # Model has an extra word not in reference
                # Use a fractional position (e.g., 2.1, 2.2) to represent insertions
                ins_pos = ref_pos - 0.5  # before current position
                insert_key = f"{ref_pos}_ins_{model_idx}"
                
                # Track insertions separately — they're model-specific
                if insert_key not in lattice:
                    lattice[insert_key] = {
                        "ref_word": EPSILON,
                        "model_words": [EPSILON] * num_models,
                        "operations": [None] * num_models,
                    }
                lattice[insert_key]["model_words"][model_idx] = hyp_word
                lattice[insert_key]["operations"][model_idx] = INSERTION

    # Fill in missing model entries (models that didn't produce output at this position)
    for pos, data in lattice.items():
        for i in range(num_models):
            if data["model_words"][i] is None:
                data["model_words"][i] = EPSILON
                data["operations"][i] = DELETION

    logger.info(f"Lattice built: {len(lattice)} positions from {num_models} models")
    return lattice


def lattice_to_readable(
    lattice: Dict[int, Dict[str, Any]],
) -> str:
    """
    Convert lattice to a human-readable string representation.
    
    Useful for debugging and methodology documentation.
    """
    lines = []
    lines.append("Position | Reference | Model Outputs | Operations")
    lines.append("-" * 70)

    for pos in sorted(lattice.keys(), key=lambda x: float(str(x).split("_")[0])):
        data = lattice[pos]
        ref = data["ref_word"]
        models = data["model_words"]
        ops = data["operations"]

        model_str = " | ".join(
            f"{w}" for w in models
        )
        ops_str = " | ".join(
            str(o)[:3] if o else "---" for o in ops
        )
        lines.append(f"{str(pos):>8s} | {ref:<12s} | {model_str} | {ops_str}")

    return "\n".join(lines)
