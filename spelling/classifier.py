"""
Hindi Spelling Classifier
===========================
4-layer classification system to identify correct vs incorrect Hindi spellings.

Layer 1: Dictionary lookup (fast, catches obvious correct words)
Layer 2: Morphological heuristics (catches inflections)
Layer 3: Unicode validity (catches impossible character sequences)
Layer 4: Edit distance + frequency (for ambiguous cases)

Special case: English words in Devanagari → correct per guidelines
"""

import re
import logging
from typing import Dict, Set, Tuple, List, Optional
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.text_utils import normalize_unicode, is_devanagari
from spelling.lexicon_loader import (
    load_hindi_lexicon, is_transliterated_english, load_english_hindi_map,
)
from spelling.unicode_validator import check_unicode_validity, is_pure_devanagari

logger = logging.getLogger(__name__)


# ─── Common Hindi Suffixes (for morphological analysis) ───────────────
# Hindi inflects heavily — valid root + valid suffix = likely correct

HINDI_SUFFIXES = [
    # Verb suffixes
    "ता", "ती", "ते", "ना", "नी", "ने",
    "या", "यी", "ये", "ा", "ी", "े",
    "ूँ", "ूं", "ें", "ो", "ओ",
    "ाना", "ाती", "ाते",
    "ेगा", "ेगी", "ेंगे", "ेंगी",
    "कर", "कार",
    # Noun/adjective suffixes  
    "ों", "ियों", "ियाँ", "ियां",
    "वाला", "वाली", "वाले",
    "पन", "पना", "आई", "आहट",
    "दार", "कार", "गार",
    # Postposition clitics
    "को", "से", "में", "पर", "ने", "का", "की", "के",
]


def check_morphological_validity(
    word: str, lexicon: Set[str]
) -> Tuple[bool, str]:
    """
    Layer 2: Check if word is a valid inflection of a known root.
    
    Tries stripping common Hindi suffixes and checking if the remaining
    root is in the lexicon.
    
    Returns (is_likely_valid, explanation)
    """
    word = normalize_unicode(word)

    # Sort suffixes by length (try longer first for greedy match)
    sorted_suffixes = sorted(HINDI_SUFFIXES, key=len, reverse=True)

    for suffix in sorted_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            root = word[: -len(suffix)]
            if root in lexicon and len(root) >= 2:
                return True, f"Root '{root}' + suffix '{suffix}'"

    return False, "No valid morphological decomposition found"


def compute_edit_distance_candidates(
    word: str,
    lexicon: Set[str],
    max_distance: int = 2,
    max_candidates: int = 5,
) -> List[Tuple[str, int]]:
    """
    Layer 4: Find closest valid words by edit distance.
    
    Uses rapidfuzz for fast fuzzy matching. If edit distance = 1,
    the word is likely a typo.
    
    Returns list of (candidate_word, distance) sorted by distance.
    """
    try:
        from rapidfuzz import fuzz, process
        from rapidfuzz.distance import Levenshtein

        # For performance, limit search to words of similar length
        word_len = len(word)
        candidates = [
            w for w in lexicon
            if abs(len(w) - word_len) <= max_distance
        ]

        if not candidates:
            return []

        # Compute distances
        results = []
        for candidate in candidates:
            dist = Levenshtein.distance(word, candidate)
            if dist <= max_distance:
                results.append((candidate, dist))

        # Sort by distance, then alphabetically
        results.sort(key=lambda x: (x[1], x[0]))
        return results[:max_candidates]

    except ImportError:
        logger.warning("rapidfuzz not installed; edit distance check skipped")
        return []


# ─── Master Classifier ───────────────────────────────────────────────

def classify_word(
    word: str,
    lexicon: Set[str],
    freq_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """
    Classify a word as correct, incorrect, or uncertain.
    
    4-layer cascade:
    1. Dictionary lookup → if found, CORRECT
    2. English transliteration check → if matched, CORRECT  
    3. Morphological analysis → if valid decomposition, CORRECT
    4. Unicode validity → if invalid sequences, INCORRECT
    5. Edit distance → if distance=1 to valid word, likely INCORRECT (typo)
    6. Otherwise → UNCERTAIN
    
    Args:
        word: Hindi word to classify
        lexicon: set of known valid words
        freq_dict: optional word frequency dictionary
    
    Returns:
        {
            "word": str,
            "classification": "correct" | "incorrect" | "uncertain",
            "reason": str,
            "layer": int,  # which layer made the decision
            "suggestion": str,  # suggested correction if applicable
        }
    """
    word = normalize_unicode(word.strip())

    if not word:
        return {"word": "", "classification": "incorrect", "reason": "Empty word", "layer": 0, "suggestion": ""}

    result = {
        "word": word,
        "classification": "uncertain",
        "reason": "",
        "layer": 0,
        "suggestion": "",
    }

    # ── Layer 1: Dictionary lookup ────────────────────────────────
    if word in lexicon:
        result["classification"] = "correct"
        result["reason"] = "Found in dictionary"
        result["layer"] = 1
        return result

    # ── Layer 1.5: English transliteration ────────────────────────
    if is_transliterated_english(word):
        result["classification"] = "correct"
        result["reason"] = f"English transliteration: {load_english_hindi_map().get(word, '???')}"
        result["layer"] = 1
        return result

    # ── Layer 2: Morphological analysis ───────────────────────────
    morph_valid, morph_reason = check_morphological_validity(word, lexicon)
    if morph_valid:
        result["classification"] = "correct"
        result["reason"] = f"Morphological: {morph_reason}"
        result["layer"] = 2
        return result

    # ── Layer 3: Unicode validity ─────────────────────────────────
    if is_pure_devanagari(word):
        unicode_valid, unicode_reason = check_unicode_validity(word)
        if not unicode_valid:
            result["classification"] = "incorrect"
            result["reason"] = f"Invalid Unicode: {unicode_reason}"
            result["layer"] = 3
            return result

    # ── Layer 4: Edit distance ────────────────────────────────────
    candidates = compute_edit_distance_candidates(word, lexicon, max_distance=1)
    if candidates:
        closest_word, distance = candidates[0]
        if distance == 1:
            result["classification"] = "incorrect"
            result["reason"] = f"Likely typo (edit distance 1 from '{closest_word}')"
            result["suggestion"] = closest_word
            result["layer"] = 4
            return result

    # ── Layer 4.5: Frequency analysis ─────────────────────────────
    if freq_dict and word in freq_dict:
        freq = freq_dict[word]
        if freq >= 5:
            # Appears multiple times — more likely correct even if not in dict
            result["classification"] = "correct"
            result["reason"] = f"High frequency ({freq} occurrences) — likely valid"
            result["layer"] = 4
            return result

    # ── Fallback: Uncertain ───────────────────────────────────────
    # Check edit distance = 2 for softer signal
    candidates_2 = compute_edit_distance_candidates(word, lexicon, max_distance=2)
    if candidates_2:
        closest_word, distance = candidates_2[0]
        result["suggestion"] = closest_word
        if distance == 2:
            result["classification"] = "uncertain"
            result["reason"] = f"Near '{closest_word}' (edit distance {distance})"
        else:
            result["classification"] = "uncertain"
            result["reason"] = "Not in dictionary, no close matches"
    else:
        result["classification"] = "uncertain"
        result["reason"] = "Not in dictionary, unknown word"

    result["layer"] = 5
    return result


def build_frequency_dict(words: List[str]) -> Dict[str, int]:
    """Build a frequency dictionary from a word list."""
    return dict(Counter(normalize_unicode(w.strip()) for w in words if w.strip()))
