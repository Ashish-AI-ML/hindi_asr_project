"""
Disfluency Detector
====================
Text-based detection of speech disfluencies in Hindi transcriptions.

Detection methods:
1. Lexicon-based filler detection (dictionary lookup)
2. Word/phrase repetition detection (n-gram comparison)
3. Prolongation detection (character repetition regex)
4. False start detection (sentence fragments)
"""

import re
import logging
from typing import Dict, List, Any, Set, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.text_utils import normalize_unicode

logger = logging.getLogger(__name__)


# ─── Filler Detection ────────────────────────────────────────────────

def detect_fillers(
    text: str,
    lexicon: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """
    Detect filler words and hesitation markers in text.
    
    Args:
        text: segment text
        lexicon: output of build_disfluency_lexicon()
    
    Returns:
        List of detections: [{"type": "filler", "word": "उम", "position": 2, "category": "filler"}, ...]
    """
    if not text or not text.strip():
        return []

    text = normalize_unicode(text)
    words = text.strip().split()
    detections = []

    for i, word in enumerate(words):
        word_lower = word.lower().strip()
        word_clean = re.sub(r'[।,\.\?\!]', '', word_lower)

        for category, markers in lexicon.items():
            if word_clean in markers or word_lower in markers:
                detections.append({
                    "type": "filler",
                    "word": word,
                    "position": i,
                    "category": category,
                })
                break  # avoid double-counting

    return detections


# ─── Repetition Detection ────────────────────────────────────────────

def detect_repetitions(text: str) -> List[Dict[str, Any]]:
    """
    Detect word and phrase repetitions in text.
    
    Types detected:
    - Single word repetition: "मैं मैं" (I I)
    - Bigram repetition: "मैं सोच मैं सोच" (I think I think)
    
    Returns list of detections with type and positions.
    """
    if not text or not text.strip():
        return []

    text = normalize_unicode(text)
    words = text.strip().split()
    detections = []

    # ── Single word repetition ────────────────────────────────────
    for i in range(len(words) - 1):
        w1 = re.sub(r'[।,\.\?\!]', '', words[i].lower())
        w2 = re.sub(r'[।,\.\?\!]', '', words[i + 1].lower())

        if w1 == w2 and len(w1) > 1:  # avoid single-char matches
            detections.append({
                "type": "word_repetition",
                "word": words[i],
                "positions": [i, i + 1],
                "category": "repetition",
            })

    # ── Bigram repetition ─────────────────────────────────────────
    if len(words) >= 4:
        for i in range(len(words) - 3):
            bigram1 = f"{words[i].lower()} {words[i+1].lower()}"
            bigram2 = f"{words[i+2].lower()} {words[i+3].lower()}"
            bigram1_clean = re.sub(r'[।,\.\?\!]', '', bigram1)
            bigram2_clean = re.sub(r'[।,\.\?\!]', '', bigram2)

            if bigram1_clean == bigram2_clean:
                detections.append({
                    "type": "phrase_repetition",
                    "phrase": f"{words[i]} {words[i+1]}",
                    "positions": [i, i + 1, i + 2, i + 3],
                    "category": "repetition",
                })

    return detections


# ─── Prolongation Detection ──────────────────────────────────────────

# Regex for character repeated 3+ times (e.g., सोोोो, हाааा)
# Covers both Devanagari vowel marks and base characters
PROLONGATION_PATTERN = re.compile(r'(.)\1{2,}')

# Devanagari vowel signs (matras) that are commonly prolonged
DEVANAGARI_VOWEL_SIGNS = set('ा ि ी ु ू े ै ो ौ ं ः ँ'.split())


def detect_prolongations(text: str) -> List[Dict[str, Any]]:
    """
    Detect prolonged sounds in text (e.g., "सोोोो" or "नहीींं").
    
    Uses regex to find any character repeated 3+ times.
    
    Caution: Some legitimate Hindi words have repeated characters.
    This detector flags candidates; false positives are possible.
    """
    if not text or not text.strip():
        return []

    text = normalize_unicode(text)
    words = text.strip().split()
    detections = []

    for i, word in enumerate(words):
        matches = PROLONGATION_PATTERN.finditer(word)
        for match in matches:
            repeated_char = match.group(1)
            repeat_count = len(match.group(0))

            detections.append({
                "type": "prolongation",
                "word": word,
                "position": i,
                "character": repeated_char,
                "repeat_count": repeat_count,
                "category": "prolongation",
            })

    return detections


# ─── False Start Detection ───────────────────────────────────────────

def detect_false_starts(text: str) -> List[Dict[str, Any]]:
    """
    Detect potential false starts (abandoned sentences).
    
    Heuristics:
    - Very short segments (1-2 words) followed by longer ones
    - Text ending with "..." or em-dash indicating interruption
    - Sentence fragments without a verb
    
    This is the hardest disfluency to detect from text alone.
    """
    if not text or not text.strip():
        return []

    detections = []

    # Check for explicit interruption markers
    if re.search(r'\.\.\.|—|–|\.\.', text):
        detections.append({
            "type": "false_start",
            "text": text,
            "marker": "interruption_punctuation",
            "category": "false_start",
        })

    # Very short text (likely abandoned utterance)
    words = text.strip().split()
    if 1 <= len(words) <= 2:
        detections.append({
            "type": "false_start",
            "text": text,
            "marker": "very_short_utterance",
            "category": "false_start",
        })

    return detections


# ─── Master Classifier ───────────────────────────────────────────────

def classify_segment_disfluency(
    segment: Dict[str, Any],
    lexicon: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """
    Master function: classify all disfluencies in a single segment.
    
    Args:
        segment: dict with at least "text" key (and optionally "start", "end")
        lexicon: disfluency lexicon from build_disfluency_lexicon()
    
    Returns:
        List of all detected disfluencies, each with:
        - type: filler | word_repetition | phrase_repetition | prolongation | false_start
        - category: broader category
        - word/phrase: the detected text
        - various detection-specific fields
    """
    text = segment.get("text", "")
    if not text or not text.strip():
        return []

    all_detections = []

    # Run all detectors
    fillers = detect_fillers(text, lexicon)
    repetitions = detect_repetitions(text)
    prolongations = detect_prolongations(text)
    false_starts = detect_false_starts(text)

    all_detections.extend(fillers)
    all_detections.extend(repetitions)
    all_detections.extend(prolongations)
    all_detections.extend(false_starts)

    # Add segment context to each detection
    for det in all_detections:
        det["segment_text"] = text
        det["segment_start"] = segment.get("start")
        det["segment_end"] = segment.get("end")

    return all_detections
