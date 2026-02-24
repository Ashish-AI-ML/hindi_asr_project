"""
Hindi Text Utilities
=====================
Text normalization, cleaning, and script detection for Hindi/Devanagari text.
Applied to transcription text before training and WER evaluation.
"""

import re
import unicodedata
from typing import Optional


# ─── Unicode Ranges ───────────────────────────────────────────────────
# Devanagari: U+0900–U+097F
# Devanagari Extended: U+A8E0–U+A8FF
# Vedic Extensions: U+1CD0–U+1CFF
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]")

# Hindi punctuation that should be removed for WER
HINDI_PUNCTUATION = re.compile(
    r"[।॥,\.\?\!;:\'\"\-\–\—\(\)\[\]\{\}\<\>\«\»…/\\@#\$%\^&\*\+\=\|~`]"
)

# Numbers (both Devanagari and ASCII)
DEVANAGARI_DIGITS = re.compile(r"[०-९]")

# Multiple spaces
MULTI_SPACE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """
    Apply NFC Unicode normalization.
    
    Hindi text can be encoded in multiple ways (composed vs decomposed forms).
    NFC ensures consistent representation for string comparison.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Basic text cleaning for transcription text.
    
    Operations:
    1. Unicode NFC normalization
    2. Strip leading/trailing whitespace
    3. Remove Hindi punctuation (।, ॥, etc.)
    4. Collapse multiple spaces to single space
    
    Preserves Devanagari characters, English words (as-is), and digits.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = HINDI_PUNCTUATION.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text


def normalize_for_wer(text: str) -> str:
    """
    Aggressive normalization for WER computation.
    
    Applied to BOTH reference and hypothesis before computing WER.
    This ensures fair comparison.
    
    Operations:
    1. All of clean_text operations
    2. Lowercase (for any English words mixed in)
    3. Remove all non-Devanagari, non-alphanumeric characters
    4. Collapse whitespace
    """
    if not text:
        return ""

    text = clean_text(text)
    text = text.lower()

    # Keep Devanagari, ASCII alphanumeric, and spaces
    filtered = []
    for char in text:
        if DEVANAGARI_PATTERN.match(char):
            filtered.append(char)
        elif char.isalnum():  # ASCII letters/digits
            filtered.append(char)
        elif char == " ":
            filtered.append(char)
        # else: discard

    text = "".join(filtered)
    text = MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text


def is_devanagari(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is primarily Devanagari script.
    
    Returns True if more than `threshold` fraction of non-space characters
    are Devanagari.
    """
    if not text:
        return False

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False

    devanagari_count = sum(1 for c in non_space if DEVANAGARI_PATTERN.match(c))
    return (devanagari_count / len(non_space)) >= threshold


def remove_english_words(text: str) -> str:
    """
    Remove English (Latin script) words from text, keeping only Devanagari.
    
    Useful for strict Hindi-only processing.
    """
    words = text.split()
    hindi_words = [w for w in words if is_devanagari(w, threshold=0.3)]
    return " ".join(hindi_words)


def transliterate_check(devanagari_word: str) -> Optional[str]:
    """
    Basic phonetic mapping check: does this Devanagari word look like
    a transliterated English word?
    
    This is a heuristic — maps common Devanagari transliterations.
    Returns the approximate English form if found, None otherwise.
    
    Note: For production use, integrate `indic-transliteration` library.
    """
    # Common English words in Devanagari (transliterations)
    COMMON_TRANSLITERATIONS = {
        "कंप्यूटर": "computer",
        "मोबाइल": "mobile",
        "इंटरनेट": "internet",
        "फोन": "phone",
        "स्कूल": "school",
        "कॉलेज": "college",
        "ऑफिस": "office",
        "बस": "bus",
        "टीवी": "tv",
        "रेडियो": "radio",
        "ट्रेन": "train",
        "डॉक्टर": "doctor",
        "इंजीनियर": "engineer",
        "टीचर": "teacher",
        "पुलिस": "police",
        "हॉस्पिटल": "hospital",
        "स्टेशन": "station",
        "मार्केट": "market",
        "बैंक": "bank",
        "होटल": "hotel",
        "पार्टी": "party",
        "वीडियो": "video",
        "गेम": "game",
        "प्रोग्राम": "program",
        "सिस्टम": "system",
        "डाटा": "data",
        "कैमरा": "camera",
        "टैक्सी": "taxi",
        "बजट": "budget",
        "रिपोर्ट": "report",
        "प्रोजेक्ट": "project",
        "मैनेजर": "manager",
    }

    normalized = normalize_unicode(devanagari_word.strip())
    return COMMON_TRANSLITERATIONS.get(normalized)


def count_words(text: str) -> int:
    """Count words in text (split on whitespace)."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())
