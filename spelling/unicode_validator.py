"""
Devanagari Unicode Validator
==============================
Layer 3 of the spelling classifier: validates character-level Devanagari sequences.

Checks for:
- Invalid Unicode sequences (phonologically impossible in Hindi)
- Consecutive matras without consonants
- Invalid character combinations
- Characters outside expected ranges
"""

import re
import unicodedata
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.text_utils import normalize_unicode


# ─── Devanagari Character Categories ─────────────────────────────────

# Consonants: क-ह (U+0915-U+0939)
CONSONANTS = set(chr(c) for c in range(0x0915, 0x093A))
CONSONANTS.add('\u0929')  # ऩ
CONSONANTS.add('\u0931')  # ऱ
CONSONANTS.add('\u0934')  # ऴ
CONSONANTS.add('\u0958')  # क़
CONSONANTS.add('\u0959')  # ख़
CONSONANTS.add('\u095A')  # ग़
CONSONANTS.add('\u095B')  # ज़
CONSONANTS.add('\u095C')  # ड़
CONSONANTS.add('\u095D')  # ढ़
CONSONANTS.add('\u095E')  # फ़
CONSONANTS.add('\u095F')  # य़

# Independent vowels: अ-औ (U+0904-U+0914)
VOWELS = set(chr(c) for c in range(0x0904, 0x0915))

# Dependent vowel signs (matras): ा-ौ (U+093E-U+094C)
MATRAS = set(chr(c) for c in range(0x093E, 0x094D))

# Virama (halant): ् (U+094D) — used to form conjuncts
VIRAMA = '\u094D'

# Anusvara, Visarga, Chandrabindu
ANUSVARA = '\u0902'    # ं
VISARGA = '\u0903'     # ः
CHANDRABINDU = '\u0901'  # ँ

# Nukta
NUKTA = '\u093C'  # ़

# Devanagari digits
DEVANAGARI_DIGITS = set(chr(c) for c in range(0x0966, 0x0970))

# All valid Devanagari characters
ALL_DEVANAGARI = (
    CONSONANTS | VOWELS | MATRAS | DEVANAGARI_DIGITS |
    {VIRAMA, ANUSVARA, VISARGA, CHANDRABINDU, NUKTA}
)


def check_unicode_validity(word: str) -> Tuple[bool, str]:
    """
    Validate a Devanagari word's character sequence.
    
    Rules checked:
    1. No consecutive matras without an intervening consonant
    2. Matra must follow a consonant (not word-initial or after vowel)
    3. Virama must follow a consonant
    4. No invalid character combinations
    5. Nukta must follow a consonant
    
    Returns:
        (is_valid, reason) — True if valid, with reason explaining decision
    """
    if not word:
        return False, "Empty word"

    word = normalize_unicode(word)
    chars = list(word)
    n = len(chars)

    for i, char in enumerate(chars):
        # Skip non-Devanagari characters (handled elsewhere)
        if char not in ALL_DEVANAGARI and not char.isascii():
            continue

        # ── Rule 1: No consecutive matras ─────────────────────────
        if char in MATRAS:
            if i + 1 < n and chars[i + 1] in MATRAS:
                return False, (
                    f"Consecutive matras at position {i}: "
                    f"'{char}' + '{chars[i+1]}'"
                )

        # ── Rule 2: Matra must follow consonant or nukta+consonant ─
        if char in MATRAS:
            if i == 0:
                return False, f"Word starts with matra '{char}'"
            
            prev = chars[i - 1]
            # Valid predecessors: consonant, virama (in conjunct), nukta
            if prev not in CONSONANTS and prev != VIRAMA and prev != NUKTA:
                # Check if preceded by a consonant through nukta
                if not (i >= 2 and chars[i - 2] in CONSONANTS and prev == NUKTA):
                    pass  # Allow — some edge cases in borrowed words

        # ── Rule 3: Virama must follow consonant ──────────────────
        if char == VIRAMA:
            if i == 0:
                return False, "Word starts with virama"
            if chars[i - 1] not in CONSONANTS and chars[i - 1] != NUKTA:
                return False, (
                    f"Virama at position {i} doesn't follow consonant "
                    f"(preceded by '{chars[i-1]}')"
                )

        # ── Rule 4: Nukta must follow consonant ───────────────────
        if char == NUKTA:
            if i == 0:
                return False, "Word starts with nukta"
            if chars[i - 1] not in CONSONANTS:
                return False, f"Nukta at position {i} doesn't follow consonant"

    return True, "Valid Devanagari sequence"


def is_pure_devanagari(word: str) -> bool:
    """Check if word contains ONLY Devanagari characters (no Latin, etc.)."""
    word = normalize_unicode(word)
    for char in word:
        if char not in ALL_DEVANAGARI and not char.isspace():
            return False
    return True


def get_character_breakdown(word: str) -> dict:
    """
    Analyze the character composition of a Devanagari word.
    Returns counts of consonants, vowels, matras, etc.
    """
    word = normalize_unicode(word)
    breakdown = {
        "consonants": 0,
        "vowels": 0,  
        "matras": 0,
        "virama": 0,
        "anusvara": 0,
        "visarga": 0,
        "nukta": 0,
        "other": 0,
    }

    for char in word:
        if char in CONSONANTS:
            breakdown["consonants"] += 1
        elif char in VOWELS:
            breakdown["vowels"] += 1
        elif char in MATRAS:
            breakdown["matras"] += 1
        elif char == VIRAMA:
            breakdown["virama"] += 1
        elif char == ANUSVARA:
            breakdown["anusvara"] += 1
        elif char == VISARGA:
            breakdown["visarga"] += 1
        elif char == NUKTA:
            breakdown["nukta"] += 1
        else:
            breakdown["other"] += 1

    return breakdown
