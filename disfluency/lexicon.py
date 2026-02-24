"""
Hindi Disfluency Lexicon
=========================
Curated dictionary of Hindi speech disfluencies categorized by type.
This is the "cheat sheet" for text-based disfluency detection.

Categories:
- fillers: verbal pauses (उम, उह, हम्म)
- hesitations: stalling words (मतलब, basically, actually)
- interjections: reactive sounds used as fillers (अच्छा, हां, ना)
"""

from typing import Dict, Set, List


def build_disfluency_lexicon() -> Dict[str, Set[str]]:
    """
    Build a categorized disfluency lexicon for Hindi.
    
    Returns dict mapping disfluency type → set of marker words.
    
    Note: Some words like "अच्छा" are context-dependent (can be filler
    OR real word). The detector should use position/repetition heuristics
    to disambiguate.
    """
    lexicon = {
        # ── Fillers (verbal pauses) ──────────────────────────────
        "filler": {
            # Devanagari
            "उम", "उम्म", "उह", "अम", "अम्म", "आह",
            "हम्म", "हम", "हं", "हां",
            "एम", "एम्म",
            # Common Hinglish fillers  
            "umm", "um", "uh", "uhh", "hmm", "hm", "ah", "ahh",
            "erm", "er",
        },

        # ── Hesitation markers (stalling words) ──────────────────
        "hesitation": {
            "मतलब", "यानी", "बोलो", "क्या बोलें",
            "कैसे बोलें", "कहें तो", "वो", "ये",
            "actually", "basically", "like", "you know",
            "I mean", "so", "well",
            "ऐसे", "वैसे", "तो",
        },

        # ── Interjections used as fillers ─────────────────────────
        "interjection_filler": {
            "अच्छा", "ठीक", "हैना", "है ना",
            "ना", "बस", "और", "तो",
            "okay", "ok", "right", "yeah",
            "अरे", "ओह", "वाह",
        },

        # ── Discourse markers (can indicate disfluency in context) ─
        "discourse_marker": {
            "देखो", "सुनो", "बताओ",
            "समझो", "मान लो",
            "see", "look", "listen",
        },
    }

    return lexicon


def get_all_filler_words() -> Set[str]:
    """Get a flat set of ALL disfluency marker words across categories."""
    lexicon = build_disfluency_lexicon()
    all_words = set()
    for category_words in lexicon.values():
        all_words.update(category_words)
    return all_words


def get_filler_categories() -> List[str]:
    """Get list of all disfluency category names."""
    return list(build_disfluency_lexicon().keys())
