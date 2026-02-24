"""
Hindi Lexicon Loader
=====================
Combines multiple Hindi word sources into a comprehensive lookup set.
Serves as Layer 1 of the spelling classifier — dictionary membership test.

Sources:
- Built-in high-frequency Hindi word list
- Optionally: NLTK Hindi corpus, IndicNLP library wordlists
- Common English-to-Devanagari transliterations (per transcription guidelines)
"""

import os
import logging
from typing import Set, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.text_utils import normalize_unicode

logger = logging.getLogger(__name__)


# ─── Built-in Core Hindi Words ───────────────────────────────────────
# High-frequency Hindi words that should always be recognized as correct.
# This is a seed list; the full lexicon is loaded from external sources.

CORE_HINDI_WORDS = {
    # Pronouns
    "मैं", "हम", "तुम", "आप", "वह", "वे", "यह", "ये", "कौन", "क्या",
    "कोई", "कुछ", "सब", "खुद", "अपना", "अपनी", "अपने", "उसका", "उसकी",
    "उनका", "उनकी", "मेरा", "मेरी", "मेरे", "हमारा", "हमारी", "तुम्हारा",
    "आपका", "आपकी", "इसका", "इसकी", "जो", "जिसका",
    
    # Postpositions
    "का", "की", "के", "को", "से", "में", "पर", "तक", "ने", "के लिए",
    "के बारे", "के साथ", "के बाद", "के पहले", "के ऊपर", "के नीचे",
    
    # Verbs (common forms)
    "है", "हैं", "था", "थी", "थे", "थीं", "हो", "होता", "होती", "होते",
    "करना", "करता", "करती", "करते", "किया", "करें", "करो", "कर",
    "जाना", "जाता", "जाती", "जाते", "गया", "गई", "गए", "जाओ", "जाएं",
    "आना", "आता", "आती", "आते", "आया", "आई", "आए", "आओ",
    "देना", "देता", "देती", "दिया", "दी", "दो", "दें",
    "लेना", "लेता", "लेती", "लिया", "ली", "लो", "लें",
    "बोलना", "बोलता", "बोलती", "बोला", "बोली", "बोलो",
    "कहना", "कहता", "कहती", "कहा", "कही", "कहो",
    "सोचना", "सोचता", "सोचती", "सोचा", "सोची",
    "देखना", "देखता", "देखती", "देखा", "देखी", "देखो",
    "सुनना", "सुनता", "सुनती", "सुना", "सुनी", "सुनो",
    "खाना", "खाता", "खाती", "खाया", "खाई", "खाओ",
    "पीना", "पीता", "पीती", "पिया", "पी",
    "रहना", "रहता", "रहती", "रहा", "रही", "रहे", "रहो",
    "चलना", "चलता", "चलती", "चला", "चली", "चलो",
    "बैठना", "बैठा", "बैठी", "बैठो",
    "उठना", "उठा", "उठी", "उठो",
    "सकना", "सकता", "सकती", "सकते",
    "चाहना", "चाहता", "चाहती", "चाहिए",
    "पड़ना", "पड़ता", "पड़ती", "पड़ा", "पड़ी",
    "मिलना", "मिलता", "मिलती", "मिला", "मिली",
    "लगना", "लगता", "लगती", "लगा", "लगी",
    "रखना", "रखता", "रखती", "रखा", "रखी", "रखो",
    "पहुँचना", "पहुँचा", "पहुँची",
    "बनाना", "बनाता", "बनाती", "बनाया", "बनाई",
    "समझना", "समझता", "समझती", "समझा", "समझी",
    "पढ़ना", "पढ़ता", "पढ़ती", "पढ़ा", "पढ़ी",
    "लिखना", "लिखता", "लिखती", "लिखा", "लिखी",
    "हुआ", "हुई", "हुए",
    
    # Adjectives
    "अच्छा", "अच्छी", "अच्छे", "बुरा", "बुरी", "बुरे",
    "बड़ा", "बड़ी", "बड़े", "छोटा", "छोटी", "छोटे",
    "नया", "नई", "नए", "पुराना", "पुरानी", "पुराने",
    "सही", "गलत", "ज़्यादा", "कम", "बहुत", "थोड़ा",
    "ऊँचा", "नीचा", "लंबा", "मोटा", "पतला",
    "सुंदर", "खूबसूरत", "गंदा", "साफ",
    
    # Adverbs & Conjunctions
    "और", "या", "लेकिन", "परंतु", "मगर", "किंतु",
    "फिर", "तब", "अब", "कभी", "हमेशा", "कभी-कभी",
    "यहाँ", "वहाँ", "कहाँ", "कब", "कैसे", "क्यों",
    "पहले", "बाद", "ऊपर", "नीचे", "अंदर", "बाहर",
    "आज", "कल", "परसों", "अभी", "तुरंत",
    "बहुत", "काफी", "थोड़ा", "ज़रा", "बस",
    "भी", "ही", "तो", "सिर्फ", "केवल",
    "शायद", "ज़रूर", "बिल्कुल", "सच", "झूठ",
    
    # Nouns (common)
    "लोग", "आदमी", "औरत", "बच्चा", "बच्ची", "लड़का", "लड़की",
    "घर", "जगह", "शहर", "गाँव", "देश", "दुनिया",
    "काम", "बात", "चीज़", "समय", "दिन", "रात",
    "पानी", "खाना", "रोटी", "चाय", "दूध",
    "पैसा", "रुपया", "सरकार", "स्कूल", "अस्पताल",
    "परिवार", "दोस्त", "भाई", "बहन", "माँ", "पिता",
    "जीवन", "मरना", "प्यार", "दिल", "मन", "आँख",
    "हाथ", "पैर", "सिर", "मुँह",
    
    # Numbers
    "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात",
    "आठ", "नौ", "दस", "सौ", "हज़ार", "लाख", "करोड़",
    
    # Negation
    "नहीं", "ना", "मत", "न",
    
    # Question words
    "कितना", "कितनी", "कितने", "किसका", "किसकी", "किसने",
    "किधर", "कहाँ", "कब", "क्यों", "कैसा", "कैसी",
}


def load_hindi_lexicon(
    external_wordlist_path: Optional[str] = None,
) -> Set[str]:
    """
    Load and combine Hindi word sources into a lookup set.
    
    Combines:
    1. Built-in core word list (~300 high-frequency words)
    2. External wordlist file (one word per line, if provided)
    3. Tries to load NLTK Hindi corpus if available
    
    Returns a set of valid Hindi words (NFC-normalized).
    """
    lexicon = set()

    # ── Layer 1: Core words ───────────────────────────────────────
    for word in CORE_HINDI_WORDS:
        lexicon.add(normalize_unicode(word))

    logger.info(f"Core lexicon: {len(lexicon)} words")

    # ── Layer 2: External wordlist ────────────────────────────────
    if external_wordlist_path and os.path.exists(external_wordlist_path):
        try:
            with open(external_wordlist_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = normalize_unicode(line.strip())
                    if word:
                        lexicon.add(word)
            logger.info(
                f"After external wordlist: {len(lexicon)} words"
            )
        except Exception as e:
            logger.warning(f"Failed to load external wordlist: {e}")

    # ── Layer 3: NLTK Hindi corpus (if available) ─────────────────
    try:
        import nltk
        from nltk.corpus import indian
        nltk.download("indian", quiet=True)
        hindi_words = set(indian.words("hindi.pos"))
        for word in hindi_words:
            lexicon.add(normalize_unicode(word))
        logger.info(f"After NLTK Hindi corpus: {len(lexicon)} words")
    except Exception:
        logger.info("NLTK Hindi corpus not available, skipping")

    logger.info(f"Final lexicon size: {len(lexicon)} words")
    return lexicon


# ─── English in Devanagari ────────────────────────────────────────────

ENGLISH_DEVANAGARI_MAP = {
    # Common English words transliterated to Devanagari
    # Per guidelines: these are CORRECT spellings
    "कंप्यूटर": "computer", "मोबाइल": "mobile", "इंटरनेट": "internet",
    "फोन": "phone", "स्कूल": "school", "कॉलेज": "college",
    "ऑफिस": "office", "बस": "bus", "टीवी": "tv",
    "रेडियो": "radio", "ट्रेन": "train", "डॉक्टर": "doctor",
    "इंजीनियर": "engineer", "टीचर": "teacher", "पुलिस": "police",
    "हॉस्पिटल": "hospital", "स्टेशन": "station", "मार्केट": "market",
    "बैंक": "bank", "होटल": "hotel", "पार्टी": "party",
    "वीडियो": "video", "गेम": "game", "प्रोग्राम": "program",
    "सिस्टम": "system", "डाटा": "data", "कैमरा": "camera",
    "टैक्सी": "taxi", "बजट": "budget", "रिपोर्ट": "report",
    "प्रोजेक्ट": "project", "मैनेजर": "manager", "ड्राइवर": "driver",
    "इंडिया": "india", "अमेरिका": "america", "चाइना": "china",
    "जापान": "japan", "कोरिया": "korea", "ऑस्ट्रेलिया": "australia",
    "यूट्यूब": "youtube", "गूगल": "google", "फेसबुक": "facebook",
    "व्हाट्सएप": "whatsapp", "इंस्टाग्राम": "instagram",
    "ऐप": "app", "वेबसाइट": "website", "ऑनलाइन": "online",
    "ऑफलाइन": "offline", "पासवर्ड": "password", "ईमेल": "email",
    "लैपटॉप": "laptop", "टैबलेट": "tablet", "सॉफ्टवेयर": "software",
    "हार्डवेयर": "hardware", "नेटवर्क": "network",
    "यूनिवर्सिटी": "university", "एग्जाम": "exam", "रिजल्ट": "result",
    "सब्जेक्ट": "subject", "प्रॉब्लम": "problem", "सॉल्यूशन": "solution",
    "एक्सपीरियंस": "experience", "इंटरव्यू": "interview",
    "कंपनी": "company", "बिजनेस": "business", "मीटिंग": "meeting",
    "प्रेजेंटेशन": "presentation",
}


def load_english_hindi_map() -> Dict[str, str]:
    """
    Get the Devanagari → English transliteration map.
    
    Per transcription guidelines, English words spoken in conversation
    are transcribed in Devanagari. These are CORRECT spellings.
    """
    return {normalize_unicode(k): v for k, v in ENGLISH_DEVANAGARI_MAP.items()}


def is_transliterated_english(word: str) -> bool:
    """Check if a Devanagari word is a known English transliteration."""
    return normalize_unicode(word) in load_english_hindi_map()
