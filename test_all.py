"""
End-to-end test for the Hindi ASR project.
Tests all modules WITHOUT requiring GPU or external data.
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    from config import SAMPLE_RATE, MODEL_NAME, HF_TOKEN, TRAINING_CONFIG
    assert SAMPLE_RATE == 16000
    assert MODEL_NAME == "openai/whisper-small"
    assert isinstance(TRAINING_CONFIG, dict)
    print("[PASS] config.py")

def test_text_utils():
    from data.text_utils import (
        normalize_unicode, clean_text, normalize_for_wer, is_devanagari
    )
    assert normalize_unicode("test") == "test"
    assert is_devanagari("\u0939\u093f\u0902\u0926\u0940")
    assert not is_devanagari("English")
    print("[PASS] data/text_utils.py")

def test_disfluency_lexicon():
    from disfluency.lexicon import build_disfluency_lexicon, get_all_filler_words
    lex = build_disfluency_lexicon()
    assert "filler" in lex
    assert len(get_all_filler_words()) > 30
    print(f"[PASS] disfluency/lexicon.py ({len(get_all_filler_words())} filler words)")

def test_disfluency_detector():
    from disfluency.lexicon import build_disfluency_lexicon
    from disfluency.detector import (
        detect_fillers, detect_repetitions, detect_prolongations
    )
    lex = build_disfluency_lexicon()
    
    fillers = detect_fillers("\u0909\u092e \u092e\u0948\u0902 \u091c\u093e \u0930\u0939\u093e \u0939\u0942\u0902", lex)
    assert len(fillers) >= 1, f"Expected fillers, got {fillers}"
    
    reps = detect_repetitions("\u092e\u0948\u0902 \u092e\u0948\u0902 \u091c\u093e \u0930\u0939\u093e \u0939\u0942\u0902")
    assert len(reps) >= 1, f"Expected repetitions, got {reps}"
    
    prol = detect_prolongations("\u0938\u094b\u094b\u094b\u094b\u091a \u0930\u0939\u093e \u0925\u093e")
    assert len(prol) >= 1, f"Expected prolongation, got {prol}"
    
    print(f"[PASS] disfluency/detector.py (fillers={len(fillers)}, reps={len(reps)}, prol={len(prol)})")

def test_spelling_unicode_validator():
    from spelling.unicode_validator import check_unicode_validity, is_pure_devanagari
    
    valid, _ = check_unicode_validity("\u0928\u092e\u0938\u094d\u0924\u0947")
    assert valid, "namaste should be valid"
    
    valid2, _ = check_unicode_validity("\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930")
    assert valid2, "computer should be valid"
    
    print("[PASS] spelling/unicode_validator.py")

def test_spelling_lexicon_loader():
    from spelling.lexicon_loader import (
        load_hindi_lexicon, is_transliterated_english
    )
    lex = load_hindi_lexicon()
    assert len(lex) > 100
    assert is_transliterated_english("\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930")
    print(f"[PASS] spelling/lexicon_loader.py ({len(lex)} words)")

def test_spelling_classifier():
    from spelling.lexicon_loader import load_hindi_lexicon
    from spelling.classifier import classify_word
    
    lex = load_hindi_lexicon()
    
    r1 = classify_word("\u092e\u0948\u0902", lex)
    assert r1["classification"] == "correct", f"Expected correct: {r1}"
    
    r2 = classify_word("\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930", lex)
    assert r2["classification"] == "correct", f"Expected correct: {r2}"
    
    print(f"[PASS] spelling/classifier.py")

def test_lattice_alignment():
    from lattice.alignment import align_hypothesis_to_reference, compute_wer_from_alignment
    
    ref = ["A", "B", "C"]
    hyp = ["A", "B", "C"]
    a = align_hypothesis_to_reference(ref, hyp)
    m = compute_wer_from_alignment(a)
    assert m["wer"] == 0.0, f"Perfect match should be 0 WER: {m}"
    
    ref2 = ["A", "B", "C"]
    hyp2 = ["A", "X", "C"]
    a2 = align_hypothesis_to_reference(ref2, hyp2)
    m2 = compute_wer_from_alignment(a2)
    assert abs(m2["wer"] - 1/3) < 0.01, f"Expected ~33 WER: {m2}"
    
    print(f"[PASS] lattice/alignment.py (perfect=0, 1sub={m2['wer']*100:.0f})")

def test_lattice_consensus():
    from lattice.consensus import compute_consensus
    
    word, count, _ = compute_consensus(["A", "A", "A", "B", "B"], threshold=3)
    assert word == "A"
    assert count == 3
    
    word2, _, _ = compute_consensus(["A", "B", "C", "D", "E"], threshold=3)
    assert word2 is None
    
    print("[PASS] lattice/consensus.py")

def test_lattice_wer():
    from lattice.wer import evaluate_all_models
    
    ref = "\u092e\u0948\u0902 \u091c\u093e\u0924\u0940 \u0939\u0942\u0902"
    models = {
        "m1": "\u092e\u0948\u0902 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
        "m2": "\u092e\u0948\u0902 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
        "m3": "\u092e\u0948\u0902 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
        "m4": "\u092e\u0948\u0902 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
        "m5": "\u092e\u0948\u0902 \u091c\u093e\u0924\u0940 \u0939\u0942\u0902",
    }
    result = evaluate_all_models(ref, models, threshold=3)
    
    assert "standard_wer" in result
    assert "lattice_wer" in result
    assert "corrections" in result
    assert len(result["corrections"]) >= 1, "Should have at least 1 correction"
    
    print(f"[PASS] lattice/wer.py (corrections={len(result['corrections'])})")

if __name__ == "__main__":
    print("=" * 60)
    print("  Hindi ASR Project — End-to-End Tests")
    print("=" * 60)
    
    tests = [
        test_config,
        test_text_utils,
        test_disfluency_lexicon,
        test_disfluency_detector,
        test_spelling_unicode_validator,
        test_spelling_lexicon_loader,
        test_spelling_classifier,
        test_lattice_alignment,
        test_lattice_consensus,
        test_lattice_wer,
    ]
    
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
