"""Quick demo of each module — shows what each one does."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  MODULE 2: DISFLUENCY DETECTION DEMO")
print("=" * 60)
from disfluency.lexicon import build_disfluency_lexicon
from disfluency.detector import detect_fillers, detect_repetitions, detect_prolongations, detect_false_starts

lex = build_disfluency_lexicon()
segments = [
    "\u0909\u092e \u092e\u0948\u0902 \u091c\u093e \u0930\u0939\u093e \u0939\u0942\u0902",
    "\u092e\u0948\u0902 \u092e\u0948\u0902 \u0938\u094b\u091a \u0930\u0939\u093e \u0925\u093e",
    "\u0906\u091c \u092e\u094c\u0938\u092e \u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u093e \u0939\u0948",
]
for seg in segments:
    fillers = detect_fillers(seg, lex)
    reps = detect_repetitions(seg)
    total = len(fillers) + len(reps)
    status = f"Found {total} disfluency" if total else "Clean"
    print(f"  [{status}] {seg}")

print("\n" + "=" * 60)
print("  MODULE 3: SPELLING CLASSIFICATION DEMO")
print("=" * 60)
from spelling.lexicon_loader import load_hindi_lexicon
from spelling.classifier import classify_word

hindi_lex = load_hindi_lexicon()
words = ["\u092e\u0948\u0902", "\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930", "\u0916\u093e\u0928\u093e\u093e", "\u091c\u093e\u0928\u0924\u093e"]
for w in words:
    r = classify_word(w, hindi_lex)
    tag = "CORRECT" if r["classification"] == "correct" else "INCORRECT" if r["classification"] == "incorrect" else "UNKNOWN"
    print(f"  [{tag}] {w} -- {r['reason']} (Layer {r['layer']})")

print("\n" + "=" * 60)
print("  MODULE 4: LATTICE WER DEMO")
print("=" * 60)
from lattice.alignment import align_hypothesis_to_reference, compute_wer_from_alignment
from lattice.wer import evaluate_all_models

ref = "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u0940 \u0939\u0942\u0902"
models = {
    "whisper":   "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
    "wav2vec2":  "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
    "conformer": "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
    "nemo":      "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u093e \u0939\u0942\u0902",
    "indic":     "\u092e\u0948\u0902 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0924\u0940 \u0939\u0942\u0902",
}
result = evaluate_all_models(ref, models, threshold=3)
print(f"  Reference:  {ref}")
print(f"  Corrected:  {result.get('corrected_reference', 'N/A')}")
print(f"  Corrections: {len(result['corrections'])}")
for name in models:
    std = result["standard_wer"].get(name, {}).get("wer", 0) * 100
    lat = result["lattice_wer"].get(name, {}).get("wer", 0) * 100
    delta = lat - std
    print(f"    {name:<12} Standard={std:.0f}%  Lattice={lat:.0f}%  Delta={delta:+.0f}%")

print("\n" + "=" * 60)
print("  ALL DEMOS COMPLETE")
print("=" * 60)
