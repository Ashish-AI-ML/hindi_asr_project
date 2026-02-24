# 🎙️ Hindi ASR — End-to-End Speech Recognition Research

A comprehensive Hindi Automatic Speech Recognition (ASR) research project covering model fine-tuning, linguistic analysis, and novel evaluation methods.

> **Highlights**: Whisper fine-tuning on Hindi • Disfluency detection & segmentation • Devanagari spelling validation • ROVER-inspired lattice WER

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                            │
│   download.py  │  audio_utils.py  │  text_utils.py          │
│   manifest.py  │  config.py                                 │
└────────────────┬──────────────┬──────────────┬──────────────┘
                 ↓              ↓              ↓
┌────────────────────┐ ┌───────────────┐ ┌────────────────────┐
│  Whisper           │ │  Disfluency   │ │  Spelling          │
│  Fine-Tuning       │ │  Detection    │ │  Classifier        │
│  + FLEURS WER Eval │ │  + Clipping   │ │  (4-layer cascade) │
│  training/         │ │  disfluency/  │ │  spelling/         │
│  evaluation/       │ │               │ │                    │
└────────────────────┘ └───────────────┘ └────────────────────┘
                                         ┌────────────────────┐
                                         │  Lattice WER       │
                                         │  (ROVER-inspired)  │
                                         │  lattice/          │
                                         └────────────────────┘
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- GPU recommended for Whisper training (Colab T4 works)

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/hindi_asr_project.git
cd hindi_asr_project
pip install -r requirements.txt
cp .env.example .env   # Add your HuggingFace token
```

### Quick Test
```bash
python test_all.py    # Run all unit tests (10 tests)
python demo.py        # See all modules in action
```

---

## Modules

### 1. Whisper Fine-Tuning & WER Evaluation

Fine-tunes `openai/whisper-small` on Hindi conversational speech and benchmarks against the FLEURS Hindi test set.

**Run via Colab** (GPU required):
```
notebooks/01_whisper_finetune_and_eval.ipynb
```

**Or locally**:
```bash
python -m training.train --manifest processed_data/manifests/train_manifest.json
python -m evaluation.wer_eval --model_path outputs/model/final_model
```

**Key decisions**:
- Learning rate `1e-5` to prevent catastrophic forgetting
- FP16 + gradient accumulation for memory efficiency
- Forced decoder IDs ensure Hindi-only output

---

### 2. Disfluency Detection & Audio Segmentation

Detects speech disfluencies (fillers, repetitions, prolongations, false starts) in Hindi transcripts and clips the corresponding audio segments.

```bash
python -m disfluency.pipeline --manifest processed_data/manifests/train_manifest.json
```

**Output**: Structured CSV with each disfluency occurrence + clipped audio segments.

**Detection methods**:
| Type | Approach |
|------|----------|
| Fillers | Lexicon matching (उम, आह, hmm, actually...) |
| Repetitions | Consecutive word comparison |
| Prolongations | Regex for repeated Devanagari characters |
| False starts | Fragment detection + interruption markers |

---

### 3. Hindi Spelling Error Detection

4-layer cascade classifier for identifying correct vs incorrect Hindi spellings at scale (~1.77L unique words).

```bash
python -m spelling.pipeline --wordlist path/to/words.txt
```

**Output**: CSV with `word, classification (correct/incorrect), reason, layer`.

**Classification cascade**:
1. **Dictionary lookup** — fast match against 300+ core Hindi words
2. **Morphological analysis** — suffix stripping to find valid roots
3. **Unicode validation** — detects invalid Devanagari sequences
4. **Edit distance** — catches typos within 1 edit of known words

> English words transliterated to Devanagari (e.g., "कंप्यूटर" for "computer") are treated as **correct** per Hindi transcription guidelines.

---

### 4. Lattice-Based WER with Consensus Correction

ROVER-inspired approach that improves WER fairness when the human reference itself contains errors.

```bash
python -m lattice.wer --input model_outputs.json --threshold 3
```

**How it works**:
1. Align all 5 ASR model outputs to the reference using DP
2. At each position, count which word the majority of models agree on
3. If ≥K models agree on a word that differs from reference → correct the reference
4. Recompute WER against the corrected reference

**Result**: Models unfairly penalized by a wrong reference see reduced WER; others stay unchanged.

See [lattice/methodology.md](lattice/methodology.md) for full theoretical justification.

---

## Project Structure

```
hindi_asr_project/
├── config.py                    # Central configuration + .env loader
├── requirements.txt             # All dependencies
├── test_all.py                  # Unit tests (10 tests, all modules)
├── demo.py                      # Quick demo script
│
├── data/                        # Shared data utilities
│   ├── download.py              #   GCS audio download + validation
│   ├── audio_utils.py           #   Load, resample, clip audio
│   ├── text_utils.py            #   Hindi/Devanagari normalization
│   └── manifest.py              #   Dataset manifest builder
│
├── training/                    # Whisper fine-tuning
│   ├── train.py                 #   Training script
│   ├── dataset.py               #   Custom PyTorch Dataset
│   └── data_collator.py         #   Batch collation with padding
│
├── evaluation/                  # WER benchmarking
│   └── wer_eval.py              #   FLEURS evaluation + tables
│
├── disfluency/                  # Disfluency detection
│   ├── lexicon.py               #   Hindi filler word dictionary
│   ├── detector.py              #   Multi-method detector
│   ├── clipper.py               #   Audio segment clipper
│   └── pipeline.py              #   End-to-end pipeline + CSV
│
├── spelling/                    # Spelling classification
│   ├── lexicon_loader.py        #   Multi-source Hindi dictionary
│   ├── unicode_validator.py     #   Devanagari sequence validator
│   ├── classifier.py            #   4-layer cascade classifier
│   └── pipeline.py              #   Batch classification pipeline
│
├── lattice/                     # Lattice-based WER
│   ├── alignment.py             #   DP word-level alignment
│   ├── lattice_builder.py       #   Position lattice construction
│   ├── consensus.py             #   Majority voting + correction
│   ├── wer.py                   #   Standard vs lattice WER
│   └── methodology.md           #   Approach & justification
│
└── notebooks/                   # Colab notebooks
    ├── 01_whisper_finetune_and_eval.ipynb
    └── 02_disfluency_spelling_lattice.ipynb
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| 16kHz mono audio | Whisper's required input format |
| NFC Unicode normalization | Hindi has multiple valid byte sequences; NFC ensures consistency |
| Segment duration 0.5–30s | < 0.5s is noise; > 30s exceeds Whisper's attention window |
| WER + CER dual reporting | CER gives a fairer picture for morphologically rich Hindi |
| Cascading classifier layers | Fast dictionary check catches 80%+ words; expensive edit-distance runs only on unknowns |
| Word-level lattice alignment | Standard WER unit, interpretable, established in ASR literature |
| Consensus threshold K=3/5 | Simple majority — robust against individual model errors |

---

## Environment Variables

Copy `.env.example` to `.env` and add your keys:

```bash
HF_TOKEN=hf_your_token_here   # Required — HuggingFace model access
```

---

## License

MIT
