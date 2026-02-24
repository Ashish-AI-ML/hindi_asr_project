# Hindi ASR Research Project

**Josh Talks AI Researcher Intern — Speech & Audio Task Assignment**

A comprehensive Hindi speech recognition research project covering data preprocessing, Whisper fine-tuning, disfluency detection, spelling error classification, and lattice-based WER evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│   data/download.py │ audio_utils.py │ text_utils.py     │
│   data/manifest.py │ config.py                         │
└─────────────────────────────────────────────────────────┘
         ↓                ↓                ↓
┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
│   MODULE 1     │ │   MODULE 2     │ │   MODULE 3       │
│ Whisper FT     │ │ Disfluency     │ │ Spell Checker    │
│ + WER Eval     │ │ Detection      │ │ (1.77L words)    │
│ training/      │ │ disfluency/    │ │ spelling/        │
│ evaluation/    │ │                │ │                  │
└────────────────┘ └────────────────┘ └──────────────────┘
                                      ┌──────────────────┐
                                      │   MODULE 4       │
                                      │ Lattice WER      │
                                      │ lattice/         │
                                      └──────────────────┘
```

## Setup

```bash
cd hindi_asr_project
pip install -r requirements.txt
```

## Quick Start

### Q1: Fine-tune Whisper + WER Evaluation

```bash
# Step 1: Build data manifest (after downloading data)
python -c "from data.manifest import *; from data.download import *; ..."

# Step 2: Fine-tune Whisper-small
python -m training.train --manifest processed_data/manifests/train_manifest.json

# Step 3: Evaluate on FLEURS Hindi
python -m evaluation.wer_eval --model_path outputs/model/final_model
```

### Q2: Disfluency Detection

```bash
# Run full pipeline (detect + clip audio)
python -m disfluency.pipeline --manifest processed_data/manifests/train_manifest.json

# Text-only mode (skip audio clipping)
python -m disfluency.pipeline --manifest processed_data/manifests/train_manifest.json --no_clip
```

**Output**: `outputs/disfluency/disfluency_detections.csv` + audio clips in `outputs/disfluency/clips/`

### Q3: Spelling Classification

```bash
# Classify word list
python -m spelling.pipeline --wordlist path/to/words.txt
```

**Output**: `outputs/spelling/spelling_classification.csv`

### Q4: Lattice-Based WER

```bash
# Prepare input (JSON with reference + 5 model outputs)
python -m lattice.wer --input path/to/model_outputs.json --threshold 3
```

**Input format** (`model_outputs.json`):
```json
{
  "reference": "मैं जाती हूं",
  "models": {
    "model_1": "मैं जाता हूं",
    "model_2": "मैं जाता हूं",
    "model_3": "मैं जाता हूं",
    "model_4": "मैं जाता हूं",
    "model_5": "मैं जाती हूं"
  }
}
```

**Output**: `outputs/lattice/lattice_wer_report.md`

## Project Structure

```
hindi_asr_project/
├── config.py                    # Global configuration
├── requirements.txt             # Dependencies
├── data/                        # Shared data layer
│   ├── download.py              #   Download + validation
│   ├── audio_utils.py           #   Audio load/resample/clip
│   ├── text_utils.py            #   Hindi text normalization
│   └── manifest.py              #   Dataset manifest + health report
├── training/                    # Module 1: Whisper fine-tuning
│   ├── dataset.py               #   Custom Dataset classes
│   ├── data_collator.py         #   WhisperDataCollator
│   └── train.py                 #   Training script
├── evaluation/                  # Module 1: WER evaluation
│   └── wer_eval.py              #   FLEURS evaluation + tables
├── disfluency/                  # Module 2: Disfluency detection
│   ├── lexicon.py               #   Hindi filler dictionary
│   ├── detector.py              #   Multi-method detector
│   ├── clipper.py               #   Audio segment clipper
│   └── pipeline.py              #   End-to-end pipeline
├── spelling/                    # Module 3: Spelling classification
│   ├── lexicon_loader.py        #   Hindi dictionary loader
│   ├── unicode_validator.py     #   Devanagari validity checker
│   ├── classifier.py            #   4-layer classifier
│   └── pipeline.py              #   Classification pipeline
├── lattice/                     # Module 4: Lattice-based WER
│   ├── alignment.py             #   DP word alignment
│   ├── lattice_builder.py       #   Position lattice construction
│   ├── consensus.py             #   Consensus + reference correction
│   ├── wer.py                   #   Lattice WER computation
│   └── methodology.md           #   Approach justification
└── outputs/                     # Generated outputs
    ├── model/                   #   Model checkpoints
    ├── evaluation/              #   WER results
    ├── disfluency/              #   Disfluency CSV + clips
    ├── spelling/                #   Spelling classification CSV
    └── lattice/                 #   Lattice WER reports
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 16kHz mono audio | Whisper's required input format |
| NFC Unicode normalization | Hindi has multiple valid encodings; NFC ensures consistency |
| Segment duration 0.5–30s | < 0.5s = noise; > 30s = Whisper struggles |
| WER + CER reporting | CER gives fairer picture for morphologically rich Hindi |
| 4-layer spelling classifier | Cascading from fast dictionary to slow edit-distance |
| Word-level lattice alignment | Natural WER unit, interpretable, standard in ASR |
| Consensus threshold K=3 | Majority vote from 5 models |

## Research Mindset Checklist

- [x] Each design decision is justified, not just implemented
- [x] Data quality quantified before training (health report)
- [x] WER computed on standardized, normalized text
- [x] Failure cases documented (high-WER examples, uncertain spellings)
- [x] Results reproducible from methodology descriptions
- [x] Limitations explicitly stated for each module
