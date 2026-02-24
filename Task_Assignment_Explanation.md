# Task Assignment — AI Researcher Intern (Speech & Audio) | Josh Talks
## Comprehensive Question-by-Question Explanation & Approach

---

## Table of Contents
1. [Question 1: Data Preprocessing + Whisper Fine-Tuning + WER Evaluation](#question-1)
2. [Question 2: Disfluency Detection in Hindi Speech](#question-2)
3. [Question 3: Hindi Spelling Classification](#question-3)
4. [Question 4: Lattice-Based WER Evaluation](#question-4)

---

<a id="question-1"></a>
## Question 1: Data Preprocessing + Fine-Tune Whisper + WER Evaluation

### What the Question Asks

We are given a Hindi speech dataset hosted on Google Cloud Storage (GCS). Each recording is identified by a `user_id` and `recording_id`, and has three associated files:

| File Type | URL Pattern | Format |
|-----------|-------------|--------|
| Audio | `https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_recording.wav` | WAV |
| Transcription | `.../{recording_id}_transcription.json` | JSON (list of segments with start, end, text) |
| Metadata | `.../{recording_id}_metadata.json` | JSON (language, duration, speaker info) |

The task is to:
1. **Download and preprocess** the dataset (handle audio + text data)
2. **Fine-tune OpenAI Whisper-small** on this Hindi dataset
3. **Evaluate Word Error Rate (WER)** on a standard benchmark (FLEURS Hindi test set) and compare baseline vs. fine-tuned performance

### Our Expert Approach

#### Phase 1: Data Pipeline (`data/` module)

**Step 1 — Download with Robustness (`data/download.py`)**

We built a production-grade download pipeline with:
- **Exponential backoff retry** (3 retries, doubling wait time) — GCS can have transient failures
- **Streaming downloads** — memory efficient for large audio files (8KB chunks)
- **File validation** — rejects zero-byte downloads
- **Idempotent** — skips already-downloaded files to support incremental runs

```
download_file(url, dest_path)
  → retry loop with exponential backoff
  → stream response to disk (8KB chunks)
  → verify file size > 0
```

**Step 2 — Transcription Parsing (`data/download.py: parse_transcription`)**

Each transcription JSON is a list of segments:
```json
[
  {"start": 0.0, "end": 3.5, "speaker_id": 1, "text": "नमस्ते दोस्तों"},
  {"start": 3.5, "end": 7.2, "speaker_id": 1, "text": "आज हम बात करेंगे"}
]
```

Our parser:
- Validates required keys (`start`, `end`, `text`)
- Coerces timestamps to float (handles string timestamps)
- Filters out segments with missing or invalid data
- Logs every rejection for debugging

**Step 3 — Audio Processing (`data/audio_utils.py`)**

All audio is resampled to **16kHz mono WAV** — this is Whisper's required input format. Key decisions:
- **16kHz sample rate**: Whisper was trained at 16kHz; resampling to any other rate degrades performance
- **Mono channel**: Whisper expects single-channel input
- **Segment clipping**: We extract individual segments from full recordings using precise timestamps with a configurable buffer (±0.1s) to avoid cutting speech

**Step 4 — Text Normalization (`data/text_utils.py`)**

Hindi text normalization is critical because:
- **NFC Unicode normalization**: Hindi characters can be encoded in multiple forms (composed vs. decomposed). E.g., "ऩ" can be represented as a single codepoint or as "न + nukta". NFC ensures consistent representation
- **Punctuation removal**: Hindi uses `।` (danda) and `॥` (double danda), plus standard punctuation — all removed for WER
- **Lowercase English**: Any mixed-in English words are lowercased for fair comparison
- **Whitespace collapse**: Multiple spaces → single space

```python
normalize_for_wer(text):
  → NFC normalization
  → remove Hindi/English punctuation
  → lowercase
  → keep only Devanagari + alphanumeric + spaces
  → collapse whitespace
```

**Step 5 — Manifest Building + Data Health Report (`data/manifest.py`)**

The manifest is the "table of contents" for training. Each entry maps an audio file path to its transcription text. We validate every segment:

| Check | Threshold | Reason |
|-------|-----------|--------|
| Segment duration min | 0.5s | < 0.5s is likely noise/silence |
| Segment duration max | 30.0s | Whisper struggles with very long segments |
| Text minimum length | 1 character | Empty transcripts are useless |
| Timestamps logical | start < end | Catches data corruption |
| End ≤ audio duration | +1s tolerance | Catches misaligned timestamps |

We also generate a **Data Health Report** showing:
- Total segments processed, valid vs. rejected counts
- Acceptance rate
- Duration distribution histogram (bucketed: <1s, 1-5s, 5-10s, etc.)
- Text statistics (word count min/max/mean)
- Rejection reasons breakdown

> **Research mindset**: Quantifying data quality BEFORE training lets us know if the dataset has issues. Training on bad data wastes compute.

#### Phase 2: Whisper Fine-Tuning (`training/` module)

**Model: `openai/whisper-small`** (244M parameters)

We chose Whisper-small as the balance between accuracy and trainability:
- `whisper-tiny` is too small for Hindi (poor baseline)
- `whisper-medium`/`whisper-large` require too much GPU memory for fine-tuning
- `whisper-small` is the standard research choice for low-resource fine-tuning

**Key Training Decisions (documented in `training/train.py`):**

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Learning rate | 1e-5 | Low LR prevents catastrophic forgetting of Whisper's multilingual knowledge |
| Warmup steps | 500 | Gradual warmup stabilizes early training |
| Epochs | 3 | Standard for fine-tuning; more risks overfitting on small datasets |
| Batch size | 8 × 2 (gradient accumulation) | Effective batch 16, fits in GPU memory |
| FP16 | True | 2× speedup on CUDA GPUs, negligible quality loss |
| Best model selection | min(WER) | We want the checkpoint with lowest validation WER |

**Critical Configuration:**
- `forced_decoder_ids` set for Hindi: Forces the model to decode in Hindi, not English (Whisper is multilingual and defaults to English)
- `use_cache = False`: Required for gradient checkpointing (saves GPU memory)
- `predict_with_generate = True`: Enables autoregressive decoding during eval

**Custom Components:**
- `WhisperHindiDataset` (`training/dataset.py`): Loads audio, extracts mel features, tokenizes text
- `WhisperDataCollator` (`training/data_collator.py`): Pads variable-length sequences, sets `-100` for label padding (ignored by CrossEntropy loss)

#### Phase 3: WER Evaluation (`evaluation/wer_eval.py`)

**Benchmark: FLEURS Hindi test set** (`google/fleurs`, `hi_in`)

FLEURS is a standard multilingual ASR benchmark used in research papers. Using it makes our results directly comparable to published work.

**Evaluation Strategy:**
1. Run baseline Whisper-small (pretrained, no fine-tuning) on FLEURS Hindi
2. Run fine-tuned model on the same test set
3. Compare WER and CER side-by-side

**Why both WER and CER?**
- **WER** (Word Error Rate): Standard metric, but penalizes whole-word errors. Hindi's rich morphology means a single suffix error counts as a full word error
- **CER** (Character Error Rate): Fairer for morphologically rich languages — captures that "जाता" vs "जाती" differ by only one character

**Output**: A formatted comparison table:
```
| Model                      | WER (%) | CER (%) | Samples |
|----------------------------|---------|---------|---------|
| Whisper-small (Baseline)   | XX.XX   | XX.XX   | NNN     |
| Whisper-small (Fine-tuned) | XX.XX   | XX.XX   | NNN     |
```

**Failure Analysis**: We also log the top 20 worst-performing samples (WER > 50%) for qualitative error analysis — understanding *why* the model fails is as important as the aggregate number.

---

<a id="question-2"></a>
## Question 2: Disfluency Detection in Hindi Speech

### What the Question Asks

Given the Hindi transcription data, build a system to **detect speech disfluencies** — the fillers, repetitions, false starts, and prolongations that occur naturally in spontaneous speech. When disfluencies are detected, **clip the corresponding audio segments** for analysis.

### Our Expert Approach

#### Architecture: Multi-Method Text-Based Detection + Audio Clipping

We chose a **text-first approach** because:
1. Transcriptions are already available (from Q1)
2. Text-based detection is fast, interpretable, and doesn't require specialized audio models
3. Audio clipping is applied afterwards using the timestamps from the transcription

#### Detection Method 1: Lexicon-Based Filler Detection (`disfluency/lexicon.py` + `detector.py`)

We curated a comprehensive **Hindi disfluency lexicon** organized by category:

| Category | Examples (Devanagari) | Examples (Hinglish) |
|----------|----------------------|---------------------|
| **Fillers** (verbal pauses) | उम, उम्म, उह, अम, हम्म, हं | umm, uh, hmm, er |
| **Hesitation markers** (stalling) | मतलब, यानी, वो, ये | actually, basically, like |
| **Interjection fillers** | अच्छा, ठीक, हैना, ना, बस | okay, right, yeah |
| **Discourse markers** | देखो, सुनो, समझो | see, look, listen |

**Detection Algorithm:**
```
For each word in transcript:
  → Strip punctuation (।,.)
  → Check against lexicon (each category)
  → If match found → record detection with position, category
  → Break after first match (avoid double-counting)
```

> **Design note**: Words like "अच्छा" are context-dependent — can be a filler ("अच्छा, तो...") OR a real adjective ("अच्छा था"). Our detector flags candidates; downstream systems can use position heuristics to disambiguate.

#### Detection Method 2: Repetition Detection (`detector.py: detect_repetitions`)

Detects two types of repetition:

**Single word repetition**: "मैं मैं" (I I)
```
For each consecutive word pair (w[i], w[i+1]):
  → Clean punctuation from both
  → If w[i] == w[i+1] and len > 1 character → word_repetition
```

**Bigram (phrase) repetition**: "मैं सोच मैं सोच" (I think I think)
```
For each 4-word window:
  → Compare bigram(w[i], w[i+1]) vs bigram(w[i+2], w[i+3])
  → If identical → phrase_repetition
```

#### Detection Method 3: Prolongation Detection (`detector.py: detect_prolongations`)

Detects when speakers stretch out sounds, which appears in text as repeated characters:
- "सोोोो" (stretched "so")
- "नहीींं" (stretched "no")

Uses regex pattern `(.)\\1{2,}` — any character repeated 3+ times.

**Caveat documented**: Some legitimate Hindi words have repeated characters. This detector flags candidates; false positives are possible.

#### Detection Method 4: False Start Detection (`detector.py: detect_false_starts`)

Detects abandoned utterances using heuristics:
- **Interruption punctuation**: `...`, `—`, `–` indicate abandoned speech
- **Very short segments** (1-2 words): Likely abandoned utterances

> This is explicitly documented as the hardest disfluency type to detect from text alone.

#### Audio Clipping (`disfluency/clipper.py`)

When disfluencies are detected in a segment with timestamps, we:
1. Load the full audio file
2. Extract the segment from `start` to `end` (with ±0.1s buffer)
3. Save as individual WAV clip for further analysis

#### Master Pipeline (`disfluency/pipeline.py`)

End-to-end pipeline that:
1. Loads the manifest
2. Builds the disfluency lexicon
3. Runs ALL four detectors on every segment
4. Clips audio for detected disfluencies
5. Saves results to CSV (`outputs/disfluency/disfluency_detections.csv`)

```bash
python -m disfluency.pipeline --manifest processed_data/manifests/train_manifest.json
```

**Output CSV columns**: segment_text, type, word, position, category, segment_start, segment_end

---

<a id="question-3"></a>
## Question 3: Hindi Spelling Classification

### What the Question Asks

Given a list of Hindi words, classify each word as **correctly spelled** or **incorrectly spelled**. The system should work with a dictionary of approximately 1.77 lakh (177,000) words.

### Our Expert Approach

#### Architecture: 4-Layer Cascading Classifier

We designed a **cascading classifier** where each layer is progressively more expensive but catches cases the previous layers missed:

```
Input Word
  │
  ▼
┌─────────────────────────────────┐
│ Layer 1: Dictionary Lookup      │ → O(1) hash lookup
│ (+ English transliteration)     │   FAST
└─────────────┬───────────────────┘
              │ Not found
              ▼
┌─────────────────────────────────┐
│ Layer 2: Morphological Analysis │ → Strip suffixes, check root
│ (Hindi inflection awareness)    │   MODERATE
└─────────────┬───────────────────┘
              │ No valid decomposition
              ▼
┌─────────────────────────────────┐
│ Layer 3: Unicode Validity       │ → Check character sequences
│ (Devanagari rules)              │   FAST
└─────────────┬───────────────────┘
              │ Valid Unicode
              ▼
┌─────────────────────────────────┐
│ Layer 4: Edit Distance          │ → Levenshtein to nearest valid word
│ (+ Frequency analysis)          │   SLOW
└─────────────────────────────────┘
```

**Why cascading?** Efficiency. Layer 1 catches ~70-80% of words instantly (O(1) lookup). Only uncertain words fall through to more expensive layers. This makes the system practical for 1.77L+ word datasets.

#### Layer 1: Dictionary Lookup (`spelling/lexicon_loader.py`)

Three sources combined into a single lookup set:

**Source 1 — Core Hindi Word List** (~300 high-frequency words, built-in):
- Pronouns: मैं, हम, तुम, आप, वह, ये...
- Verbs (all major inflections): करता/करती/करते, जाता/जाती/जाते...
- Postpositions: का, की, के, को, से, में, पर...
- Adjectives, adverbs, conjunctions, numbers

**Source 2 — External Wordlist** (loaded from file, one word per line):
- The user can supply their own wordlist path
- All words are NFC-normalized before insertion

**Source 3 — NLTK Hindi Corpus** (if available):
- Automatically downloads `nltk.corpus.indian` Hindi data
- Adds academic Hindi word coverage

**Special Case: English Transliterations**
Per transcription guidelines, English words spoken in Hindi conversations are transcribed in Devanagari. These are CORRECT spellings:

| Devanagari | English |
|-----------|---------|
| कंप्यूटर | computer |
| मोबाइल | mobile |
| इंटरनेट | internet |
| यूट्यूब | youtube |
| व्हाट्सएप | whatsapp |

We maintain a map of 60+ common English-to-Devanagari transliterations.

#### Layer 2: Morphological Analysis (`spelling/classifier.py`)

Hindi is a **morphologically rich language** — verbs, nouns, and adjectives inflect heavily with suffixes. A word not in the dictionary might still be correct if it's a valid root + valid suffix.

**Algorithm:**
```
For each known Hindi suffix (sorted longest first):
  If word ends with suffix AND remaining root is in lexicon:
    → CORRECT (morphological decomposition found)
```

**Recognized suffix categories:**
- Verb: ता, ती, ते, ना, ेगा, ेगी, ेंगे...
- Noun/Adjective: ों, ियों, ियाँ, वाला, वाली, पन...
- Postposition clitics: को, से, में, पर, का, की, के...

**Example**: "चलाता" → root "चला" (in lexicon) + suffix "ता" → CORRECT

#### Layer 3: Unicode Validity (`spelling/unicode_validator.py`)

Even if a word looks like Hindi, it might have **impossible character sequences** in Devanagari. This layer catches gibberish.

**Rules checked:**
1. **No consecutive matras** without intervening consonant (e.g., ाी is invalid)
2. **Matra must follow consonant** (word can't start with a matra like ा)
3. **Virama (halant ्) must follow consonant** — used for conjuncts like क्ष
4. **Nukta (़) must follow consonant** — used for borrowed sounds like ज़

**Character classification system:**
- Consonants: क-ह (U+0915-U+0939) + nukta forms (क़, ख़, ग़, ज़, ड़, ढ़, फ़)
- Vowels: अ-औ (U+0904-U+0914)
- Matras (dependent vowels): ा-ौ (U+093E-U+094C)
- Modifiers: anusvara (ं), visarga (ः), chandrabindu (ँ)

If a word **fails Unicode validation** → classified as INCORRECT

#### Layer 4: Edit Distance (`spelling/classifier.py`)

For words that pass Unicode validation but aren't in the dictionary:

**Edit distance = 1** → Likely a **typo** (INCORRECT, with suggestion)
```
Example: "करत" → edit distance 1 from "करता" → INCORRECT (suggest "करता")
```

**Edit distance = 2** → UNCERTAIN (could be typo or rare valid word)

**Frequency analysis** (if available): If a word appears 5+ times in the corpus but isn't in the dictionary, it's likely valid (new slang, domain term, etc.)

**Performance optimization**: Only compare against lexicon words within ±2 characters of the target word's length (reduces search space dramatically). Uses `rapidfuzz` library for fast Levenshtein computation.

#### Final Classification Output

Each word gets:
```json
{
  "word": "करता",
  "classification": "correct",
  "reason": "Found in dictionary",
  "layer": 1,
  "suggestion": ""
}
```

```json
{
  "word": "करत",
  "classification": "incorrect",
  "reason": "Likely typo (edit distance 1 from 'करता')",
  "layer": 4,
  "suggestion": "करता"
}
```

**Three possible classifications:**
- `correct` — word is valid Hindi
- `incorrect` — word is definitely wrong (with suggested correction)
- `uncertain` — word can't be confirmed either way (rare/unknown)

---

<a id="question-4"></a>
## Question 4: Lattice-Based WER Evaluation

### What the Question Asks

Given transcription outputs from **5 ASR models** for the same audio AND a **human reference transcription** (which may contain errors), design an approach to:

1. **Construct a lattice** that captures all valid transcription alternatives from the model outputs
2. **Handle insertions, deletions, and substitutions** in a way that does not unfairly penalize models when the reference is wrong
3. **Decide when to trust model agreement** over the human reference
4. **Choose and justify the alignment unit** (word / subword / phrase)
5. **Compute WER** for each model using the lattice-based transcription — reducing WER for unfairly penalized models while keeping it unchanged for others

### Our Expert Approach

#### Key Insight: ROVER-Inspired Reference Correction

Standard WER assumes the human reference is **ground truth**. But human transcribers make mistakes:
- Mishearing words in noisy audio
- Inconsistent spelling of borrowed words
- Typos in transcription tools

When the reference is wrong and a model gets it right, the model is **unfairly penalized**. Our approach detects and corrects these cases using **ROVER (Recognizer Output Voting Error Reduction)** — a well-established technique in ASR research — repurposed for **reference correction**.

#### Step 1: Word-Level Alignment (`lattice/alignment.py`)

Each model's hypothesis is aligned to the reference using **classic Levenshtein edit-distance dynamic programming**.

**Algorithm (O(n×m) time, O(n×m) space):**

```
Build DP table dp[i][j] = min edits to transform ref[:i] → hyp[:j]

Base cases:
  dp[i][0] = i  (delete all ref words)
  dp[0][j] = j  (insert all hyp words)

Fill:
  If ref[i-1] == hyp[j-1]:  dp[i][j] = dp[i-1][j-1]     (correct, cost 0)
  Else: dp[i][j] = 1 + min(
    dp[i-1][j-1],  ← substitution
    dp[i-1][j],    ← deletion (ref word missing from hyp)
    dp[i][j-1],    ← insertion (hyp has extra word)
  )

Backtrace to get alignment sequence.
```

**Example:**
```
Reference:  ["मैं", "जा",   "रहा", "हूं"]
Hypothesis: ["मैं", "जाता",        "हूं"]

Alignment:
  ("मैं",  "मैं",   correct)
  ("जा",   "जाता",  substitution)
  ("रहा",  <eps>,   deletion)
  ("हूं",  "हूं",   correct)
```

#### Alignment Unit Justification: Why WORDS

| Alternative | Problem |
|-------------|---------|
| **Subwords** (BPE tokens) | Too fragmented for Hindi's complex morphology. BPE tokenization varies across models, making cross-model alignment inconsistent |
| **Characters** | Too fine-grained — creates noisy alignment with many spurious insertions/deletions |
| **Phrases** | Too coarse — reduces lattice resolution, making consensus unreliable. Hard to define phrase boundaries automatically |
| **Words** ✅ | Natural WER unit. Interpretable by humans. Consistent across models. Well-defined boundaries (whitespace) |

#### Step 2: Position Lattice Construction (`lattice/lattice_builder.py`)

The lattice is a graph where:
- Each **node** = a position in the transcript (numbered by reference word position)
- Each **edge** = a word that could appear at that position
- **Multiple parallel edges** = different models' hypotheses at the same position

```
Position │ Reference │ Model 1 │ Model 2 │ Model 3 │ Model 4 │ Model 5
─────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────
   0     │  मैं      │  मैं     │  मैं     │  मैं     │  मैं     │  मैं
   1     │  जाती     │  जाता    │  जाता    │  जाता    │  जाता    │  जाती
   2     │  हूं      │  हूं     │  हूं     │  हूं     │  हूं     │  हूं
```

**How it works:**
1. Align each of the 5 hypotheses to the reference (Step 1)
2. Use the reference as the **backbone** (position numbering)
3. For each position, record what each model produces:
   - `correct` → model produced the same word as reference
   - `substitution` → model produced a different word
   - `deletion` → model has nothing at this position (marked as `<eps>`)
   - `insertion` → model has an extra word (stored in a separate position like `2_ins_3`)

#### Step 3: Consensus Voting (`lattice/consensus.py`)

At each position, we count how many models agree on the same word:

```python
word_counts = Counter(non_epsilon_model_words)
most_common_word, count = word_counts.most_common(1)[0]

if count >= threshold:
    consensus = most_common_word  # Models agree!
```

**Consensus Threshold: K = 3 (majority of 5 models)**

| Threshold | Problem |
|-----------|---------|
| K = 2 (40%) | Too aggressive — only 2 models need to agree, many false corrections |
| K = 3 (60%) ✅ | Majority rule — clear majority disagrees with reference = likely error |
| K = 4 (80%) | Too conservative — misses cases where 1 model makes same error as reference |
| K = 5 (100%) | Too strict — unanimity almost never triggers |

**Reference Correction Logic:**
```
At each position:
  If consensus agrees WITH reference → keep reference word
  If consensus disagrees AND ≥ K models agree → replace reference with consensus
  If no consensus → keep reference (benefit of the doubt)
```

**Example**: Position 1 above — 4 models say "जाता" but reference says "जाती" → corrected to "जाता"

#### Step 4: WER Recomputation (`lattice/wer.py`)

After correcting the reference, we recompute WER for each model against the **corrected reference**:

```
Standard WER:  WER(original_reference, model_hypothesis)
Lattice WER:   WER(corrected_reference, model_hypothesis)
Delta:         Lattice WER - Standard WER
```

**Expected behavior:**
- Models that were **unfairly penalized** (because reference was wrong and they were right) → WER **decreases** ✅
- Models that **genuinely made errors** → WER stays approximately the **same**

**Output Report:**
```
| Model   | Standard WER | Lattice WER | Δ WER    |
|---------|-------------|-------------|----------|
| Model 1 | 33.33%      | 0.00%       | -33.33% ✅ |
| Model 2 | 33.33%      | 0.00%       | -33.33% ✅ |
| Model 3 | 33.33%      | 0.00%       | -33.33% ✅ |
| Model 4 | 33.33%      | 0.00%       | -33.33% ✅ |
| Model 5 | 0.00%       | 33.33%      | +33.33%    |
```

In this example, Models 1-4 agreed on "जाता" and were penalized for disagreeing with the (wrong) reference "जाती". After correction, their WER drops to 0%. Model 5 agreed with the original (wrong) reference, so its WER goes up.

#### Documented Limitations

1. **Small lattice**: Only 5 models → sparse lattice. More models would give more reliable consensus
2. **Correlated errors**: If multiple models are fine-tuned from the same base model, they may share biases (correlated errors look like consensus)
3. **Word-level granularity**: Hindi agglutinative words might differ only in suffix. "जाता" vs "जाती" are treated as fully different, even though they share the root "जा-"
4. **Insertion/deletion handling**: Insertions create model-specific positions that don't participate in general consensus

#### References
- Fiscus, J.G. (1997). "A post-processing system to yield reduced word error rates: ROVER."
- Evermann, G., & Woodland, P. (2000). "Large vocabulary decoding and confidence estimation."

---

## Overall Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│   data/download.py │ audio_utils.py │ text_utils.py     │
│   data/manifest.py │ config.py                         │
└─────────────────────────────────────────────────────────┘
         ↓                ↓                ↓
┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
│   Q1: Whisper  │ │   Q2:          │ │   Q3: Spelling   │
│   Fine-Tuning  │ │   Disfluency   │ │   Classifier     │
│   + WER Eval   │ │   Detection    │ │   (4-Layer)      │
│   training/    │ │   disfluency/  │ │   spelling/      │
│   evaluation/  │ │                │ │                  │
└────────────────┘ └────────────────┘ └──────────────────┘
                                      ┌──────────────────┐
                                      │   Q4: Lattice    │
                                      │   WER (ROVER)    │
                                      │   lattice/       │
                                      └──────────────────┘
```

## Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| 16kHz mono audio | Whisper's required input format |
| NFC Unicode normalization | Hindi has multiple valid encodings; NFC ensures consistency |
| Segment duration 0.5–30s | < 0.5s = noise; > 30s = Whisper struggles |
| WER + CER reporting | CER gives fairer picture for morphologically rich Hindi |
| 4-layer spelling classifier | Cascading from fast dictionary to slow edit-distance for efficiency |
| Word-level lattice alignment | Natural WER unit, interpretable, standard in ASR research |
| Consensus threshold K=3 | Majority vote from 5 models |
| FLEURS benchmark | Industry-standard, makes results comparable to published work |

## Research Mindset Checklist

- ✅ Each design decision is justified, not just implemented
- ✅ Data quality quantified before training (health report)
- ✅ WER computed on standardized, normalized text
- ✅ Failure cases documented (high-WER examples, uncertain spellings)
- ✅ Results reproducible from methodology descriptions
- ✅ Limitations explicitly stated for each module

---

*Document prepared by Ashish Chaturvedi — AI Researcher Intern Assignment, Josh Talks*

