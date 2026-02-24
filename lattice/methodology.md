# Lattice-Based WER — Methodology & Justification

## Approach: ROVER-Inspired Reference Correction

Our lattice-based WER approach is inspired by **ROVER (Recognizer Output Voting Error Reduction)**, a well-established technique in ASR research. Instead of using ROVER's original purpose (combining multiple ASR outputs for better accuracy), we repurpose it for **reference correction** — identifying and fixing errors in the human reference transcript.

### Why Reference Correction Matters

Standard WER assumes the human reference is ground truth. But human transcribers make mistakes:
- Mishearing words in noisy audio
- Inconsistent spelling of borrowed words
- Typos in transcription tools

When the reference is wrong and a model gets it right, the model is **unfairly penalized**. Our approach detects and corrects these cases.

## Algorithm

### Step 1: Word-Level Alignment

All 5 model hypotheses are aligned to the reference using **classic edit-distance dynamic programming** (the same algorithm used to compute WER internally).

Alignment maps each reference word to the corresponding model output word, marking operations as:
- **Correct**: model produced the same word
- **Substitution**: model produced a different word
- **Deletion**: model missed a reference word
- **Insertion**: model produced an extra word

### Step 2: Position Lattice Construction

A lattice is built with the reference as the backbone:

```
Position │ Reference │ Model 1 │ Model 2 │ Model 3 │ Model 4 │ Model 5
─────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────
   0     │  मैं      │  मैं     │  मैं     │  मैं     │  मैं     │  मैं
   1     │  जाती     │  जाता    │  जाता    │  जाता    │  जाता    │  जाती
   2     │  हूं      │  हूं     │  हूं     │  हूं     │  हूं     │  हूं
```

### Step 3: Consensus Voting

At each position, we count how many models agree on the same word:
- If ≥ K models agree on a word **different from the reference** → likely reference error
- Default K = 3 (majority of 5 models)

In the example above, position 1: 4 models say "जाता" but reference says "जाती" → reference is likely wrong.

### Step 4: Reference Correction + WER Recomputation

The corrected reference replaces suspicious words with the consensus word. WER is then recomputed for each model against this corrected reference.

**Expected behavior:**
- Models that were unfairly penalized → WER **decreases** ✅
- Models that genuinely made errors → WER stays **same** or changes minimally

## Alignment Unit: Word-Level

### Our Choice: Words

### Justification

| Alternative | Problems |
|-------------|----------|
| **Subwords** | Too fragmented for Hindi's complex morphology. BPE tokenization varies across models, making cross-model alignment inconsistent. |
| **Characters** | Too fine-grained — creates noisy alignment with many spurious insertions/deletions. |
| **Phrases** | Too coarse — reduces lattice resolution, making consensus less reliable. Hard to define phrase boundaries automatically. |
| **Words** | Natural unit for WER (standard in ASR). Interpretable by humans. Consistent across models. Well-defined boundaries (whitespace). |

## Trust Threshold Rationale

- **K = 3** (out of 5): Majority rule. Ensures correction only happens when a clear majority disagrees with the reference.
- K = 2 would be too aggressive (only 40% agreement needed → many false corrections)
- K = 4 would be too conservative (80% agreement → misses cases where one model makes the same error as the reference)
- K = 5 (unanimity) is too strict — almost never triggers

## Limitations

1. **Small lattice**: With only 5 models, the lattice is sparse. More models would give more reliable consensus.
2. **Correlated errors**: If multiple models are fine-tuned from the same base model, they may share biases (correlated errors look like consensus).
3. **Word-level granularity**: Hindi agglutinative words might differ only in suffix. Word-level alignment treats "जाता" and "जाती" as fully different, even though they share the root "जा-".
4. **Insertion/deletion handling**: Insertions create model-specific positions that don't participate in general consensus, potentially missing some correction opportunities.

## References

- Fiscus, J.G. (1997). "A post-processing system to yield reduced word error rates: ROVER."
- Evermann, G., & Woodland, P. (2000). "Large vocabulary decoding and confidence estimation."
