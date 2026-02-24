"""
Word-Level Alignment (Dynamic Programming)
=============================================
Classic edit-distance alignment between reference and hypothesis sentences.
Same algorithm used internally for WER computation, exposed here for lattice building.

Alignment unit: WORDS (not subwords or phrases).
Justification: Words are the natural unit for WER, interpretable by humans,
and standard in ASR evaluation. Hindi's complex morphology makes subword
alignment too fragmented, while phrase-level alignment is too coarse.
"""

from typing import List, Tuple


# Operation types
CORRECT = "correct"
SUBSTITUTION = "substitution"
INSERTION = "insertion"
DELETION = "deletion"
EPSILON = "<eps>"  # placeholder for insertions/deletions in lattice


def align_hypothesis_to_reference(
    reference: List[str],
    hypothesis: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Align hypothesis to reference using dynamic programming (Levenshtein).
    
    Returns a list of (ref_word, hyp_word, operation) tuples.
    
    Operations:
    - "correct": ref_word == hyp_word
    - "substitution": ref_word != hyp_word  
    - "insertion": ref_word = <eps>, hyp has extra word
    - "deletion": hyp_word = <eps>, ref word was missed
    
    Example:
        ref:  ["मैं", "जा", "रहा", "हूं"]
        hyp:  ["मैं", "जाता", "हूं"]
        
        alignment: [
            ("मैं", "मैं", "correct"),
            ("जा", "जाता", "substitution"),
            ("रहा", "<eps>", "deletion"),
            ("हूं", "हूं", "correct"),
        ]
    """
    n = len(reference)
    m = len(hypothesis)

    # ── Build DP table ────────────────────────────────────────────
    # dp[i][j] = minimum edit operations to transform ref[:i] → hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i  # delete all ref words
    for j in range(m + 1):
        dp[0][j] = j  # insert all hyp words

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # correct, no cost
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i - 1][j] + 1,        # deletion
                    dp[i][j - 1] + 1,        # insertion
                )

    # ── Backtrace to get alignment ────────────────────────────────
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            alignment.append((reference[i - 1], hypothesis[j - 1], CORRECT))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((reference[i - 1], hypothesis[j - 1], SUBSTITUTION))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((reference[i - 1], EPSILON, DELETION))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append((EPSILON, hypothesis[j - 1], INSERTION))
            j -= 1
        else:
            # Fallback (shouldn't reach here)
            break

    alignment.reverse()
    return alignment


def compute_wer_from_alignment(
    alignment: List[Tuple[str, str, str]]
) -> dict:
    """
    Compute WER metrics from an alignment.
    
    Returns:
        {
            "wer": float (0-1),
            "substitutions": int,
            "deletions": int,
            "insertions": int,
            "correct": int,
            "total_ref_words": int,
        }
    """
    subs = sum(1 for _, _, op in alignment if op == SUBSTITUTION)
    dels = sum(1 for _, _, op in alignment if op == DELETION)
    ins = sum(1 for _, _, op in alignment if op == INSERTION)
    correct = sum(1 for _, _, op in alignment if op == CORRECT)

    total_ref = subs + dels + correct  # reference word count

    wer = (subs + dels + ins) / max(total_ref, 1)

    return {
        "wer": wer,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "correct": correct,
        "total_ref_words": total_ref,
    }
