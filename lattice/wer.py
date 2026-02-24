"""
Lattice-Based WER Computation
================================
Computes both standard and lattice-corrected WER for each model.

Key idea: Models that were unfairly penalized (because the human reference
was wrong) should see REDUCED WER under the corrected reference.
Models that genuinely made errors → WER stays same or changes minimally.

Usage:
    python -m lattice.wer --input path/to/model_outputs.json
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LATTICE_OUTPUT_DIR, CONSENSUS_THRESHOLD
from data.text_utils import normalize_for_wer
from lattice.alignment import (
    align_hypothesis_to_reference,
    compute_wer_from_alignment,
)
from lattice.lattice_builder import build_position_lattice, lattice_to_readable
from lattice.consensus import (
    correct_reference, analyze_corrections,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_standard_wer(
    reference: List[str],
    hypothesis: List[str],
) -> Dict[str, Any]:
    """Compute standard WER between reference and hypothesis."""
    alignment = align_hypothesis_to_reference(reference, hypothesis)
    return compute_wer_from_alignment(alignment)


def compute_lattice_wer(
    corrected_reference: List[str],
    hypothesis: List[str],
) -> Dict[str, Any]:
    """Compute WER using the lattice-corrected reference."""
    alignment = align_hypothesis_to_reference(corrected_reference, hypothesis)
    return compute_wer_from_alignment(alignment)


def evaluate_all_models(
    reference_text: str,
    model_outputs: Dict[str, str],
    threshold: int = CONSENSUS_THRESHOLD,
) -> Dict[str, Any]:
    """
    Full lattice-based evaluation for all models.
    
    Args:
        reference_text: human reference transcription
        model_outputs: dict mapping model_name → hypothesis_text
        threshold: consensus threshold for reference correction
    
    Returns:
        {
            "standard_wer": {model_name: wer, ...},
            "lattice_wer": {model_name: wer, ...},
            "corrected_reference": str,
            "corrections": [...],
            "analysis": {...},
        }
    """
    # Normalize texts
    ref_normalized = normalize_for_wer(reference_text)
    ref_words = ref_normalized.split()

    model_names = list(model_outputs.keys())
    hyp_texts = {}
    hyp_word_lists = []

    for name in model_names:
        normalized = normalize_for_wer(model_outputs[name])
        hyp_texts[name] = normalized
        hyp_word_lists.append(normalized.split())

    # ── Step 1: Compute standard WER ──────────────────────────────
    standard_results = {}
    for name, hyp_words in zip(model_names, hyp_word_lists):
        wer_result = compute_standard_wer(ref_words, hyp_words)
        standard_results[name] = wer_result

    # ── Step 2: Build lattice ─────────────────────────────────────
    lattice = build_position_lattice(ref_words, hyp_word_lists)

    # ── Step 3: Correct reference ─────────────────────────────────
    corrected_ref, correction_log = correct_reference(
        ref_words, lattice, threshold
    )

    # ── Step 4: Compute lattice WER ───────────────────────────────
    lattice_results = {}
    for name, hyp_words in zip(model_names, hyp_word_lists):
        wer_result = compute_lattice_wer(corrected_ref, hyp_words)
        lattice_results[name] = wer_result

    # ── Analysis ──────────────────────────────────────────────────
    analysis = analyze_corrections(correction_log)

    return {
        "reference": ref_normalized,
        "corrected_reference": " ".join(corrected_ref),
        "standard_wer": {
            name: {"wer": f"{r['wer']*100:.2f}%", **r}
            for name, r in standard_results.items()
        },
        "lattice_wer": {
            name: {"wer": f"{r['wer']*100:.2f}%", **r}
            for name, r in lattice_results.items()
        },
        "corrections": correction_log,
        "analysis": analysis,
        "lattice_readable": lattice_to_readable(lattice),
    }


def generate_wer_report(
    results: Dict[str, Any],
) -> str:
    """Generate a formatted WER comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  LATTICE-BASED WER EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Reference: {results['reference']}")
    lines.append(f"Corrected: {results['corrected_reference']}")
    lines.append(f"Corrections made: {results['analysis']['total_corrections']}")
    lines.append("")

    # Standard WER table
    lines.append("─" * 80)
    lines.append(f"{'Model':<25} {'Standard WER':<15} {'Lattice WER':<15} {'Δ WER':<10}")
    lines.append("─" * 80)

    for model_name in results["standard_wer"]:
        std_wer = results["standard_wer"][model_name]["wer"]
        lat_wer = results["lattice_wer"][model_name]["wer"]

        # Parse percentages for delta
        std_val = float(std_wer.replace("%", ""))
        lat_val = float(lat_wer.replace("%", ""))
        delta = lat_val - std_val

        delta_str = f"{delta:+.2f}%"
        if delta < 0:
            delta_str += " ✅"  # improved
        elif delta == 0:
            delta_str += " ─"   # unchanged

        lines.append(f"{model_name:<25} {std_wer:<15} {lat_wer:<15} {delta_str:<10}")

    lines.append("─" * 80)

    # Corrections detail
    if results["corrections"]:
        lines.append("")
        lines.append("Reference Corrections:")
        for corr in results["corrections"]:
            lines.append(
                f"  Position {corr['position']}: "
                f"'{corr['original_ref']}' → '{corr['corrected_to']}' "
                f"({corr['agreement']} models agree)"
            )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def generate_wer_report_markdown(results: Dict[str, Any]) -> str:
    """Generate Markdown-formatted WER report."""
    md = []
    md.append("# Lattice-Based WER Evaluation\n")
    md.append("## Methodology\n")
    md.append("We use a ROVER-inspired approach to construct a lattice from 5 ASR model outputs,\n")
    md.append("compute consensus at each word position, and correct the human reference where\n")
    md.append(f"≥{CONSENSUS_THRESHOLD} models agree on a different word.\n\n")
    
    md.append("### Alignment Unit Justification\n")
    md.append("**Words** are used as the alignment unit because:\n")
    md.append("- Natural unit for WER computation (standard in ASR evaluation)\n")
    md.append("- Interpretable by humans for error analysis\n")
    md.append("- Hindi's complex morphology makes subword alignment too fragmented\n")
    md.append("- Phrase-level alignment is too coarse for accurate consensus\n\n")

    md.append("## Results\n\n")
    md.append(f"**Reference**: {results['reference']}\n\n")
    md.append(f"**Corrected Reference**: {results['corrected_reference']}\n\n")
    md.append(f"**Corrections**: {results['analysis']['total_corrections']}\n\n")

    md.append("| Model | Standard WER | Lattice WER | Δ WER |\n")
    md.append("|-------|-------------|-------------|-------|\n")

    for model_name in results["standard_wer"]:
        std_wer = results["standard_wer"][model_name]["wer"]
        lat_wer = results["lattice_wer"][model_name]["wer"]
        std_val = float(std_wer.replace("%", ""))
        lat_val = float(lat_wer.replace("%", ""))
        delta = lat_val - std_val
        delta_str = f"{delta:+.2f}%"
        md.append(f"| {model_name} | {std_wer} | {lat_wer} | {delta_str} |\n")

    if results["corrections"]:
        md.append("\n## Reference Corrections\n\n")
        md.append("| Position | Original | Corrected | Agreement |\n")
        md.append("|----------|----------|-----------|----------|\n")
        for corr in results["corrections"]:
            md.append(
                f"| {corr['position']} | {corr['original_ref']} | "
                f"{corr['corrected_to']} | {corr['agreement']} |\n"
            )

    return "".join(md)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lattice-based WER evaluation")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSON file with reference + model outputs",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CONSENSUS_THRESHOLD,
        help=f"Consensus threshold (default {CONSENSUS_THRESHOLD})",
    )
    args = parser.parse_args()

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expected format:
    # {
    #   "reference": "...",
    #   "models": {
    #     "model_1": "hypothesis text",
    #     "model_2": "hypothesis text",
    #     ...
    #   }
    # }
    # OR a list of such entries for multiple sentences

    if isinstance(data, list):
        # Multiple sentences
        all_results = []
        for i, entry in enumerate(data):
            logger.info(f"\n--- Sentence {i+1}/{len(data)} ---")
            result = evaluate_all_models(
                entry["reference"],
                entry["models"],
                threshold=args.threshold,
            )
            all_results.append(result)
            print(generate_wer_report(result))

        # Save all results
        os.makedirs(LATTICE_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(LATTICE_OUTPUT_DIR, "lattice_wer_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    else:
        # Single sentence
        result = evaluate_all_models(
            data["reference"],
            data["models"],
            threshold=args.threshold,
        )

        print(generate_wer_report(result))

        # Save results
        os.makedirs(LATTICE_OUTPUT_DIR, exist_ok=True)

        json_path = os.path.join(LATTICE_OUTPUT_DIR, "lattice_wer_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        md_path = os.path.join(LATTICE_OUTPUT_DIR, "lattice_wer_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_wer_report_markdown(result))

        logger.info(f"Results saved to {json_path}")
        logger.info(f"Report saved to {md_path}")


if __name__ == "__main__":
    main()
