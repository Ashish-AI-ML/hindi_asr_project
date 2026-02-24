"""
WER Evaluation on FLEURS Hindi
================================
Evaluates both the pretrained Whisper-small baseline and fine-tuned model
on the FLEURS Hindi test set, reporting WER and CER in a structured table.

Usage:
    python -m evaluation.wer_eval
    python -m evaluation.wer_eval --model_path outputs/model/final_model
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple

import torch
import jiwer
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_NAME, LANGUAGE, TASK,
    FLEURS_DATASET, FLEURS_LANG_CODE, FLEURS_SPLIT,
    EVAL_OUTPUT_DIR, SAMPLE_RATE,
)
from data.text_utils import normalize_for_wer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Model Loading ────────────────────────────────────────────────────

def load_whisper_model(model_path: str = None):
    """
    Load a Whisper model and processor.
    
    Args:
        model_path: path to fine-tuned model, or None for pretrained baseline
    
    Returns:
        (model, processor)
    """
    path = model_path if model_path else MODEL_NAME
    logger.info(f"Loading model from: {path}")

    processor = WhisperProcessor.from_pretrained(
        path if model_path else MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(path)

    # Set forced decoder IDs for Hindi
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    return model, processor


# ─── FLEURS Dataset Loading ──────────────────────────────────────────

def load_fleurs_test():
    """
    Load the Hindi portion of the FLEURS test dataset.
    
    FLEURS is a standard benchmark — using the same test set
    makes results comparable to published papers.
    """
    logger.info(f"Loading FLEURS {FLEURS_LANG_CODE} ({FLEURS_SPLIT})...")
    dataset = load_dataset(FLEURS_DATASET, FLEURS_LANG_CODE, split=FLEURS_SPLIT)
    logger.info(f"Loaded {len(dataset)} test samples")
    return dataset


# ─── Evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    model,
    processor,
    test_dataset,
    batch_size: int = 8,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Run inference on test set and compute WER + CER.
    
    Args:
        model: Whisper model
        processor: Whisper processor
        test_dataset: FLEURS test dataset
        batch_size: inference batch size
        max_samples: limit evaluation to N samples (for testing)
    
    Returns:
        {
            "wer": float,
            "cer": float,
            "num_samples": int,
            "predictions": [(reference, hypothesis), ...],
            "failures": [(index, reference, hypothesis), ...]  # worst cases
        }
    """
    device = next(model.parameters()).device
    all_references = []
    all_hypotheses = []
    failure_cases = []

    samples = test_dataset
    if max_samples:
        samples = test_dataset.select(range(min(max_samples, len(test_dataset))))

    logger.info(f"Evaluating on {len(samples)} samples...")

    for i in tqdm(range(len(samples)), desc="Evaluating"):
        item = samples[i]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        reference = normalize_for_wer(item["transcription"])

        if not reference.strip():
            continue

        try:
            # Prepare input
            input_features = processor.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
            ).input_features.to(device)

            # Generate
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language=LANGUAGE,
                    task=TASK,
                )

            # Decode
            hypothesis = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            hypothesis = normalize_for_wer(hypothesis)

            all_references.append(reference)
            all_hypotheses.append(hypothesis)

            # Track high-error cases for qualitative analysis
            sample_wer = jiwer.wer(reference, hypothesis) if hypothesis.strip() else 1.0
            if sample_wer > 0.5:  # > 50% WER
                failure_cases.append({
                    "index": i,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": f"{sample_wer*100:.1f}%",
                })

        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            continue

    if not all_references:
        logger.error("No valid samples evaluated!")
        return {"wer": 100.0, "cer": 100.0, "num_samples": 0}

    # Compute overall metrics
    overall_wer = jiwer.wer(all_references, all_hypotheses) * 100
    overall_cer = jiwer.cer(all_references, all_hypotheses) * 100

    results = {
        "wer": round(overall_wer, 2),
        "cer": round(overall_cer, 2),
        "num_samples": len(all_references),
        "num_failures": len(failure_cases),
        "top_failures": sorted(failure_cases, key=lambda x: x["wer"], reverse=True)[:20],
    }

    logger.info(f"  WER: {results['wer']}%")
    logger.info(f"  CER: {results['cer']}%")
    logger.info(f"  Samples: {results['num_samples']}")

    return results


# ─── Results Table ────────────────────────────────────────────────────

def build_wer_table(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a formatted results table.
    
    Args:
        results_dict: {"Model Name": {"wer": float, "cer": float, "num_samples": int}}
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 65)
    lines.append(f"{'Model':<30} {'WER (%)':<12} {'CER (%)':<12} {'Samples':<10}")
    lines.append("-" * 65)

    for model_name, metrics in results_dict.items():
        lines.append(
            f"{model_name:<30} {metrics['wer']:<12.2f} {metrics['cer']:<12.2f} "
            f"{metrics['num_samples']:<10}"
        )

    lines.append("=" * 65)
    return "\n".join(lines)


def build_wer_table_markdown(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """Build a Markdown-formatted results table."""
    lines = [
        "| Model | WER (%) | CER (%) | Samples |",
        "|-------|---------|---------|---------|",
    ]
    for model_name, metrics in results_dict.items():
        lines.append(
            f"| {model_name} | {metrics['wer']:.2f} | {metrics['cer']:.2f} | "
            f"{metrics['num_samples']} |"
        )
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on FLEURS Hindi")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (omit for baseline only)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Only evaluate the pretrained baseline",
    )
    args = parser.parse_args()

    # Load test data
    test_dataset = load_fleurs_test()

    results = {}

    # ── Baseline evaluation ───────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("   Evaluating BASELINE Whisper-small")
    logger.info("=" * 60)

    baseline_model, baseline_processor = load_whisper_model(None)
    baseline_results = evaluate_model(
        baseline_model, baseline_processor, test_dataset,
        max_samples=args.max_samples,
    )
    results["Whisper-small (Baseline)"] = baseline_results

    # Free memory
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Fine-tuned evaluation ─────────────────────────────────────
    if not args.baseline_only and args.model_path:
        logger.info("\n" + "=" * 60)
        logger.info(f"   Evaluating FINE-TUNED model: {args.model_path}")
        logger.info("=" * 60)

        ft_model, ft_processor = load_whisper_model(args.model_path)
        ft_results = evaluate_model(
            ft_model, ft_processor, test_dataset,
            max_samples=args.max_samples,
        )
        results["Whisper-small (Fine-tuned)"] = ft_results

        del ft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Print and save results ────────────────────────────────────
    table = build_wer_table(results)
    md_table = build_wer_table_markdown(results)

    print("\n" + table)
    print("\n### Markdown Format:\n")
    print(md_table)

    # Save results
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(EVAL_OUTPUT_DIR, "wer_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # Remove non-serializable fields
        save_results = {}
        for k, v in results.items():
            save_results[k] = {kk: vv for kk, vv in v.items()}
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    table_path = os.path.join(EVAL_OUTPUT_DIR, "wer_results_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# WER Evaluation Results\n\n")
        f.write(f"## FLEURS Hindi Test Set ({FLEURS_LANG_CODE})\n\n")
        f.write(md_table + "\n\n")
        f.write("### Failure Analysis\n\n")
        for model_name, res in results.items():
            f.write(f"#### {model_name}\n\n")
            f.write(f"High-error samples (WER > 50%):\n\n")
            for fail in res.get("top_failures", [])[:5]:
                f.write(f"- **Ref**: {fail['reference']}\n")
                f.write(f"  **Hyp**: {fail['hypothesis']}\n")
                f.write(f"  **WER**: {fail['wer']}\n\n")

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Table saved to {table_path}")


if __name__ == "__main__":
    main()
