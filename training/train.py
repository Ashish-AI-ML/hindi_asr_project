"""
Whisper Fine-Tuning Script
============================
Fine-tunes openai/whisper-small on Hindi ASR data.

Usage:
    python -m training.train --manifest processed_data/manifests/train_manifest.json

Key decisions documented:
- Learning rate 1e-5: prevents catastrophic forgetting of English knowledge
- Gradient accumulation 2: simulates batch size 16 on limited GPU
- fp16: faster training on CUDA GPUs
- forced_decoder_ids for Hindi: ensures Hindi decoding, not English
"""

import os
import sys
import json
import argparse
import logging
from functools import partial

import torch
import jiwer
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_NAME, LANGUAGE, TASK, MODEL_OUTPUT_DIR,
    TRAINING_CONFIG, MANIFEST_DIR,
)
from data.manifest import load_manifest
from data.text_utils import normalize_for_wer
from training.dataset import WhisperHindiDataset
from training.data_collator import WhisperDataCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(pred, tokenizer):
    """
    Compute WER during training evaluation.
    
    This function:
    1. Decodes predicted token IDs back to text
    2. Decodes reference label IDs back to text
    3. Normalizes both for fair WER comparison
    4. Computes WER using jiwer
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 (padding) with pad_token_id for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize for WER
    pred_str = [normalize_for_wer(p) for p in pred_str]
    label_str = [normalize_for_wer(l) for l in label_str]

    # Filter out empty references (would cause division by zero)
    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return {"wer": 1.0}

    pred_filtered, label_filtered = zip(*pairs)

    wer = jiwer.wer(list(label_filtered), list(pred_filtered))
    return {"wer": wer * 100}  # as percentage


def setup_model_and_processor():
    """
    Load Whisper model and processor with Hindi configuration.
    
    Critical settings:
    - language="hi": forces Hindi decoding
    - task="transcribe": transcription mode (not translation)
    - forced_decoder_ids: ensures language/task tokens in every generation
    """
    logger.info(f"Loading model: {MODEL_NAME}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Set forced decoder IDs for Hindi transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    # Enable gradient checkpointing to save memory
    model.config.use_cache = False

    logger.info(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    return model, processor


def split_manifest(manifest, eval_ratio=0.1):
    """Split manifest into train/eval sets."""
    import random
    random.seed(42)
    shuffled = manifest.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on Hindi")
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(MANIFEST_DIR, "train_manifest.json"),
        help="Path to training manifest JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MODEL_OUTPUT_DIR,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    logger.info(f"Loading manifest from {args.manifest}")
    manifest = load_manifest(os.path.basename(args.manifest))
    train_manifest, eval_manifest = split_manifest(manifest, args.eval_ratio)
    logger.info(f"Train: {len(train_manifest)}, Eval: {len(eval_manifest)}")

    # ── Setup model ───────────────────────────────────────────────
    model, processor = setup_model_and_processor()

    # ── Create datasets ───────────────────────────────────────────
    train_dataset = WhisperHindiDataset(
        manifest=train_manifest,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )
    eval_dataset = WhisperHindiDataset(
        manifest=eval_manifest,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )

    # ── Data collator ─────────────────────────────────────────────
    data_collator = WhisperDataCollator(processor=processor)

    # ── Training arguments ────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        fp16=TRAINING_CONFIG["fp16"] and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        greater_is_better=TRAINING_CONFIG["greater_is_better"],
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=processor.tokenizer),
        tokenizer=processor.feature_extractor,  # for padding
    )

    # ── Train ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("   Starting Whisper-small Hindi fine-tuning")
    logger.info("=" * 60)

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # ── Save final model ──────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    # ── Final evaluation ──────────────────────────────────────────
    results = trainer.evaluate()
    logger.info(f"Final evaluation results: {results}")

    # Save results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
