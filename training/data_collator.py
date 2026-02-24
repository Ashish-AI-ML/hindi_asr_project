"""
Whisper Data Collator
======================
The most critical piece of the training pipeline — packages variable-length
audio features and text labels into uniform batches for the Trainer.

Key responsibilities:
1. Pad audio spectrograms to same length
2. Pad label token IDs, replacing padding with -100 (so loss ignores them)
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class WhisperDataCollator:
    """
    Data collator for Whisper sequence-to-sequence training.
    
    Think of this as a "packaging machine" that takes individual samples
    and creates uniform batches the model can process.
    
    Args:
        processor: WhisperProcessor (contains both feature extractor + tokenizer)
    """
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        For each sample in the batch:
        - input_features: log-mel spectrogram tensor
        - labels: list of token IDs
        """
        # ── Pad audio features ────────────────────────────────────
        # Feature extractor handles padding spectrograms to same length
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # ── Pad label sequences ───────────────────────────────────
        # Tokenizer pads label sequences; we then replace pad tokens with -100
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so cross-entropy loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If all sequences start with BOS token, remove it
        # (Whisper decoder handles BOS internally during generation)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
