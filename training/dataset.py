"""
Whisper Hindi Dataset
======================
HuggingFace-compatible dataset for fine-tuning Whisper on Hindi audio.
Wraps the manifest and applies feature extraction + tokenization on-the-fly.
"""

import os
import logging
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.audio_utils import load_audio, clip_audio
from data.text_utils import clean_text
from config import SAMPLE_RATE, LANGUAGE, TASK

logger = logging.getLogger(__name__)


class WhisperHindiDataset(Dataset):
    """
    Dataset that loads audio segments and prepares them for Whisper fine-tuning.
    
    Each item returns:
        - input_features: log-mel spectrogram (from WhisperFeatureExtractor)
        - labels: tokenized text (from WhisperTokenizer)
    """

    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        feature_extractor,
        tokenizer,
        max_audio_len: float = 30.0,
    ):
        """
        Args:
            manifest: list of dicts with 'audio_path', 'text', 'start', 'end', 'duration'
            feature_extractor: WhisperFeatureExtractor instance
            tokenizer: WhisperTokenizer instance
            max_audio_len: maximum audio length in seconds
        """
        self.manifest = manifest
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.manifest[idx]
        audio_path = entry["audio_path"]
        start = entry["start"]
        end = entry["end"]
        text = entry["text"]

        try:
            # Load and extract the segment from full audio
            waveform, sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
            
            # Extract segment samples
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = waveform[start_sample:end_sample]

            # Truncate if too long
            max_samples = int(self.max_audio_len * sr)
            if len(segment_audio) > max_samples:
                segment_audio = segment_audio[:max_samples]

            # Extract log-mel features
            input_features = self.feature_extractor(
                segment_audio,
                sampling_rate=sr,
                return_tensors="pt",
            ).input_features[0]

            # Tokenize text
            labels = self.tokenizer(text).input_ids

            return {
                "input_features": input_features,
                "labels": labels,
            }

        except Exception as e:
            logger.error(f"Error loading item {idx} ({audio_path}): {e}")
            # Return a dummy item to avoid crashing the training loop
            dummy_audio = torch.zeros(1, dtype=torch.float32)
            input_features = self.feature_extractor(
                dummy_audio.numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features[0]
            labels = self.tokenizer("").input_ids
            return {"input_features": input_features, "labels": labels}


class FLEURSHindiDataset(Dataset):
    """
    Wrapper around the FLEURS Hindi test set for evaluation.
    Applies feature extraction on-the-fly.
    """

    def __init__(self, hf_dataset, feature_extractor, tokenizer):
        """
        Args:
            hf_dataset: HuggingFace dataset (google/fleurs hi_in test split)
            feature_extractor: WhisperFeatureExtractor
            tokenizer: WhisperTokenizer
        """
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        text = clean_text(item["transcription"])

        # Resample if needed (FLEURS is usually 16kHz)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        input_features = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features[0]

        labels = self.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "reference_text": text,
        }
