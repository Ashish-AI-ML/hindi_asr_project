"""
Audio Utilities
================
Loading, resampling, duration checking, and segment clipping for audio files.
All audio operations standardize to 16kHz mono WAV (Whisper requirement).
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional

import librosa
import soundfile as sf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_RATE, CLIP_BUFFER_SEC

logger = logging.getLogger(__name__)


def load_audio(
    path: str, target_sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        path: path to audio file
        target_sr: target sample rate (default 16kHz)
    
    Returns:
        (waveform as np.ndarray, sample_rate)
    
    Raises:
        FileNotFoundError: if audio file doesn't exist
        RuntimeError: if audio loading fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = librosa.load(path, sr=target_sr, mono=True)
        logger.debug(f"Loaded {path}: {len(waveform)} samples at {sr}Hz")
        return waveform, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}")


def get_duration(path: str) -> float:
    """Get audio duration in seconds without loading entire file."""
    try:
        return librosa.get_duration(filename=path)
    except Exception as e:
        logger.error(f"Cannot get duration for {path}: {e}")
        return -1.0


def resample(
    waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Resample audio waveform to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def clip_audio(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    buffer_sec: float = CLIP_BUFFER_SEC,
    target_sr: int = SAMPLE_RATE,
) -> Optional[str]:
    """
    Extract an audio segment and save as WAV.
    
    Adds a small buffer before and after to avoid sharp cuts.
    Always outputs at target_sr (16kHz) mono.
    
    Args:
        audio_path: source audio file
        start_sec: segment start in seconds
        end_sec: segment end in seconds
        output_path: where to save the clipped audio
        buffer_sec: padding in seconds (default 0.1s)
        target_sr: output sample rate
    
    Returns:
        output_path on success, None on failure
    """
    try:
        # Load full audio
        waveform, sr = load_audio(audio_path, target_sr=target_sr)
        total_duration = len(waveform) / sr

        # Apply buffer (clamp to valid range)
        buffered_start = max(0.0, start_sec - buffer_sec)
        buffered_end = min(total_duration, end_sec + buffer_sec)

        # Convert to sample indices
        start_sample = int(buffered_start * sr)
        end_sample = int(buffered_end * sr)

        # Extract segment
        segment = waveform[start_sample:end_sample]

        if len(segment) == 0:
            logger.warning(
                f"Empty segment [{start_sec}s - {end_sec}s] in {audio_path}"
            )
            return None

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, segment, sr)
        logger.info(
            f"Clipped [{buffered_start:.2f}s - {buffered_end:.2f}s] -> {output_path}"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to clip audio {audio_path}: {e}")
        return None


def compute_snr(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Compute a simple Signal-to-Noise Ratio estimate.
    
    Uses the ratio of RMS of the loudest 10% to the quietest 10%.
    Higher = cleaner audio.
    
    Returns SNR in dB, or -inf if computation fails.
    """
    try:
        # Frame-level RMS
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)     # 10ms hop
        rms = librosa.feature.rms(
            y=waveform, frame_length=frame_length, hop_length=hop_length
        )[0]

        if len(rms) < 10:
            return float("-inf")

        sorted_rms = np.sort(rms)
        n = len(sorted_rms)

        # Noise estimate: bottom 10% RMS  
        noise_rms = np.mean(sorted_rms[: max(1, n // 10)]) + 1e-10
        # Signal estimate: top 10% RMS
        signal_rms = np.mean(sorted_rms[-(max(1, n // 10)):]) + 1e-10

        snr_db = 20 * np.log10(signal_rms / noise_rms)
        return float(snr_db)

    except Exception as e:
        logger.error(f"SNR computation failed: {e}")
        return float("-inf")


def validate_audio_file(path: str) -> Tuple[bool, str]:
    """
    Validate that an audio file is loadable and has reasonable properties.
    
    Returns (is_valid, reason).
    """
    if not os.path.exists(path):
        return False, "File not found"

    if os.path.getsize(path) == 0:
        return False, "Empty file"

    try:
        duration = get_duration(path)
        if duration <= 0:
            return False, f"Invalid duration: {duration}s"
        if duration > 7200:  # 2 hours — sanity check
            return False, f"Suspiciously long: {duration}s"
        return True, f"OK ({duration:.1f}s)"
    except Exception as e:
        return False, f"Load error: {e}"
