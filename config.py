"""
Hindi ASR Project — Global Configuration
==========================================
Central configuration for all modules: paths, URLs, model settings, and constants.
"""

import os

# ─── Load environment variables from .env ─────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # dotenv not installed — env vars must be set manually

# API Keys (loaded from .env)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ─── Project Root ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Data Directories ────────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
AUDIO_DIR = os.path.join(RAW_DATA_DIR, "audio")
TRANSCRIPTION_DIR = os.path.join(RAW_DATA_DIR, "transcriptions")
METADATA_DIR = os.path.join(RAW_DATA_DIR, "metadata")
MANIFEST_DIR = os.path.join(PROCESSED_DATA_DIR, "manifests")

# ─── Output Directories (per module) ─────────────────────────────────
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "model")
DISFLUENCY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "disfluency")
DISFLUENCY_CLIPS_DIR = os.path.join(DISFLUENCY_OUTPUT_DIR, "clips")
SPELLING_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "spelling")
LATTICE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lattice")
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation")

# ─── Create all directories ──────────────────────────────────────────
for _dir in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, AUDIO_DIR, TRANSCRIPTION_DIR,
    METADATA_DIR, MANIFEST_DIR, MODEL_OUTPUT_DIR, DISFLUENCY_OUTPUT_DIR,
    DISFLUENCY_CLIPS_DIR, SPELLING_OUTPUT_DIR, LATTICE_OUTPUT_DIR,
    EVAL_OUTPUT_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# ─── Audio Settings ──────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper requires 16kHz
AUDIO_CHANNELS = 1           # Mono
AUDIO_FORMAT = "wav"
CLIP_BUFFER_SEC = 0.1        # Buffer added before/after clips

# ─── Segment Validation ──────────────────────────────────────────────
MIN_SEGMENT_DURATION = 0.5   # seconds — shorter is likely noise
MAX_SEGMENT_DURATION = 30.0  # seconds — Whisper limit
MIN_TEXT_LENGTH = 1           # minimum characters in transcript

# ─── GCS URL Template ────────────────────────────────────────────────
# Pattern: https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_{type}
GCS_BASE_URL = "https://storage.googleapis.com/upload_goai"

def build_gcs_url(user_id, recording_id, file_type):
    """
    Construct a GCS URL for audio/transcription/metadata.
    
    file_type: 'recording' | 'transcription' | 'metadata'
    Extension: .wav for recording, .json for transcription/metadata
    """
    ext = ".wav" if file_type == "recording" else ".json"
    return f"{GCS_BASE_URL}/{user_id}/{recording_id}_{file_type}{ext}"

# ─── Model Settings ──────────────────────────────────────────────────
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"

# ─── Training Hyperparameters ─────────────────────────────────────────
TRAINING_CONFIG = {
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 50,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
}

# ─── FLEURS Evaluation ───────────────────────────────────────────────
FLEURS_DATASET = "google/fleurs"
FLEURS_LANG_CODE = "hi_in"
FLEURS_SPLIT = "test"

# ─── Lattice WER Settings ────────────────────────────────────────────
CONSENSUS_THRESHOLD = 3      # out of 5 models must agree
NUM_ASR_MODELS = 5

# ─── Download Settings ───────────────────────────────────────────────
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30        # seconds
DOWNLOAD_BACKOFF = 2         # exponential backoff multiplier
