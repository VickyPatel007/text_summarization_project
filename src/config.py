from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
FINETUNED_MODEL = MODEL_DIR / "text-summarizer"

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
MIN_SUMMARY_LENGTH = 20
MAX_SUMMARY_LENGTH = 128

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
