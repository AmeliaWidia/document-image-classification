"""Konfigurasi terpusat untuk dataset dan eksperimen."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DOC_DATA_DIR", PROJECT_ROOT / "data"))
CHECKPOINT_DIR = Path(os.getenv("DOC_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoint"))
SUBMISSION_DIR = Path(os.getenv("DOC_SUBMISSION_DIR", PROJECT_ROOT / "submissions"))
REPORT_DIR = Path(os.getenv("DOC_REPORT_DIR", PROJECT_ROOT / "reports"))
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder", "form",
    "handwritten", "invoice", "letter", "memo", "news_article",
    "presentation", "questionnaire", "resume", "scientific_publication",
    "scientific_report", "specification",
]
NUM_CLASSES = 16
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
VALID_SIZE = 0.10
NUM_WORKERS = 2
PIN_MEMORY = True
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
WEIGHTS_FILENAME = "efficientnet_b0_rwightman-7f5810bc.pth"
WEIGHTS_PATH = os.getenv("DOC_WEIGHTS_PATH", "") or None

EXPERIMENTS = {
    "baseline": {
        "display_name": "Baseline", "dropout": 0.30,
        "backbone_lr": 1e-4, "head_lr": 8e-4, "weight_decay": 1e-4,
        "epochs": 10, "checkpoint": "best_document_efficientnet_b0_baseline.pth",
        "history": "baseline_history.json",
    },
    "tuning1": {
        "display_name": "Hyperparameter Tuning 1", "dropout": 0.40,
        "backbone_lr": 5e-5, "head_lr": 3e-4, "weight_decay": 5e-4,
        "epochs": 12, "checkpoint": "best_document_efficientnet_b0_tuning.pth",
        "history": "tuning1_history.json",
    },
    "tuning2": {
        "display_name": "Hyperparameter Tuning 2", "dropout": 0.35,
        "backbone_lr": 7.5e-5, "head_lr": 5e-4, "weight_decay": 2.5e-4,
        "epochs": 12, "checkpoint": "best_document_efficientnet_b0_tuning2.pth",
        "history": "tuning2_history.json",
    },
}

for folder in (CHECKPOINT_DIR, SUBMISSION_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
