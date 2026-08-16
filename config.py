"""Central configuration for the Signity application."""

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Model / inference thresholds
MIN_CONFIDENCE = float(os.environ.get("SIGNITY_MIN_CONFIDENCE", "0.50"))
MIN_MARGIN = float(os.environ.get("SIGNITY_MIN_MARGIN", "0.02"))
MIN_VALID_FRAME_RATIO = float(os.environ.get("SIGNITY_MIN_VALID_FRAME_RATIO", "0.40"))
MAX_SEQUENCE_FRAMES = 32
INFERENCE_DEBUG = os.environ.get("SIGNITY_INFERENCE_DEBUG") == "1"

# Model asset paths
MODEL_PATH = BASE_DIR / "model.keras"
LEGACY_MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
TASK_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# Database
DATABASE_PATH = Path(os.environ.get("SIGNITY_DATABASE", BASE_DIR / "signity.db"))

# Flask / session security
SECRET_KEY = os.environ.get("SIGNITY_SECRET_KEY") or secrets.token_urlsafe(32)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = os.environ.get("SIGNITY_SECURE_COOKIES") == "1"
MAX_CONTENT_LENGTH = 12 * 1024 * 1024

# Server defaults
DEFAULT_HOST = os.environ.get("SIGNITY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SIGNITY_PORT", "5500"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
