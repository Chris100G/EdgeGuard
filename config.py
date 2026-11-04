"""
EdgeGuard - Configuration Module
Centralized settings and paths for EdgeGuard TinyML Privacy System.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"

# Ensure directories exist
for directory in [MODELS_DIR, ASSETS_DIR, DATA_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DEFAULT_EVENTS_CSV = DATA_DIR / "events.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"
CSS_PATH = ASSETS_DIR / "styles.css"

# Model Registry
# Note: SSD MobileNet V2 INT8 is our primary quantized model (~6.2 MB)
# EfficientDet-Lite0 Float32 (~4.5 MB) is our secondary high-resolution detector
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ssd_mobilenet_v2_quant": {
        "name": "SSD MobileNet V2 (INT8 Quantized)",
        "filename": "ssd_mobilenet_v2_quant.tflite",
        "path": MODELS_DIR / "ssd_mobilenet_v2_quant.tflite",
        "input_size": (300, 300),
        "quantized": True,
        "precision": "INT8 / UINT8",
        "person_class_id": 0.0,
        "description": "Ultra-fast INT8 quantized detector optimized for low-power edge CPUs.",
        "url": "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    },
    "efficientdet_lite0": {
        "name": "EfficientDet-Lite0 (Float32)",
        "filename": "efficientdet_lite0.tflite",
        "path": MODELS_DIR / "efficientdet_lite0.tflite",
        "input_size": (320, 320),
        "quantized": False,
        "precision": "FP32",
        "person_class_id": 0.0,
        "description": "High-accuracy Google EfficientDet architecture for edge devices.",
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite"
    },
    "ssd_mobilenet_v1_quant": {
        "name": "SSD MobileNet V1 (INT8 Quantized)",
        "filename": "ssd_mobilenet_v1_quant.tflite",
        "path": MODELS_DIR / "ssd_mobilenet_v1_quant.tflite",
        "input_size": (300, 300),
        "quantized": True,
        "precision": "UINT8",
        "person_class_id": 0.0,
        "description": "Lightweight quantized mobile detector for minimal resource footprints.",
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    }
}

DEFAULT_MODEL_KEY = "ssd_mobilenet_v2_quant"

# Risk Engine Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
MIN_CONFIDENCE_THRESHOLD = 0.20
MAX_CONFIDENCE_THRESHOLD = 0.95

# Temporal Smoothing & Stability
DEFAULT_CONFIRMATION_FRAMES = 2  # Consecutive frames required to flip state
DEFAULT_ALERT_COOLDOWN_SEC = 3.5  # Min seconds between audio alerts for ongoing violation

# Performance & Throttling
DEFAULT_FRAME_INTERVAL = 1  # Process every Nth frame (1 = every frame, 2 = every 2nd frame)
TARGET_FPS = 30

# Risk State Constants
STATE_SAFE = "SAFE"
STATE_RISK = "POTENTIAL_PRIVACY_RISK"
STATE_ERROR = "SYSTEM_ERROR"
STATE_INACTIVE = "MONITORING_INACTIVE"

# Privacy Guarantees
PRIVACY_DISCLAIMER = (
    "Your camera frames are processed locally. EdgeGuard does not upload or save video recordings."
)

SECURITY_NOTICE = (
    "EdgeGuard is a privacy-awareness system that detects human presence to alert against visual eavesdropping "
    "(shoulder surfing). It does NOT perform facial recognition, biometric identity verification, or screen reading."
)

