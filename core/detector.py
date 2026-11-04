"""
EdgeGuard - Detector Module
Real-Time TinyML Person Detection using TensorFlow Lite / LiteRT.
Runs 100% locally with low CPU latency and no external cloud calls.
"""

import time
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import cv2

from config import MODEL_REGISTRY, DEFAULT_MODEL_KEY, DEFAULT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Dynamic import: Try LiteRT first (Google's official next-gen TFLite runtime),
# then fallback to tflite_runtime, then tensorflow.lite
Interpreter = None
BACKEND_NAME = "Unknown"

try:
    from ai_edge_litert.interpreter import Interpreter
    BACKEND_NAME = "Google LiteRT (ai-edge-litert)"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        BACKEND_NAME = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            BACKEND_NAME = "TensorFlow Lite"
        except ImportError:
            pass


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates [0.0 - 1.0]."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def to_pixel_coords(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to integer pixel coordinates (xmin, ymin, xmax, ymax)."""
        left = max(0, int(self.xmin * frame_width))
        top = max(0, int(self.ymin * frame_height))
        right = min(frame_width - 1, int(self.xmax * frame_width))
        bottom = min(frame_height - 1, int(self.ymax * frame_height))
        return left, top, right, bottom


@dataclass
class Detection:
    """Represents a single detected person."""
    box: BoundingBox
    confidence: float
    class_id: int
    label: str = "person"


@dataclass
class DetectionResult:
    """Output from a single detection pass."""
    detections: List[Detection]
    person_count: int
    inference_latency_ms: float
    model_name: str
    is_quantized: bool
    model_size_mb: float


class PersonDetector:
    """
    Local TinyML Person Detector leveraging TensorFlow Lite / LiteRT.
    Designed for real-time edge execution with zero cloud telemetry.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        if Interpreter is None:
            raise RuntimeError(
                "No compatible TensorFlow Lite / LiteRT runtime found. "
                "Please install 'ai-edge-litert' or 'tflite-runtime'."
            )

        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

        self.model_key = model_key
        self.model_info = MODEL_REGISTRY[model_key]
        self.confidence_threshold = confidence_threshold
        self.model_path = Path(self.model_info["path"])

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        self.model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        self.is_quantized = self.model_info.get("quantized", False)
        self.target_class_id = self.model_info.get("person_class_id", 0.0)

        # Initialize Interpreter
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        # Input & Output specs
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]  # e.g., [1, 300, 300, 3]
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_height = int(self.input_shape[1])
        self.input_width = int(self.input_shape[2])
        self.input_index = self.input_details[0]["index"]

        # Cache output indices
        self._inspect_outputs()

        # Warm up interpreter
        self._warmup()
        logger.info(
            f"Initialized {self.model_info['name']} ({self.model_size_mb:.2f} MB, {self.input_dtype.__name__}) "
            f"via {BACKEND_NAME}"
        )

    def _inspect_outputs(self) -> None:
        """Map output tensor indices for boxes, classes, scores, and count."""
        # Standard TFLite Object Detection models output 4 tensors:
        # [0] Boxes: [1, N, 4]
        # [1] Classes: [1, N]
        # [2] Scores: [1, N]
        # [3] Count: [1]
        self.box_idx = None
        self.class_idx = None
        self.score_idx = None
        self.count_idx = None

        for output in self.output_details:
            shape = output["shape"]
            if len(shape) == 3 and shape[-1] == 4:
                self.box_idx = output["index"]
            elif len(shape) == 2:
                # Distinguish between classes and scores based on name or dtype if available,
                # otherwise order: index 1 is typically classes or scores
                name = output.get("name", "").lower()
                if "class" in name:
                    self.class_idx = output["index"]
                elif "score" in name:
                    self.score_idx = output["index"]
            elif len(shape) == 1 and shape[0] == 1:
                self.count_idx = output["index"]

        # Fallback to positional mapping if names don't match (standard TFLite SSD/EfficientDet layout)
        if None in (self.box_idx, self.class_idx, self.score_idx, self.count_idx):
            self.box_idx = self.output_details[0]["index"]
            self.class_idx = self.output_details[1]["index"]
            self.score_idx = self.output_details[2]["index"]
            self.count_idx = self.output_details[3]["index"]

    def _warmup(self, iterations: int = 2) -> None:
        """Warm up interpreter to initialize XNNPACK delegates and memory allocations."""
        dummy = np.zeros(self.input_shape, dtype=self.input_dtype)
        for _ in range(iterations):
            self.interpreter.set_tensor(self.input_index, dummy)
            self.interpreter.invoke()

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model ingestion:
        - Convert BGR to RGB
        - Resize to expected input dimensions
        - Format dtype and expand batch dimension
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        if self.input_dtype == np.float32:
            input_data = (resized.astype(np.float32) / 127.5) - 1.0
        elif self.input_dtype == np.uint8:
            input_data = resized.astype(np.uint8)
        else:
            input_data = resized.astype(self.input_dtype)

        return np.expand_dims(input_data, axis=0)

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: Optional[float] = None) -> DetectionResult:
        """
        Run person detection on a single video frame.
        Filters specifically for human presence (class 0: person).
        Returns filtered detections and measured latency.
        """
        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # 1. Preprocess
        input_tensor = self.preprocess(frame_bgr)

        # 2. Inference
        t_start = time.perf_counter()
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # 3. Fetch Output Tensors
        raw_boxes = self.interpreter.get_tensor(self.box_idx)[0]
        raw_classes = self.interpreter.get_tensor(self.class_idx)[0]
        raw_scores = self.interpreter.get_tensor(self.score_idx)[0]

        try:
            num_detections = int(self.interpreter.get_tensor(self.count_idx)[0])
        except Exception:
            num_detections = len(raw_scores)

        # 4. Filter for 'person' class above confidence threshold
        person_detections: List[Detection] = []
        for i in range(num_detections):
            score = float(raw_scores[i])
            cls_id = float(raw_classes[i])

            if score >= thresh and cls_id == self.target_class_id:
                # Bounding box coordinates: [ymin, xmin, ymax, xmax]
                ymin = max(0.0, min(1.0, float(raw_boxes[i][0])))
                xmin = max(0.0, min(1.0, float(raw_boxes[i][1])))
                ymax = max(0.0, min(1.0, float(raw_boxes[i][2])))
                xmax = max(0.0, min(1.0, float(raw_boxes[i][3])))

                # Validate valid box area
                if ymax > ymin and xmax > xmin:
                    person_detections.append(
                        Detection(
                            box=BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax),
                            confidence=score,
                            class_id=int(cls_id),
                            label="person",
                        )
                    )

        # Sort detections by confidence descending
        person_detections.sort(key=lambda d: d.confidence, reverse=True)

        return DetectionResult(
            detections=person_detections,
            person_count=len(person_detections),
            inference_latency_ms=latency_ms,
            model_name=self.model_info["name"],
            is_quantized=self.is_quantized,
            model_size_mb=self.model_size_mb,
        )

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        is_risk: bool = False,
    ) -> np.ndarray:
        """
        Draw cybersecurity-styled bounding boxes, confidence tags, and reticles on frame.
        Safe color: Cyan/Green (#10B981 / #06B6D4)
        Risk color: Amber/Red (#F59E0B / #EF4444)
        """
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        if is_risk:
            # Amber/Orange for privacy risk
            box_color = (11, 158, 245)      # BGR for Amber #F59E0B
            tag_bg_color = (15, 23, 42)     # Dark slate
            text_color = (255, 255, 255)
            badge = "POTENTIAL OBSERVER"
        else:
            # Electric Cyan / Emerald for Safe
            box_color = (212, 182, 6)       # BGR for Cyan #06B6D4
            tag_bg_color = (15, 23, 42)
            text_color = (255, 255, 255)
            badge = "PRIMARY USER"

        for idx, det in enumerate(detections):
            left, top, right, bottom = det.box.to_pixel_coords(w, h)

            # Draw outer bounding box
            cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2, cv2.LINE_AA)

            # Corner bracket accents for modern cybersecurity look
            bracket_len = min(20, (right - left) // 4, (bottom - top) // 4)
            corner_color = (255, 255, 255)
            # Top-left
            cv2.line(annotated, (left, top), (left + bracket_len, top), corner_color, 2)
            cv2.line(annotated, (left, top), (left, top + bracket_len), corner_color, 2)
            # Top-right
            cv2.line(annotated, (right, top), (right - bracket_len, top), corner_color, 2)
            cv2.line(annotated, (right, top), (right, top + bracket_len), corner_color, 2)
            # Bottom-left
            cv2.line(annotated, (left, bottom), (left + bracket_len, bottom), corner_color, 2)
            cv2.line(annotated, (left, bottom), (left, bottom - bracket_len), corner_color, 2)
            # Bottom-right
            cv2.line(annotated, (right, bottom), (right - bracket_len, bottom), corner_color, 2)
            cv2.line(annotated, (right, bottom), (right, bottom - bracket_len), corner_color, 2)

            # Label text
            tag_label = f"{badge if idx > 0 and is_risk else ('PRIMARY' if idx == 0 else badge)}: {det.confidence*100:.1f}%"
            (font_w, font_h), baseline = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Label badge background
            label_ymin = max(0, top - font_h - 8)
            label_ymax = max(font_h + 8, top)
            cv2.rectangle(annotated, (left, label_ymin), (left + font_w + 10, label_ymax), tag_bg_color, -1)
            cv2.rectangle(annotated, (left, label_ymin), (left + font_w + 10, label_ymax), box_color, 1)

            # Label text
            cv2.putText(
                annotated,
                tag_label,
                (left + 5, label_ymax - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    def benchmark(self, iterations: int = 30) -> Dict[str, Any]:
        """
        Run genuine local latency and throughput benchmark.
        Returns measured latency statistics on the current hardware.
        """
        dummy = np.zeros(self.input_shape, dtype=self.input_dtype)

        # Warmup
        for _ in range(5):
            self.interpreter.set_tensor(self.input_index, dummy)
            self.interpreter.invoke()

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.interpreter.set_tensor(self.input_index, dummy)
            self.interpreter.invoke()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        std_ms = float(np.std(latencies))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "model_name": self.model_info["name"],
            "model_size_mb": self.model_size_mb,
            "precision": self.model_info.get("precision", "Unknown"),
            "quantized": self.is_quantized,
            "mean_latency_ms": mean_ms,
            "min_latency_ms": min_ms,
            "max_latency_ms": max_ms,
            "std_latency_ms": std_ms,
            "approx_fps": fps,
            "backend": BACKEND_NAME,
            "iterations": iterations,
        }

