"""
EdgeGuard - Unit Tests: Person Detector Module
Tests genuine model loading, inference, confidence thresholds, and bounding box validation.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from config import SAMPLES_DIR
from core.detector import PersonDetector, BoundingBox, DetectionResult


@pytest.fixture(scope="module")
def detector_quant():
    """Shared INT8 quantized detector instance."""
    return PersonDetector(model_key="ssd_mobilenet_v2_quant")


@pytest.fixture(scope="module")
def detector_fp32():
    """Shared FP32 detector instance."""
    return PersonDetector(model_key="efficientdet_lite0")


def test_detector_quantized_properties(detector_quant):
    """Verify quantized model attributes and input tensor specs."""
    assert detector_quant.is_quantized is True
    assert detector_quant.input_shape.tolist() == [1, 300, 300, 3]
    assert detector_quant.input_dtype == np.uint8
    assert detector_quant.model_size_mb > 0.0


def test_detector_fp32_properties(detector_fp32):
    """Verify FP32 model attributes and input tensor specs."""
    assert detector_fp32.is_quantized is False
    assert detector_fp32.input_shape.tolist() == [1, 320, 320, 3]
    assert detector_fp32.model_size_mb > 0.0


def test_empty_frame_inference(detector_quant):
    """Verify inference on synthetic black frame yields zero persons."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector_quant.detect(black_frame, confidence_threshold=0.5)
    assert isinstance(result, DetectionResult)
    assert result.person_count == 0
    assert len(result.detections) == 0
    assert result.inference_latency_ms > 0.0


def test_single_person_detection(detector_quant):
    """Verify detection on sample image with 1 person."""
    img_path = SAMPLES_DIR / "single_person.jpg"
    assert img_path.exists(), "single_person.jpg test sample missing"
    frame = cv2.imread(str(img_path))
    assert frame is not None

    result = detector_quant.detect(frame, confidence_threshold=0.45)
    assert result.person_count == 1
    det = result.detections[0]
    assert det.label == "person"
    assert det.confidence >= 0.70
    assert 0.0 <= det.box.ymin < det.box.ymax <= 1.0
    assert 0.0 <= det.box.xmin < det.box.xmax <= 1.0


def test_shoulder_surfing_multiple_people(detector_quant):
    """Verify detection of >= 2 people on the shoulder surfing sample image."""
    img_path = SAMPLES_DIR / "shoulder_surfing.jpg"
    assert img_path.exists(), "shoulder_surfing.jpg test sample missing"
    frame = cv2.imread(str(img_path))
    assert frame is not None

    result = detector_quant.detect(frame, confidence_threshold=0.40)
    assert result.person_count >= 2
    for det in result.detections:
        assert det.label == "person"
        assert det.confidence >= 0.40


def test_confidence_threshold_filtering(detector_quant):
    """Verify raising the threshold strictly filters lower-confidence detections."""
    img_path = SAMPLES_DIR / "single_person.jpg"
    frame = cv2.imread(str(img_path))

    # Low threshold
    res_low = detector_quant.detect(frame, confidence_threshold=0.20)
    # High threshold (near 1.0)
    res_high = detector_quant.detect(frame, confidence_threshold=0.999)

    assert len(res_low.detections) >= len(res_high.detections)


def test_bounding_box_pixel_conversion():
    """Verify normalized bounding box scales correctly to frame pixels."""
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9)
    left, top, right, bottom = box.to_pixel_coords(frame_width=640, frame_height=480)
    assert left == int(0.2 * 640)
    assert top == int(0.1 * 480)
    assert right == int(0.9 * 640)
    assert bottom == int(0.8 * 480)


def test_drawing_annotations(detector_quant):
    """Verify annotation function returns an annotated frame of matching dimensions."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [BoundingBox(ymin=0.1, xmin=0.1, ymax=0.6, xmax=0.6)]
    from core.detector import Detection
    dets = [Detection(box=boxes[0], confidence=0.85, class_id=0, label="person")]
    annotated = detector_quant.draw_detections(frame, dets, is_risk=False)
    assert annotated.shape == frame.shape
    # Check that pixel values were drawn (not all zeros)
    assert np.any(annotated > 0)

