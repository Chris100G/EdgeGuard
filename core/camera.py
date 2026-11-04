"""
EdgeGuard - Camera Module
Manages video capture across physical webcams and calibrated offline mock feeds.
Ensures graceful error handling, resource cleanup, and mock labeling.
"""

import time
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

from config import SAMPLES_DIR

logger = logging.getLogger(__name__)


# Calibrated Mock Scenarios for offline grading and demonstration
MOCK_SCENARIOS = {
    "single_person": {
        "label": "Demo: 1 Person (Normal / Safe State)",
        "file": SAMPLES_DIR / "single_person.jpg",
        "expected_persons": 1,
        "description": "Single primary user working in front of workstation.",
    },
    "shoulder_surfing": {
        "label": "Demo: 2 People (Shoulder Surfing Risk)",
        "file": SAMPLES_DIR / "shoulder_surfing.jpg",
        "expected_persons": 2,
        "description": "Primary user with an unauthorized observer standing over shoulder.",
    },
    "crowded_group": {
        "label": "Demo: 3 People (Crowded / High Risk)",
        "file": SAMPLES_DIR / "crowded_group.jpg",
        "expected_persons": 3,
        "description": "Multiple bystanders clustered around the workstation perimeter.",
    },
    "empty_desk": {
        "label": "Demo: 0 People (Empty Desk / Idle)",
        "file": SAMPLES_DIR / "empty_desk.jpg",
        "expected_persons": 0,
        "description": "Unattended computer display with no human presence.",
    },
}


def list_available_cameras(max_tested: int = 3) -> List[int]:
    """
    Probe system video capture devices to detect available hardware cameras.
    Returns list of working camera integer indices (e.g. [0]).
    """
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if cv2.os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


class VideoStreamManager:
    """
    Manages live camera streams and simulated mock video feeds.
    Provides deterministic fail-safe recovery on camera disconnects.
    """

    def __init__(
        self,
        camera_index: int = 0,
        use_mock: bool = False,
        mock_scenario: str = "single_person",
    ):
        self.camera_index = camera_index
        self.use_mock = use_mock
        self.mock_scenario = mock_scenario

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.cached_mock_frame: Optional[np.ndarray] = None
        self._frame_count = 0

    def start(self) -> Tuple[bool, Optional[str]]:
        """
        Acquire video device or load mock asset.
        Returns (success: bool, error_message: Optional[str]).
        """
        self.stop()
        self._frame_count = 0

        if self.use_mock:
            scenario_info = MOCK_SCENARIOS.get(self.mock_scenario)
            if not scenario_info:
                return False, f"Unknown mock scenario: {self.mock_scenario}"

            file_path = scenario_info["file"]
            if not file_path.exists():
                return False, f"Mock asset file not found: {file_path}"

            img = cv2.imread(str(file_path))
            if img is None:
                return False, f"Failed to decode mock image: {file_path}"

            self.cached_mock_frame = img
            self.is_running = True
            logger.info(f"Initialized Mock Video Stream: {scenario_info['label']}")
            return True, None

        # Hardware Camera Mode
        try:
            # On Windows, cv2.CAP_DSHOW ensures fast hardware initialization
            backend = cv2.CAP_DSHOW if cv2.os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_index, backend)

            if not self.cap.isOpened():
                # Retry with default backend
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                return False, f"Unable to open Camera (Index {self.camera_index}). Check camera privacy permissions or ensure no other app is using it."

            # Set preferred resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.stop()
                return False, f"Camera device {self.camera_index} opened but returned an empty frame."

            self.is_running = True
            logger.info(f"Initialized Hardware Webcam: index {self.camera_index}")
            return True, None

        except Exception as e:
            self.stop()
            return False, f"Webcam acquisition error: {str(e)}"

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        Capture the next frame from the active stream.
        Returns (success: bool, frame_bgr: Optional[np.ndarray], error_message: Optional[str]).
        """
        if not self.is_running:
            return False, None, "Camera stream is not active."

        self._frame_count += 1

        if self.use_mock:
            if self.cached_mock_frame is None:
                return False, None, "Mock frame buffer is empty."

            # Create a live simulated copy with subtle simulated clock / telemetry watermark
            frame = self.cached_mock_frame.copy()
            h, w, _ = frame.shape

            # Add clear, prominent "MOCK DEMO MODE" banner as required by Section 9
            banner_text = f"[DEMO MODE - MOCK FEED] {self.mock_scenario.upper()}"
            cv2.rectangle(frame, (10, 10), (w - 10, 36), (15, 23, 42), -1)
            cv2.rectangle(frame, (10, 10), (w - 10, 36), (6, 182, 212), 1)
            cv2.putText(
                frame,
                banner_text,
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (6, 182, 212),
                1,
                cv2.LINE_AA,
            )

            # Timestamp simulation
            ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                ts_text,
                (w - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Subtle frame tick to simulate camera sensor jitter
            time.sleep(0.015)
            return True, frame, None

        # Hardware Camera Read
        if self.cap is None or not self.cap.isOpened():
            return False, None, "Webcam handle lost or disconnected."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, "Failed to read frame from webcam."

        return True, frame, None

    def stop(self) -> None:
        """Release camera resource and free internal buffers."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.cached_mock_frame = None

    def set_mock_scenario(self, scenario_key: str) -> bool:
        """Dynamically switch mock scenario while running."""
        if scenario_key not in MOCK_SCENARIOS:
            return False
        self.mock_scenario = scenario_key
        if self.use_mock:
            file_path = MOCK_SCENARIOS[scenario_key]["file"]
            img = cv2.imread(str(file_path))
            if img is not None:
                self.cached_mock_frame = img
                return True
        return False

