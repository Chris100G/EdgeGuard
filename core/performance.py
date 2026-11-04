"""
EdgeGuard - Performance & Telemetry Module
Tracks real-time inference latency, frame rates, CPU utilization, and system metrics.
Explicitly distinguishes measured values from statistical approximations.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetrySnapshot:
    """Consolidated performance telemetry report."""
    inference_latency_ms: float      # MEASURED: Direct TFLite invoke time
    total_latency_ms: float          # MEASURED: Preprocess + Inference + Risk + Postprocess
    approx_fps: float                # MEASURED: Rolling frame rate over last N seconds
    frames_processed: int            # MEASURED: Total frame counter
    frames_skipped: int              # MEASURED: Frames throttled for CPU power saving
    cpu_percent_approx: float        # ESTIMATED: Process or system CPU load
    is_measured_fps: bool = True     # Transparency flag


class PerformanceTracker:
    """
    High-precision sliding window telemetry monitor.
    Uses time.perf_counter for sub-millisecond hardware profiling.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_timestamps = deque(maxlen=window_size)
        self.inference_latencies = deque(maxlen=window_size)
        self.total_latencies = deque(maxlen=window_size)

        self.frames_processed = 0
        self.frames_skipped = 0
        self.start_time = time.time()

    def reset(self) -> None:
        """Reset counters and latency histories."""
        self.frame_timestamps.clear()
        self.inference_latencies.clear()
        self.total_latencies.clear()
        self.frames_processed = 0
        self.frames_skipped = 0
        self.start_time = time.time()

    def record_frame(self, inference_ms: float, total_ms: float) -> None:
        """Record timing metrics for a completed frame cycle."""
        now = time.perf_counter()
        self.frame_timestamps.append(now)
        self.inference_latencies.append(inference_ms)
        self.total_latencies.append(total_ms)
        self.frames_processed += 1

    def record_skipped_frame(self) -> None:
        """Record a frame skipped intentionally due to frame_interval configuration."""
        self.frames_skipped += 1

    def get_snapshot(self) -> TelemetrySnapshot:
        """Compute rolling telemetry metrics."""
        # Calculate Rolling FPS
        approx_fps = 0.0
        if len(self.frame_timestamps) >= 2:
            elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if elapsed > 0:
                approx_fps = (len(self.frame_timestamps) - 1) / elapsed

        # Latencies
        avg_inf_ms = float(sum(self.inference_latencies) / len(self.inference_latencies)) if self.inference_latencies else 0.0
        avg_tot_ms = float(sum(self.total_latencies) / len(self.total_latencies)) if self.total_latencies else 0.0

        # Estimated CPU utilization (approximated via duty cycle of processing vs wall clock)
        cpu_load_est = 0.0
        if approx_fps > 0 and avg_tot_ms > 0:
            duty_cycle = (avg_tot_ms / 1000.0) * approx_fps
            cpu_load_est = min(100.0, duty_cycle * 100.0)

        return TelemetrySnapshot(
            inference_latency_ms=round(avg_inf_ms, 2),
            total_latency_ms=round(avg_tot_ms, 2),
            approx_fps=round(approx_fps, 1),
            frames_processed=self.frames_processed,
            frames_skipped=self.frames_skipped,
            cpu_percent_approx=round(cpu_load_est, 1),
            is_measured_fps=True,
        )

