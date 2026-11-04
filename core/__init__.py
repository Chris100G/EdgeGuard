"""
EdgeGuard Core Package
Contains camera capture, person detection, risk assessment, event logging, and telemetry modules.
"""

from config import (
    STATE_SAFE,
    STATE_RISK,
    STATE_ERROR,
    STATE_INACTIVE,
    PRIVACY_DISCLAIMER,
)

__all__ = [
    "STATE_SAFE",
    "STATE_RISK",
    "STATE_ERROR",
    "STATE_INACTIVE",
    "PRIVACY_DISCLAIMER",
]

