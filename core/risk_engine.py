"""
EdgeGuard - Privacy Risk Engine Module
Evaluates human presence detections to identify visual privacy exposure (shoulder surfing).
Operates with transparent deterministic rules, temporal smoothing, and alert cooldown.
"""

import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict, Any

from config import (
    STATE_SAFE,
    STATE_RISK,
    STATE_ERROR,
    STATE_INACTIVE,
    DEFAULT_CONFIRMATION_FRAMES,
    DEFAULT_ALERT_COOLDOWN_SEC,
)


@dataclass
class RiskAssessment:
    """Represents the real-time privacy status evaluated by the risk engine."""
    state: str                     # SAFE, POTENTIAL_PRIVACY_RISK, SYSTEM_ERROR, MONITORING_INACTIVE
    person_count: int              # Detected people count in frame
    reason: str                    # Transparent technical rationale
    should_alert: bool             # Whether an audio/visual alert trigger should fire
    max_confidence: float          # Peak detection confidence among present individuals
    timestamp: float               # Epoch timestamp of assessment
    is_state_changed: bool         # True if state changed compared to previous confirmed state


class PrivacyRiskEngine:
    """
    Deterministic Cybersecurity Privacy Assessment Engine.
    
    Principles:
    1. Transparency: Explicit rules based solely on person presence.
    2. Zero Biometrics: No facial recognition, identity tracking, or biometric profiles.
    3. Temporal Stability: Prevents flickering via consecutive frame confirmation.
    4. Alert Throttling: Suppresses repeated noisy alerts during continuous observation.
    5. Fail-Safe: System errors or camera dropouts never default to 'SAFE'.
    """

    def __init__(
        self,
        confirmation_frames: int = DEFAULT_CONFIRMATION_FRAMES,
        alert_cooldown_sec: float = DEFAULT_ALERT_COOLDOWN_SEC,
    ):
        self.confirmation_frames = max(1, confirmation_frames)
        self.alert_cooldown_sec = alert_cooldown_sec

        # State tracking
        self.current_state = STATE_INACTIVE
        self.confirmed_person_count = 0
        self.last_alert_time: float = 0.0

        # Temporal smoothing history
        self._recent_states = deque(maxlen=self.confirmation_frames)
        self._recent_counts = deque(maxlen=self.confirmation_frames)

    def reset(self) -> None:
        """Reset internal buffers on monitoring restart or camera reconfiguration."""
        self.current_state = STATE_INACTIVE
        self.confirmed_person_count = 0
        self.last_alert_time = 0.0
        self._recent_states.clear()
        self._recent_counts.clear()

    def evaluate(
        self,
        person_count: int,
        max_confidence: float = 0.0,
        camera_error: Optional[str] = None,
        model_error: Optional[str] = None,
    ) -> RiskAssessment:
        """
        Evaluate real-time privacy risk from detector output and camera health.
        
        Rules:
        - If camera_error or model_error: STATE_ERROR
        - If person_count >= 2: STATE_RISK
        - If person_count == 1: STATE_SAFE (Normal monitoring state)
        - If person_count == 0: STATE_SAFE (Workstation idle / no presence)
        """
        now = time.time()

        # Rule 1: Fail-Safe System Error (Never display SAFE on failure)
        if camera_error or model_error:
            err_msg = camera_error or model_error
            self.current_state = STATE_ERROR
            self._recent_states.clear()
            self._recent_counts.clear()
            return RiskAssessment(
                state=STATE_ERROR,
                person_count=0,
                reason=f"Detection offline: {err_msg}. Privacy protection cannot be guaranteed.",
                should_alert=False,
                max_confidence=0.0,
                timestamp=now,
                is_state_changed=True,
            )

        # Rule 2: Instantaneous State Determination
        if person_count >= 2:
            instant_state = STATE_RISK
        else:
            instant_state = STATE_SAFE

        self._recent_states.append(instant_state)
        self._recent_counts.append(person_count)

        # Rule 3: Temporal Smoothing Confirmation
        # State only updates if all samples in confirmation window agree
        is_state_changed = False
        should_alert = False

        if len(self._recent_states) == self.confirmation_frames:
            # Check if all recent states match
            all_agree = all(s == self._recent_states[0] for s in self._recent_states)
            if all_agree:
                new_state = self._recent_states[0]
                if new_state != self.current_state:
                    is_state_changed = True
                    self.current_state = new_state
                self.confirmed_person_count = person_count
        elif self.current_state == STATE_INACTIVE:
            # Initializing: default to safe baseline until window fills
            self.current_state = STATE_SAFE
            self.confirmed_person_count = person_count

        # Rule 4: Alert Suppression & Trigger Logic
        if self.current_state == STATE_RISK:
            # Trigger alert if new transition into RISK, or cooldown passed in ongoing risk
            if is_state_changed:
                should_alert = True
                self.last_alert_time = now
            elif (now - self.last_alert_time) >= self.alert_cooldown_sec:
                should_alert = True
                self.last_alert_time = now

        # Rule 5: Transparent Technical Explanations
        if self.current_state == STATE_RISK:
            reason = (
                f"POTENTIAL VISUAL EAVESDROPPING: {self.confirmed_person_count} individuals detected in camera view. "
                f"Screen contents may be visible to surrounding bystanders (shoulder surfing risk)."
            )
        elif self.current_state == STATE_SAFE:
            if self.confirmed_person_count == 1:
                reason = "SECURE: Exactly 1 primary operator detected. No secondary observers visible in perimeter."
            else:
                reason = "IDLE / SECURE: No individuals detected in proximity to display."
        else:
            reason = "Awaiting camera activation."

        return RiskAssessment(
            state=self.current_state,
            person_count=self.confirmed_person_count,
            reason=reason,
            should_alert=should_alert,
            max_confidence=max_confidence,
            timestamp=now,
            is_state_changed=is_state_changed,
        )
