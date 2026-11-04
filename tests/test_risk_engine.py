"""
EdgeGuard - Unit Tests: Privacy Risk Engine
Verifies state machine, temporal smoothing, alert suppression, and fail-safe error handling.
"""

import pytest
import time
from config import (
    STATE_SAFE,
    STATE_RISK,
    STATE_ERROR,
    STATE_INACTIVE,
)
from core.risk_engine import PrivacyRiskEngine


def test_zero_people_safe_state():
    """Verify that zero detected individuals evaluates to SAFE / Idle."""
    engine = PrivacyRiskEngine(confirmation_frames=1)
    assessment = engine.evaluate(person_count=0, max_confidence=0.0)
    assert assessment.state == STATE_SAFE
    assert assessment.person_count == 0
    assert not assessment.should_alert
    assert "IDLE" in assessment.reason or "SECURE" in assessment.reason


def test_one_person_normal_monitoring_state():
    """Verify that exactly 1 person evaluates to SAFE / Normal."""
    engine = PrivacyRiskEngine(confirmation_frames=1)
    assessment = engine.evaluate(person_count=1, max_confidence=0.88)
    assert assessment.state == STATE_SAFE
    assert assessment.person_count == 1
    assert not assessment.should_alert
    assert "SECURE" in assessment.reason or "Exactly 1" in assessment.reason


def test_two_people_triggers_privacy_risk():
    """Verify that 2 people evaluate to POTENTIAL_PRIVACY_RISK and triggers alert."""
    engine = PrivacyRiskEngine(confirmation_frames=1)
    assessment = engine.evaluate(person_count=2, max_confidence=0.92)
    assert assessment.state == STATE_RISK
    assert assessment.person_count == 2
    assert assessment.should_alert is True
    assert "POTENTIAL VISUAL EAVESDROPPING" in assessment.reason


def test_three_or_more_people_triggers_privacy_risk():
    """Verify that 3+ people also triggers POTENTIAL_PRIVACY_RISK."""
    engine = PrivacyRiskEngine(confirmation_frames=1)
    assessment = engine.evaluate(person_count=3, max_confidence=0.95)
    assert assessment.state == STATE_RISK
    assert assessment.person_count == 3
    assert assessment.should_alert is True


def test_temporal_smoothing_prevents_flicker():
    """
    Verify that an isolated single-frame detection glitch does NOT flip state
    when confirmation_frames > 1.
    """
    engine = PrivacyRiskEngine(confirmation_frames=2)
    # Establish confirmed SAFE state
    engine.evaluate(person_count=1, max_confidence=0.90)
    a1 = engine.evaluate(person_count=1, max_confidence=0.90)
    assert a1.state == STATE_SAFE

    # Glitch frame (sudden 2 people for 1 frame only)
    a_glitch = engine.evaluate(person_count=2, max_confidence=0.55)
    # Should NOT flip to RISK yet
    assert a_glitch.state == STATE_SAFE
    assert a_glitch.should_alert is False

    # Drops back to 1 person
    a_back = engine.evaluate(person_count=1, max_confidence=0.89)
    assert a_back.state == STATE_SAFE
    assert a_back.should_alert is False


def test_temporal_smoothing_confirms_on_consecutive_frames():
    """Verify state transitions once consecutive frames agree."""
    engine = PrivacyRiskEngine(confirmation_frames=2)
    # Frame 1 of risk
    a1 = engine.evaluate(person_count=2, max_confidence=0.90)
    # Frame 2 of risk (confirmed)
    a2 = engine.evaluate(person_count=2, max_confidence=0.91)
    assert a2.state == STATE_RISK
    assert a2.should_alert is True


def test_alert_suppression_cooldown():
    """Verify repeated alerts are suppressed during ongoing continuous violation."""
    engine = PrivacyRiskEngine(confirmation_frames=1, alert_cooldown_sec=2.0)
    # First violation -> Alerts
    a1 = engine.evaluate(person_count=2, max_confidence=0.90)
    assert a1.should_alert is True

    # Immediate next frame -> Violation continues, but alert suppressed
    a2 = engine.evaluate(person_count=2, max_confidence=0.90)
    assert a2.state == STATE_RISK
    assert a2.should_alert is False


def test_camera_error_fail_safe():
    """
    Verify that camera failure triggers STATE_ERROR and NEVER falsely displays SAFE.
    """
    engine = PrivacyRiskEngine()
    assessment = engine.evaluate(
        person_count=1,
        max_confidence=0.90,
        camera_error="Webcam device disconnected",
    )
    assert assessment.state == STATE_ERROR
    assert assessment.state != STATE_SAFE
    assert "Detection offline" in assessment.reason
    assert assessment.should_alert is False


def test_model_error_fail_safe():
    """Verify model inference failure triggers STATE_ERROR."""
    engine = PrivacyRiskEngine()
    assessment = engine.evaluate(
        person_count=0,
        max_confidence=0.0,
        model_error="Interpreter out of memory",
    )
    assert assessment.state == STATE_ERROR
    assert assessment.state != STATE_SAFE

