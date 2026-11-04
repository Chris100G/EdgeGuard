"""
EdgeGuard - Unit Tests: Security Event Logger Module
Verifies in-memory history, CSV persistence, export formatting, and zero-image privacy compliance.
"""

import pytest
import os
import tempfile
from pathlib import Path

from core.event_logger import SecurityEventLogger, SecurityEvent


@pytest.fixture
def temp_logger(tmp_path):
    """Creates an isolated temporary logger."""
    csv_file = tmp_path / "test_events.csv"
    return SecurityEventLogger(csv_path=csv_file, max_memory_records=10)


def test_event_logging(temp_logger):
    """Verify logging an event updates in-memory list and writes to CSV."""
    event = temp_logger.log_event(
        event_category="PRIVACY_WARNING",
        person_count=2,
        confidence_max=0.945,
        risk_status="POTENTIAL_PRIVACY_RISK",
        description="Shoulder surfing detected",
    )

    assert isinstance(event, SecurityEvent)
    assert event.person_count == 2
    assert event.risk_status == "POTENTIAL_PRIVACY_RISK"
    assert len(temp_logger.in_memory_events) == 1

    # Check file exists and has content
    assert temp_logger.csv_path.exists()
    content = temp_logger.csv_path.read_text(encoding="utf-8")
    assert "PRIVACY_WARNING" in content
    assert "Shoulder surfing detected" in content


def test_in_memory_limit(temp_logger):
    """Verify logger respects maximum in-memory record cap without unbounded RAM growth."""
    for i in range(15):
        temp_logger.log_event(
            event_category="STATUS_CHANGE",
            person_count=1,
            confidence_max=0.8,
            risk_status="SAFE",
            description=f"Event {i}",
        )

    assert len(temp_logger.in_memory_events) == 10
    # Oldest record should have been shifted out
    assert temp_logger.in_memory_events[-1].description == "Event 14"


def test_dataframe_conversion(temp_logger):
    """Verify DataFrame export schema matches expected fields."""
    temp_logger.log_event(
        event_category="PRIVACY_WARNING",
        person_count=2,
        confidence_max=0.91,
        risk_status="POTENTIAL_PRIVACY_RISK",
        description="Test risk",
    )
    df = temp_logger.to_dataframe()
    assert not df.empty
    assert "event_category" in df.columns
    assert "risk_status" in df.columns
    assert df.iloc[0]["risk_status"] == "POTENTIAL_PRIVACY_RISK"


def test_clear_history(temp_logger):
    """Verify clear history resets RAM buffer and truncates disk CSV."""
    temp_logger.log_event(
        event_category="PRIVACY_WARNING",
        person_count=2,
        confidence_max=0.91,
        risk_status="POTENTIAL_PRIVACY_RISK",
        description="To be deleted",
    )
    assert len(temp_logger.in_memory_events) == 1

    temp_logger.clear_history()
    assert len(temp_logger.in_memory_events) == 0

    content = temp_logger.csv_path.read_text(encoding="utf-8")
    # File should only contain the CSV headers
    assert "To be deleted" not in content
    assert "timestamp,event_category,person_count" in content


def test_export_csv_bytes(temp_logger):
    """Verify CSV export returns binary UTF-8 stream with headers."""
    temp_logger.log_event(
        event_category="PRIVACY_WARNING",
        person_count=2,
        confidence_max=0.95,
        risk_status="POTENTIAL_PRIVACY_RISK",
        description="Export test",
    )
    raw_bytes = temp_logger.export_csv_bytes()
    assert isinstance(raw_bytes, bytes)
    text = raw_bytes.decode("utf-8")
    assert "Export test" in text


def test_zero_image_storage_privacy(temp_logger):
    """
    STRICT PRIVACY AUDIT:
    Verify that NO binary image data, base64 data, or raw pixel matrices
    are recorded into security events or the CSV file.
    """
    event = temp_logger.log_event(
        event_category="PRIVACY_WARNING",
        person_count=2,
        confidence_max=0.95,
        risk_status="POTENTIAL_PRIVACY_RISK",
        description="Standard metadata event",
    )

    # Ensure event fields contain only textual/numerical metadata
    for key, val in event.__dict__.items():
        assert not isinstance(val, (bytes, bytearray, memoryview))
        # Ensure no huge base64 strings or binary encodings
        if isinstance(val, str):
            assert len(val) < 500, f"Field {key} contains suspiciously large payload"

    # Ensure disk CSV contains no binary bytes
    with open(temp_logger.csv_path, "rb") as f:
        file_bytes = f.read()
    # JPEG / PNG / WebP magic bytes should never appear in CSV
    assert b"\xff\xd8\xff" not in file_bytes, "JPEG magic header found in event log!"
    assert b"\x89PNG" not in file_bytes, "PNG magic header found in event log!"

