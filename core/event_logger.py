"""
EdgeGuard - Security Event Logger Module
Maintains in-memory security telemetry and persistent CSV audit logging.
Strictly logs security metadata only; NEVER stores raw video or facial frames.
"""

import os
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import pandas as pd

from config import DEFAULT_EVENTS_CSV


@dataclass
class SecurityEvent:
    """Security event record representing a visual privacy change or risk trigger."""
    timestamp: str           # ISO 8601 or formatted local time
    event_category: str      # PRIVACY_WARNING, STATUS_CHANGE, SYSTEM_ERROR, MONITORING_START, MONITORING_STOP
    person_count: int        # Number of detected people
    confidence_max: float    # Max confidence score
    risk_status: str         # SAFE, POTENTIAL_PRIVACY_RISK, SYSTEM_ERROR
    description: str         # Human-readable incident summary


class SecurityEventLogger:
    """
    Thread-safe / file-backed security auditor.
    Adheres strictly to the EdgeGuard zero-image privacy standard.
    """

    CSV_HEADERS = [
        "timestamp",
        "event_category",
        "person_count",
        "confidence_max",
        "risk_status",
        "description",
    ]

    def __init__(self, csv_path: Path = DEFAULT_EVENTS_CSV, max_memory_records: int = 200):
        self.csv_path = Path(csv_path)
        self.max_memory_records = max_memory_records
        self.in_memory_events: List[SecurityEvent] = []

        # Ensure directory exists and initialize CSV with header if missing
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_initialized()
        self._load_recent_from_csv()

    def _ensure_csv_initialized(self) -> None:
        """Create empty CSV with correct schema if it does not exist."""
        if not self.csv_path.exists() or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def _load_recent_from_csv(self) -> None:
        """Load the last N records into in-memory cache on startup."""
        if not self.csv_path.exists():
            return

        try:
            df = pd.read_csv(self.csv_path)
            recent_rows = df.tail(self.max_memory_records)
            self.in_memory_events = []
            for _, row in recent_rows.iterrows():
                self.in_memory_events.append(
                    SecurityEvent(
                        timestamp=str(row.get("timestamp", "")),
                        event_category=str(row.get("event_category", "")),
                        person_count=int(row.get("person_count", 0)),
                        confidence_max=float(row.get("confidence_max", 0.0)),
                        risk_status=str(row.get("risk_status", "")),
                        description=str(row.get("description", "")),
                    )
                )
        except Exception:
            self.in_memory_events = []

    def log_event(
        self,
        event_category: str,
        person_count: int,
        confidence_max: float,
        risk_status: str,
        description: str,
        timestamp_str: Optional[str] = None,
    ) -> SecurityEvent:
        """
        Record a security event both to RAM and persistent local storage.
        """
        ts = timestamp_str or time.strftime("%Y-%m-%d %H:%M:%S")

        event = SecurityEvent(
            timestamp=ts,
            event_category=event_category,
            person_count=person_count,
            confidence_max=round(float(confidence_max), 3),
            risk_status=risk_status,
            description=description,
        )

        # Update RAM
        self.in_memory_events.append(event)
        if len(self.in_memory_events) > self.max_memory_records:
            self.in_memory_events.pop(0)

        # Append to CSV
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    event.event_category,
                    event.person_count,
                    event.confidence_max,
                    event.risk_status,
                    event.description,
                ])
        except Exception as e:
            # Avoid crashing live loop if file is temporarily locked
            pass

        return event

    def get_recent_events(self, limit: int = 50) -> List[SecurityEvent]:
        """Return the most recent security events, sorted newest first."""
        return list(reversed(self.in_memory_events[-limit:]))

    def to_dataframe(self) -> pd.DataFrame:
        """Export in-memory events as a Pandas DataFrame for Streamlit rendering."""
        if not self.in_memory_events:
            return pd.DataFrame(columns=self.CSV_HEADERS)
        return pd.DataFrame([asdict(e) for e in reversed(self.in_memory_events)])

    def clear_history(self) -> None:
        """Wipe event history from both RAM and local CSV disk storage."""
        self.in_memory_events.clear()
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)

    def export_csv_bytes(self) -> bytes:
        """Generate binary CSV contents for in-browser download."""
        if self.csv_path.exists():
            with open(self.csv_path, "rb") as f:
                return f.read()
        return ",".join(self.CSV_HEADERS).encode("utf-8") + b"\n"

