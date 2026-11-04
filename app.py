from __future__ import annotations

import base64
import html
import io
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_INTERVAL,
    MODEL_REGISTRY,
    PRIVACY_DISCLAIMER,
    STATE_ERROR,
    STATE_INACTIVE,
    STATE_RISK,
    STATE_SAFE,
)
from core.camera import MOCK_SCENARIOS, VideoStreamManager
from core.event_logger import SecurityEventLogger
from core.performance import PerformanceTracker
from core.risk_engine import PrivacyRiskEngine


st.set_page_config(page_title="EdgeGuard | Visual Privacy Monitor", page_icon="E", layout="wide")


def load_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {
        "monitoring": lambda: False,
        "manager": lambda: None,
        "detector": lambda: None,
        "risk_engine": PrivacyRiskEngine,
        "event_logger": SecurityEventLogger,
        "performance": PerformanceTracker,
        "assessment": lambda: None,
        "latest_detection": lambda: None,
        "latest_frame": lambda: None,
        "latest_alert": lambda: None,
        "frame_number": lambda: 0,
        "last_error": lambda: None,
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def play_alert_tone() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(22050)
        samples = bytearray()
        for index in range(3308):
            amplitude = int(6000 * (1 - index / 3308))
            sample = int(amplitude * (1 if (index * 880 // 22050) % 2 == 0 else -1))
            samples.extend(sample.to_bytes(2, "little", signed=True))
        sound.writeframes(samples)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    components.html(
        f'<audio autoplay><source src="data:audio/wav;base64,{encoded}" type="audio/wav"></audio>',
        height=0,
    )


def log_assessment_change(assessment) -> None:
    previous_state = st.session_state.get("last_logged_state")
    if assessment.state == previous_state:
        return
    st.session_state["last_logged_state"] = assessment.state
    if assessment.state == STATE_RISK:
        category = "PRIVACY_WARNING"
    elif assessment.state == STATE_ERROR:
        category = "SYSTEM_ERROR"
    else:
        category = "STATUS_CHANGE"
    st.session_state.event_logger.log_event(
        event_category=category,
        person_count=assessment.person_count,
        confidence_max=assessment.max_confidence,
        risk_status=assessment.state,
        description=assessment.reason,
    )


def status_panel(assessment) -> None:
    if assessment is None:
        state = STATE_INACTIVE
        count = 0
        reason = "Start monitoring to assess people in the camera view."
    else:
        state = assessment.state
        count = assessment.person_count
        reason = assessment.reason

    state_class = {
        STATE_SAFE: "safe",
        STATE_RISK: "risk",
        STATE_ERROR: "error",
        STATE_INACTIVE: "inactive",
    }.get(state, "inactive")
    labels = {
        STATE_SAFE: "NORMAL",
        STATE_RISK: "POTENTIAL PRIVACY RISK",
        STATE_ERROR: "DETECTION ERROR",
        STATE_INACTIVE: "MONITORING OFF",
    }
    st.markdown(
        f'<div class="status-banner status-banner-{state_class}">'
        f'<div class="status-headline {state_class}">{labels.get(state, "UNKNOWN")}</div>'
        f'<div class="status-description">{html.escape(reason)}</div></div>',
        unsafe_allow_html=True,
    )
    metric, confidence = st.columns(2)
    metric.metric("People detected", count)
    peak_confidence = assessment.max_confidence if assessment else 0.0
    confidence.metric("Peak confidence", f"{peak_confidence:.0%}" if peak_confidence else "--")
    st.caption("Presence only. EdgeGuard does not identify people or determine intent.")


def monitor_tick() -> None:
    manager = st.session_state.manager
    detector = st.session_state.detector
    if not manager or not detector:
        return

    ok, frame, camera_error = manager.read_frame()
    if not ok or frame is None:
        assessment = st.session_state.risk_engine.evaluate(0, camera_error=camera_error)
        st.session_state.assessment = assessment
        st.session_state.last_error = camera_error
        log_assessment_change(assessment)
        manager.stop()
        st.session_state.monitoring = False
        return

    st.session_state.frame_number += 1
    interval = st.session_state.processing_interval
    if st.session_state.frame_number % interval == 0 or st.session_state.latest_detection is None:
        started = time.perf_counter()
        try:
            result = detector.detect(frame, confidence_threshold=st.session_state.confidence_threshold)
            assessment = st.session_state.risk_engine.evaluate(
                result.person_count,
                max((item.confidence for item in result.detections), default=0.0),
            )
            st.session_state.latest_detection = result
            st.session_state.assessment = assessment
            st.session_state.performance.record_frame(
                result.inference_latency_ms,
                (time.perf_counter() - started) * 1000,
            )
            log_assessment_change(assessment)
            if assessment.should_alert:
                st.session_state.latest_alert = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.toast("Potential privacy risk: multiple people detected.", icon="⚠️")
                if st.session_state.alert_sound:
                    play_alert_tone()
        except Exception as error:
            assessment = st.session_state.risk_engine.evaluate(0, model_error=str(error))
            st.session_state.assessment = assessment
            st.session_state.last_error = str(error)
            log_assessment_change(assessment)
            manager.stop()
            st.session_state.monitoring = False
            return
    else:
        st.session_state.performance.record_skipped_frame()

    detections = st.session_state.latest_detection.detections
    is_risk = st.session_state.assessment.state == STATE_RISK
    annotated = detector.draw_detections(frame, detections, is_risk=is_risk)
    st.session_state.latest_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


@st.fragment(run_every=0.1)
def live_monitor(model_key, mock_mode, scenario_key, camera_index, confidence, frame_interval) -> None:
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.subheader("Monitoring")
        start_col, stop_col, state_col = st.columns([1, 1, 2])
        with start_col:
            start_clicked = st.button(
                "Start monitoring", type="primary", disabled=st.session_state.monitoring,
            )
        with stop_col:
            stop_clicked = st.button("Stop", disabled=not st.session_state.monitoring)
        with state_col:
            st.caption("Active" if st.session_state.monitoring else "Stopped")

        if start_clicked:
            try:
                from core.detector import PersonDetector

                detector = PersonDetector(model_key=model_key, confidence_threshold=confidence)
                manager = VideoStreamManager(
                    camera_index=camera_index,
                    use_mock=mock_mode,
                    mock_scenario=scenario_key,
                )
                started, error = manager.start()
                if not started:
                    st.session_state.last_error = error
                    assessment = st.session_state.risk_engine.evaluate(0, camera_error=error)
                    st.session_state.assessment = assessment
                    log_assessment_change(assessment)
                else:
                    st.session_state.detector = detector
                    st.session_state.manager = manager
                    st.session_state.monitoring = True
                    st.session_state.last_error = None
                    st.session_state.latest_detection = None
                    st.session_state.latest_frame = None
                    st.session_state.frame_number = 0
                    st.session_state.risk_engine.reset()
                    st.session_state.performance.reset()
                    st.session_state.last_logged_state = STATE_INACTIVE
                    st.session_state.event_logger.log_event(
                        "MONITORING_START", 0, 0.0, STATE_INACTIVE,
                        "Monitoring started in " + ("sample image demo" if mock_mode else "local camera") + " mode.",
                    )
            except Exception as error:
                st.session_state.last_error = str(error)
                assessment = st.session_state.risk_engine.evaluate(0, model_error=str(error))
                st.session_state.assessment = assessment
                log_assessment_change(assessment)
            st.rerun()

        if stop_clicked:
            if st.session_state.manager:
                st.session_state.manager.stop()
            st.session_state.monitoring = False
            st.session_state.risk_engine.reset()
            st.session_state.assessment = None
            st.session_state.latest_frame = None
            st.session_state.last_error = None
            st.session_state.event_logger.log_event(
                "MONITORING_STOP", 0, 0.0, STATE_INACTIVE, "Monitoring stopped by the user.",
            )
            st.rerun()

        if st.session_state.monitoring:
            monitor_tick()
        if st.session_state.latest_frame is not None and st.session_state.monitoring:
            st.image(st.session_state.latest_frame, width="stretch")
            mode = "MOCK SAMPLE FEED" if mock_mode else "LOCAL CAMERA FEED"
            st.caption(f"{mode} | Frames are processed in memory and not recorded.")
        elif st.session_state.last_error:
            st.error(st.session_state.last_error)
        else:
            st.markdown(
                '<div class="camera-placeholder"><h3>Camera is standing by</h3>'
                '<p>Select a camera or sample scenario, then start monitoring to begin local inference.</p></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("Security status")
        status_panel(st.session_state.assessment)
        st.markdown("#### Latest alert")
        if st.session_state.latest_alert:
            st.warning(f"{st.session_state.latest_alert} · Multiple people are present in view.")
        else:
            st.caption("No active privacy warning.")

    st.divider()
    st.subheader("Performance and model")
    snapshot = st.session_state.performance.get_snapshot()
    detection = st.session_state.latest_detection
    model_meta = MODEL_REGISTRY[model_key]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Inference", f"{snapshot.inference_latency_ms:.1f} ms", help="Measured model invocation average")
    metric_cols[1].metric("Throughput", f"{snapshot.approx_fps:.1f} FPS", help="Measured rolling inference rate")
    metric_cols[2].metric("Frames", snapshot.frames_processed)
    metric_cols[3].metric("Model", "Loaded" if st.session_state.detector else "Not loaded")
    size_mb = detection.model_size_mb if detection else (
        Path(model_meta["path"]).stat().st_size / (1024 * 1024) if Path(model_meta["path"]).exists() else 0.0
    )
    metric_cols[4].metric("Model size", f"{size_mb:.1f} MB" if size_mb else "Unavailable")
    if detection:
        st.caption(
            f"{detection.model_name} · {'Quantized' if detection.is_quantized else 'Float'} · "
            "FPS and latency are measured on this runtime."
        )
    else:
        st.caption(f"{model_meta['name']} · {model_meta['precision']} · Metrics appear after inference starts.")

    st.divider()
    events_col, privacy_col = st.columns([1.5, 1], gap="large")
    with events_col:
        st.subheader("Recent security events")
        event_logger = st.session_state.event_logger
        st.download_button(
            "Export CSV", data=event_logger.export_csv_bytes(), file_name="edgeguard_events.csv",
            mime="text/csv", disabled=not event_logger.in_memory_events,
        )
        events = event_logger.to_dataframe()
        if events.empty:
            st.caption("No events recorded in this session.")
        else:
            st.dataframe(events.head(10), width="stretch", hide_index=True)
    with privacy_col:
        st.subheader("Privacy and scope")
        st.info(PRIVACY_DISCLAIMER)
        st.caption(
            "Only person presence is assessed. No facial recognition, identity tracking, or screen reading. "
            "A hosted demo can access only its host runtime, not your laptop webcam."
        )


load_styles()
initialize_state()

logo_path = Path(__file__).parent / "assets" / "logo.png"
logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii") if logo_path.exists() else ""
st.markdown(
    '<div class="edgeguard-header"><div class="edgeguard-brand">'
    f'<img class="edgeguard-logo" src="data:image/png;base64,{logo_data}" alt="EdgeGuard shield mark">'
    '<div class="edgeguard-title-group"><h1>EDGEGUARD</h1>'
    '<p>REAL-TIME VISUAL PRIVACY MONITORING</p></div></div>'
    '<div class="edgeguard-pill-group"><span class="edgeguard-pill edgeguard-pill-active">'
    '<span class="edgeguard-pill-dot cyan"></span>LOCAL INFERENCE</span></div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Monitoring settings")
    mock_mode = st.toggle("Sample image demo", key="mock_mode", disabled=st.session_state.monitoring)
    scenario_key = st.selectbox(
        "Sample scenario",
        options=list(MOCK_SCENARIOS),
        format_func=lambda key: MOCK_SCENARIOS[key]["label"],
        disabled=st.session_state.monitoring or not mock_mode,
    )
    camera_index = st.number_input(
        "Camera index", min_value=0, max_value=8, value=0, step=1,
        disabled=st.session_state.monitoring or mock_mode,
        help="The camera is opened only after you press Start monitoring.",
    )
    model_key = st.selectbox(
        "Detection model", options=list(MODEL_REGISTRY),
        format_func=lambda key: MODEL_REGISTRY[key]["name"],
        disabled=st.session_state.monitoring,
    )
    confidence = st.slider(
        "Confidence threshold", min_value=0.20, max_value=0.95,
        value=float(DEFAULT_CONFIDENCE_THRESHOLD), step=0.05,
    )
    frame_interval = st.slider(
        "Process every N frames", min_value=1, max_value=5,
        value=int(DEFAULT_FRAME_INTERVAL),
    )
    st.toggle("Alert sound", value=False, key="alert_sound")
    st.divider()
    st.subheader("Event history")
    if st.button("Clear event history", width="stretch"):
        st.session_state.event_logger.clear_history()
        st.session_state.latest_alert = None
        st.rerun()

st.session_state.confidence_threshold = confidence
st.session_state.processing_interval = frame_interval
live_monitor(model_key, mock_mode, scenario_key, int(camera_index), confidence, frame_interval)
