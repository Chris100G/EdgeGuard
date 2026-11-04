# EdgeGuard

Real-time visual privacy awareness using local person detection. EdgeGuard detects the number of people visible to the camera; it does not determine who they are, whether they are authorized, or whether anyone is reading the screen.

## Status

- Local application: ready to run with webcam or labeled sample-image demo mode.
- GitHub repository: [Chris100G/EdgeGuard](https://github.com/Chris100G/EdgeGuard).
- Hosted demo: not deployed yet. Streamlit Community Cloud requires the account owner to sign in with Google and authorize the GitHub repository.

## Run locally

Use Python 3.10 or newer. From the project directory in PowerShell:

```powershell
$python = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, create and activate a virtual environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then run the same pip and Streamlit commands.

Choose a model and confidence threshold in the sidebar. Select **Sample image demo** to run real local inference on the included example images without a webcam. Otherwise, press **Start monitoring** to open the selected local camera. The camera is not probed or opened before that button is pressed. Press **Stop** to release it.

## Models and inference

The `models/` directory contains the models used by this project. The default is SSD MobileNet V2 quantized; EfficientDet-Lite0 is available as a float model for comparison. Inference uses Google LiteRT on the current system, with no cloud AI service. Model format and quantization labels are reported from the model registry; performance values are measured on the machine running the app.

Run the local model benchmark with:

```powershell
python benchmark.py
```

Benchmark measured on this development machine on 2026-11-03 using Google LiteRT CPU, after 5 warmup runs and 35 timed invocations per model:

| Model | Precision | Size | Mean invoke latency | Approx. model-only throughput |
| --- | --- | ---: | ---: | ---: |
| SSD MobileNet V2 | INT8/UINT8 | 5.93 MB | 49.55 ms | 20.2 FPS |
| EfficientDet-Lite0 | FP32 | 4.35 MB | 76.75 ms | 13.0 FPS |
| SSD MobileNet V1 | UINT8 | 3.99 MB | 28.58 ms | 35.0 FPS |

Throughput is computed as 1000 divided by mean model invocation latency. It excludes camera capture, image preprocessing, and UI rendering; results vary by device.

## Tests

```powershell
python -m pytest -q
```

Tests cover model loading and sample-image inference, confidence filtering, risk smoothing and alert suppression, error states, and event history/export.

## Events and privacy

Event records contain timestamps, counts, confidence, status, and a description. They are stored in `data/events.csv` on the machine running the application and can be exported or cleared in the dashboard. Camera frames are held in memory for inference and are not written to disk by EdgeGuard.

When run locally, frames are processed by the local Python process. A hosted Streamlit deployment runs on its hosting server and cannot access a visitor's laptop webcam through OpenCV; use the clearly labeled sample-image demo there. Do not treat hosted operation as local processing on the visitor's device.

## Deploy to Streamlit Community Cloud

1. Open [Deploy EdgeGuard](https://share.streamlit.io/deploy?repository=Chris100G/EdgeGuard&branch=main&mainModule=app.py).
2. Sign in with the owner's Google account and authorize Streamlit Community Cloud to access the GitHub repository.
3. Wait for the build to complete, then test in sample-image demo mode. Hosted deployments cannot access a visitor's laptop webcam.

**Hosted app:** Not deployed. Complete the owner sign-in above to create the live app.

## Limitations

- Multiple visible people indicate a potential privacy exposure, not proven shoulder surfing or malicious intent.
- Detector misses and false detections are possible; this is an awareness aid, not a security guarantee.
- This Streamlit project is not a native Android application.
