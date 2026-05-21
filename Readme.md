# Eye Detect System

Eye Detect System is a Python desktop application that uses a webcam to detect whether a user's eyes are open or closed in real time. It combines OpenCV video capture, cvzone face mesh landmarks, Eye Aspect Ratio (EAR) calculation, and a Tkinter control panel to provide a simple eye-closure alert system.

## Features

- Real-time webcam-based face and eye landmark detection.
- Eye Aspect Ratio (EAR) calculation for both eyes.
- Smoothed EAR readings to reduce frame-to-frame noise.
- Calibration mode for adapting the eye-open threshold to the current user.
- Automatic threshold clamping to keep detection behavior within practical bounds.
- Hysteresis-based open/closed state handling to avoid rapid flickering.
- Audible Windows alert when eyes remain closed for multiple frames.
- Tkinter GUI with Start, Stop, Calibrate, and Quit controls.
- OpenCV camera preview with eye landmark points drawn on the frame.

## Implementation Overview

The main application lives in `main.py`.

1. The Tkinter GUI is created by `build_gui()`.
2. Pressing **Start** launches `detection_loop()` in a background thread so the GUI stays responsive.
3. OpenCV reads frames from the default webcam using `cv2.VideoCapture(0)`.
4. Frames are converted to grayscale and enhanced with CLAHE before face mesh detection.
5. `cvzone.FaceMeshModule.FaceMeshDetector` extracts facial landmarks.
6. Six landmarks from each eye are used to calculate EAR.
7. Recent EAR values are averaged with a small smoothing window.
8. Calibration records open-eye EAR values and uses the median to compute a user-specific threshold.
9. If smoothed EAR stays below the threshold for enough frames, the app marks eyes as closed and plays a beep.

## Project Structure

```text
Eye-Detect-System/
+-- main.py
+-- requirements.txt
+-- Readme.md
+-- .gitignore
```

## Requirements

- Python 3.10 or compatible Python 3.x version
- Webcam
- Windows for the built-in `winsound.Beep()` alert

Python packages are listed in `requirements.txt`:

```text
opencv-python
numpy
cvzone
mediapipe
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Usage

1. Run the application.
2. Click **Start** to begin webcam detection.
3. Keep your face visible in the camera preview.
4. Click **Calibrate (keep eyes open)** while looking normally at the camera.
5. After calibration completes, the app uses the personalized threshold for eye-closure detection.
6. Press **Esc** in the OpenCV preview window or click **Stop** to end detection.

## Key Parameters

The detection behavior can be tuned near the top of `main.py`:

- `CALIBRATE_FRAMES`: Number of frames collected during calibration.
- `SMOOTHING_WINDOW`: Number of recent EAR values averaged together.
- `CLOSED_FRAMES`: Number of consecutive closed-eye frames required before alerting.
- `CALIB_FACTOR`: Multiplier used to derive the closed-eye threshold from calibrated open-eye EAR.
- `MIN_THRESHOLD` and `MAX_THRESHOLD`: Lower and upper threshold bounds.
- `EAR_HYSTERESIS`: Buffer around the threshold to reduce flickering.

## Notes

- The app currently uses the default camera index `0`.
- The audible alert uses Windows-only `winsound`, so another sound backend is needed for Linux or macOS.
- Detection quality depends on lighting, camera angle, and whether the user's face is clearly visible.
