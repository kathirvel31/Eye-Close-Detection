# main.py
import cv2
import numpy as np
import threading
import time
import winsound
from collections import deque
from cvzone.FaceMeshModule import FaceMeshDetector
import tkinter as tk
from tkinter import ttk, messagebox

# ----- Parameters (tweak if needed) -----
CALIBRATE_FRAMES = 80
SMOOTHING_WINDOW = 5
CLOSED_FRAMES = 15
CALIB_FACTOR = 0.60
MIN_THRESHOLD = 0.23
print(MIN_THRESHOLD)
MAX_THRESHOLD = 0.35
EAR_HYSTERESIS = 0.02

# ----- Globals used by GUI and thread -----
cap = None
detector = None
running = False
calibrating = False
calibrated = False
calib_values = []
calibrated_open_ear = None
ear_history = deque(maxlen=SMOOTHING_WINDOW)
counter = 0
last_beep = 0.0

# Mutex for safe state changes
state_lock = threading.Lock()

# ----- EAR helper -----
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

# ----- Video / detection loop (runs in thread) -----
def detection_loop(status_label, ear_label, thr_label):
    global cap, detector, running
    global calib_values, calibrating, calibrated, calibrated_open_ear
    global ear_history, counter, last_beep

    # initialize video + detector (do it inside thread)
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while True:
        with state_lock:
            if not running:
                break

        success, img = cap.read()
        if not success:
            status_label.config(text="Camera error")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)
        img_proc = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)


        img_mesh, faces = detector.findFaceMesh(img_proc, draw=False)
        ear_smoothed = 0.0

        if faces:
            face = faces[0]
            leftEye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
            rightEye = [face[i] for i in [362, 385, 387, 263, 373, 380]]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear_raw = (leftEAR + rightEAR) / 2.0

            ear_history.append(ear_raw)
            ear_smoothed = float(np.mean(ear_history))

            # Calibration block
            if calibrating:
                calib_values.append(ear_smoothed)
                status_label.config(text=f"Calibrating... {len(calib_values)}/{CALIBRATE_FRAMES}")
                if len(calib_values) >= CALIBRATE_FRAMES:
                    calibrated_open_ear = float(np.median(calib_values))
                    calibrated = True
                    calibrating = False
                    calib_values.clear()
                    status_label.config(text=f"Calibrated: openEAR={calibrated_open_ear:.3f}")
            # threshold
            if calibrated and calibrated_open_ear is not None:
                threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, calibrated_open_ear * CALIB_FACTOR))
            else:
                threshold = 0.25

            # state detection
            if ear_smoothed < threshold - EAR_HYSTERESIS:
                counter += 1
            elif ear_smoothed > threshold + EAR_HYSTERESIS:
                counter = 0

            if counter >= CLOSED_FRAMES:
                # eyes closed
                status_label.config(text="EYES CLOSED!")
                now = time.time()
                if now - last_beep > 0.5:
                    winsound.Beep(1800, 700)  # beep
                    last_beep = now
            else:
                status_label.config(text="Eyes open")
            ear_label.config(text=f"EAR: {ear_smoothed:.3f}")
            if calibrated:
                thr_label.config(text=f"THR: {threshold:.3f}")
            else:
                thr_label.config(text="Not calibrated")
            # draw points on image for debug
            for p in leftEye + rightEye:
                cv2.circle(img, tuple(p), 2, (0, 255, 255), -1)
        else:
            ear_label.config(text="EAR: -")
            thr_label.config(text="No face detected")
            counter = 0

        # show frame in its own window (OpenCV)
        cv2.imshow("Eye Alert System - Camera", img)
        # important for window responsiveness
        if cv2.waitKey(1) & 0xFF == 27:
            # user pressed ESC inside the OpenCV window
            with state_lock:
                running = False
            break

    # cleanup when loop ends
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Stopped")
    ear_label.config(text="EAR: -")
    thr_label.config(text="")

# ----- GUI command functions -----
def start_detection(status_label, ear_label, thr_label, start_btn, stop_btn, calib_btn):
    global running
    with state_lock:
        if running:
            messagebox.showinfo("Info", "Already running")
            return
        running = True
    start_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    calib_btn.config(state=tk.NORMAL)
    # spawn thread
    t = threading.Thread(target=detection_loop, args=(status_label, ear_label, thr_label), daemon=True)
    t.start()
    status_label.config(text="Starting...")

def stop_detection(start_btn, stop_btn, calib_btn):
    global running
    with state_lock:
        running = False
    start_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    calib_btn.config(state=tk.NORMAL)

def calibrate_now():
    global calibrating, calib_values, calibrated
    if not running:
        messagebox.showwarning("Warning", "Start detection first")
        return
    calibrating = True
    calib_values = []
    calibrated = False

# ----- Build GUI -----
def build_gui():
    root = tk.Tk()
    root.title("Eye Close Detection")
    root.geometry("350x220")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)

    title = ttk.Label(frame, text="Eye Alert System", font=("Segoe UI", 14, "bold"))
    title.pack(pady=(0,8))

    status_label = ttk.Label(frame, text="Stopped", font=("Segoe UI", 11))
    status_label.pack()

    ear_label = ttk.Label(frame, text="EAR: -", font=("Segoe UI", 10))
    ear_label.pack(pady=(6,0))

    thr_label = ttk.Label(frame, text="", font=("Segoe UI", 9))
    thr_label.pack()

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(pady=12)

    start_btn = ttk.Button(btn_frame, text="Start", width=10,
                           command=lambda: start_detection(status_label, ear_label, thr_label, start_btn, stop_btn, calib_btn))
    start_btn.grid(row=0, column=0, padx=6)

    stop_btn = ttk.Button(btn_frame, text="Stop", width=10,
                          command=lambda: stop_detection(start_btn, stop_btn, calib_btn))
    stop_btn.grid(row=0, column=1, padx=6)
    stop_btn.config(state=tk.DISABLED)

    calib_btn = ttk.Button(frame, text="Calibrate (keep eyes open)", width=28, command=calibrate_now)
    calib_btn.pack()
    calib_btn.config(state=tk.NORMAL)

    quit_btn = ttk.Button(frame, text="Quit", width=10, command=lambda: root.quit())
    quit_btn.pack(pady=(10,0))

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    build_gui()
