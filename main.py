#!/usr/bin/env python3
"""
Eye & Head Gesture Control for YouTube (Thonny-friendly)
---------------------------------------------------------
Features included:
- Eye gestures: double-blink (play/pause), long blink (mute)
- Eye movement: look LEFT/RIGHT (rewind/forward)
- Head turn (yaw): big head turns -> larger skip / repeated action
- Head tilt (roll): tilt left/right -> mapped to volume up/down (optional)
- Smart Auto-Pause when phone rings: pauses when face disappears or eyes shift away quickly

Dependencies:
    pip install opencv-python mediapipe numpy pynput

Thonny tips: open Tools -> Manage packages and install the packages there if "pip" from terminal fails.
If mediapipe installation fails on Windows, ensure you have a recent pip (pip install --upgrade pip) and try again.

Run: Open this file in Thonny and press Run (or run with python3).
Keep the YouTube tab focused so keystrokes go to the player.

"""

import time
from collections import deque
import math

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(
        "MediaPipe not found. Install with: pip install mediapipe\n"
        f"Original error: {e}"
    )

from pynput.keyboard import Controller, Key

# ---------------- Configuration ---------------- #
CALIBRATION_FRAMES = 60          # ~2 seconds at 30 FPS
SMOOTHING_WINDOW = 5             # moving average for gaze
GAZE_ACTIVATION = 0.35           # magnitude in [-1,1] to trigger left/right
ACTION_COOLDOWN = 0.9            # seconds between repeated actions
DOUBLE_BLINK_MAX_GAP = 0.40
LONG_BLINK_MIN = 1.00
FACE_AWAY_SECONDS = 0.8          # pause when face missing for this long
EYES_SHIFT_VARIANCE = 0.40       # sudden large gaze variance => possible distraction (phone ring)

USE_JL_KEYS = True               # True->'j'/'l' (10s), False->Left/Right arrows (5s)
DRAW_DEBUG = True

# Face mesh indices (MediaPipe) - iris landmarks require refine_landmarks=True
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

keyboard = Controller()

# ---------------- Utilities ---------------- #

def ear(points):
    p1, p2, p3, p4, p5, p6 = points
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def normalized_gaze_x(landmarks, w, h):
    def lm(idx):
        pt = landmarks[idx]
        return np.array([pt.x * w, pt.y * h])
    # left
    l_outer, l_inner = lm(LEFT_EYE_OUTER), lm(LEFT_EYE_INNER)
    l_iris = lm(LEFT_IRIS_CENTER)
    l_min, l_max = min(l_outer[0], l_inner[0]), max(l_outer[0], l_inner[0])
    l_norm = (l_iris[0] - l_min) / max(1.0, (l_max - l_min))
    l_norm = (l_norm - 0.5) * 2.0
    # right
    r_inner, r_outer = lm(RIGHT_EYE_INNER), lm(RIGHT_EYE_OUTER)
    r_iris = lm(RIGHT_IRIS_CENTER)
    r_min, r_max = min(r_outer[0], r_inner[0]), max(r_outer[0], r_inner[0])
    r_norm = (r_iris[0] - r_min) / max(1.0, (r_max - r_min))
    r_norm = (r_norm - 0.5) * 2.0
    g = float(np.clip((l_norm + r_norm) / 2.0, -1.0, 1.0))
    return g


def gather_points(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        ll = landmarks[i]
        pts.append(np.array([ll.x * w, ll.y * h]))
    return pts


def eye_center_from_corners(landmarks, w, h):
    # use outer/inner corners to compute a stable eye center
    left_c = (np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h]) +
              np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])) / 2.0
    right_c = (np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h]) +
               np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])) / 2.0
    return left_c, right_c


def press_key(key):
    try:
        if key == "left":
            keyboard.press(Key.left); keyboard.release(Key.left)
        elif key == "right":
            keyboard.press(Key.right); keyboard.release(Key.right)
        elif key == "pause":
            keyboard.press('k'); keyboard.release('k')
        elif key == "rewind":
            if USE_JL_KEYS:
                keyboard.press('j'); keyboard.release('j')
            else:
                keyboard.press(Key.left); keyboard.release(Key.left)
        elif key == "forward":
            if USE_JL_KEYS:
                keyboard.press('l'); keyboard.release('l')
            else:
                keyboard.press(Key.right); keyboard.release(Key.right)
        elif key == "mute":
            keyboard.press('m'); keyboard.release('m')
        elif key == "vol_up":
            # volume up mapping — you can map to other keys if needed
            keyboard.press(Key.up); keyboard.release(Key.up)
        elif key == "vol_down":
            keyboard.press(Key.down); keyboard.release(Key.down)
    except Exception as e:
        print("Key press failed:", e)


# ---------------- Main application ---------------- #

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ear_open_samples = []
    gaze_zero_samples = []
    gaze_buffer = deque(maxlen=SMOOTHING_WINDOW)
    gaze_short_history = deque(maxlen=8)  # for detecting sudden shifts

    eye_closed = False
    eye_closed_start = 0.0
    last_blink_time = 0.0
    EAR_THRESH = None
    GAZE_BIAS = 0.0

    last_action_time = {"rewind": 0.0, "forward": 0.0, "pause": 0.0, "mute": 0.0, "vol": 0.0}

    frame_count = 0
    last_face_time = time.time()
    paused_by_autopause = False

    fps_time = time.time(); fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        now = time.time()

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            last_face_time = now
            paused_by_autopause = False

            left_pts = gather_points(lms, LEFT_EYE_EAR, w, h)
            right_pts = gather_points(lms, RIGHT_EYE_EAR, w, h)
            left_ear = ear(left_pts); right_ear = ear(right_pts)
            mean_ear = (left_ear + right_ear) / 2.0

            gaze_x = normalized_gaze_x(lms, w, h)
            gaze_buffer.append(gaze_x); gaze_short_history.append(gaze_x)

            # head metrics
            left_eye_center, right_eye_center = eye_center_from_corners(lms, w, h)
            inter_eye_dist = np.linalg.norm(right_eye_center - left_eye_center)
            # yaw-like measure: positive when looking right
            head_yaw = ((right_eye_center[0] - left_eye_center[0]) - inter_eye_dist) / max(1.0, inter_eye_dist)
            # roll (tilt) angle in degrees
            dx = right_eye_center[0] - left_eye_center[0]
            dy = right_eye_center[1] - left_eye_center[1]
            head_tilt_deg = math.degrees(math.atan2(dy, dx))

            if frame_count < CALIBRATION_FRAMES:
                ear_open_samples.append(mean_ear)
                gaze_zero_samples.append(gaze_x)
                cv2.putText(frame, f"Calibrating... {frame_count}/{CALIBRATION_FRAMES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                if EAR_THRESH is None:
                    base_ear = float(np.median(ear_open_samples)) if ear_open_samples else 0.28
                    EAR_THRESH = max(0.14, base_ear * 0.72)
                    GAZE_BIAS = float(np.mean(gaze_zero_samples)) if gaze_zero_samples else 0.0

                smooth_gaze = float(np.mean(gaze_buffer)) - GAZE_BIAS

                # ----- Blink detection -----
                if mean_ear < EAR_THRESH and not eye_closed:
                    eye_closed = True; eye_closed_start = now
                elif mean_ear >= EAR_THRESH and eye_closed:
                    closed_dur = now - eye_closed_start
                    eye_closed = False
                    if closed_dur < LONG_BLINK_MIN:
                        # normal blink -> double-blink check
                        if (now - last_blink_time) <= DOUBLE_BLINK_MAX_GAP:
                            if (now - last_action_time["pause"]) >= ACTION_COOLDOWN:
                                press_key("pause"); last_action_time["pause"] = now
                        last_blink_time = now
                    else:
                        if (now - last_action_time["mute"]) >= ACTION_COOLDOWN:
                            press_key("mute"); last_action_time["mute"] = now

                # ----- Gaze-based skip -----
                # If user looks strongly left/right continuously, trigger skip
                if smooth_gaze <= -GAZE_ACTIVATION:
                    if (now - last_action_time["rewind"]) >= ACTION_COOLDOWN:
                        press_key("rewind"); last_action_time["rewind"] = now
                elif smooth_gaze >= GAZE_ACTIVATION:
                    if (now - last_action_time["forward"]) >= ACTION_COOLDOWN:
                        press_key("forward"); last_action_time["forward"] = now

                # ----- Head-turn (yaw) handling -----
                # Larger head turns cause repeated/stronger skips. head_yaw is approximate.
                if head_yaw < -0.18:
                    # turned left strongly
                    if (now - last_action_time["rewind"]) >= ACTION_COOLDOWN:
                        press_key("rewind"); last_action_time["rewind"] = now
                elif head_yaw > 0.18:
                    if (now - last_action_time["forward"]) >= ACTION_COOLDOWN:
                        press_key("forward"); last_action_time["forward"] = now

                # ----- Head tilt mapping (optional) -----
                # tilt left -> volume down; tilt right -> volume up (small threshold)
                if head_tilt_deg < -12:
                    if (now - last_action_time["vol"]) >= 0.6:
                        press_key("vol_down"); last_action_time["vol"] = now
                elif head_tilt_deg > 12:
                    if (now - last_action_time["vol"]) >= 0.6:
                        press_key("vol_up"); last_action_time["vol"] = now

                # ----- Smart Auto-Pause on eyes shifting (phone ring) -----
                # if gaze variance in short history exceeds threshold -> pause
                if len(gaze_short_history) >= 5:
                    var = float(np.std(np.array(gaze_short_history)))
                    if var >= EYES_SHIFT_VARIANCE and (now - last_action_time["pause"]) >= ACTION_COOLDOWN:
                        press_key("pause"); last_action_time["pause"] = now

                # Draw debug overlays
                if DRAW_DEBUG:
                    cx = int((w / 2) * (1 + np.clip(smooth_gaze, -1, 1)))
                    cv2.line(frame, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)
                    cv2.circle(frame, (cx, h - 30), 12, (0, 255, 0), -1)
                    txt = f"gaze={smooth_gaze:+.2f} EAR={mean_ear:.2f} thr={EAR_THRESH:.2f} yaw={head_yaw:+.2f} tilt={head_tilt_deg:+.0f}" 
                    cv2.putText(frame, txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        else:
            # No face detected
            if (now - last_face_time) >= FACE_AWAY_SECONDS:
                # send pause if not already done recently
                if (now - last_action_time["pause"]) >= ACTION_COOLDOWN:
                    press_key("pause"); last_action_time["pause"] = now
                    paused_by_autopause = True
            if DRAW_DEBUG:
                cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # FPS estimate
        frame_count += 1
        if frame_count % 10 == 0:
            t = time.time(); dt = t - fps_time
            fps = 10.0 / dt if dt > 0 else fps
            fps_time = t
        if DRAW_DEBUG:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Eye+Head Gesture Control (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
