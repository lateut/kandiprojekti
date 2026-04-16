import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os
import threading
import platform

# --------------------------------------------------
# Optional Windows beep
# --------------------------------------------------
HAS_WINSOUND = False
if platform.system().lower() == "windows":
    try:
        import winsound
        HAS_WINSOUND = True
    except ImportError:
        HAS_WINSOUND = False

# --------------------------------------------------
# Settings
# --------------------------------------------------
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"

# Smaller processing size = better speed
PROCESS_W = 640
PROCESS_H = 360

# Display size (single camera)
DISPLAY_W = 960
DISPLAY_H = 540

# Two-camera display size (each camera)
DISPLAY_W_DUAL = 480
DISPLAY_H_DUAL = 540

# Side camera index (change if needed)
SIDE_CAMERA_INDEX = 1

# Calibration
CALIBRATION_FRAMES = 45

# Alert
BEEP_INTERVAL_SEC = 1.0
BEEP_FREQ = 900
BEEP_DURATION_MS = 180

# Front view thresholds
FRONT_SHOULDER_RATIO = 0.78
FRONT_HEAD_FORWARD_RATIO = 1.22

# Side view thresholds
SIDE_GOOD_ANGLE = 160
SIDE_WARN_ANGLE = 135

# Smoothing
STATUS_BUFFER_LEN = 6

# Added calibration for side view
SIDE_CALIBRATION_ANGLE = 170  # Expected good angle during calibration

# --------------------------------------------------
# Model check
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    print("Download it and place it in the same folder:")
    print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    raise SystemExit(1)

# --------------------------------------------------
# Pose landmarker
# --------------------------------------------------
def create_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_segmentation_masks=False
    )
    return PoseLandmarker.create_from_options(options)

# --------------------------------------------------
# Drawing
# --------------------------------------------------
UPPER_BODY_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,12),
    (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
    (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),
    (11,23), (12,24), (23,24)
]

# Even lighter side camera drawing
SIDE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (11,12),
    (11,13), (13,15),
    (12,14), (14,16),
    (11,23), (12,24), (23,24)
]

def draw_landmarks(image, detection_result, side_mode=False):
    if not detection_result or not detection_result.pose_landmarks:
        return

    connections = SIDE_CONNECTIONS if side_mode else UPPER_BODY_CONNECTIONS

    for landmark_list in detection_result.pose_landmarks:
        used_points = set()
        for a, b in connections:
            used_points.add(a)
            used_points.add(b)

        for idx in used_points:
            if idx < len(landmark_list):
                lm = landmark_list[idx]
                x = int(lm.x * image.shape[1])
                y = int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        for a, b in connections:
            if a < len(landmark_list) and b < len(landmark_list):
                p1 = landmark_list[a]
                p2 = landmark_list[b]
                x1, y1 = int(p1.x * image.shape[1]), int(p1.y * image.shape[0])
                x2, y2 = int(p2.x * image.shape[1]), int(p2.y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (220, 220, 220), 2)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def beep_async():
    if not HAS_WINSOUND:
        return
    threading.Thread(target=lambda: winsound.Beep(BEEP_FREQ, BEEP_DURATION_MS), daemon=True).start()

def safe_presence(landmarks, indices, threshold=0.5):
    for i in indices:
        if i >= len(landmarks):
            return False
        if getattr(landmarks[i], "presence", 1.0) < threshold:
            return False
    return True

def analyze_front_view(landmarks, ref_delta_y, ref_ear_width):
    needed = [7, 8, 11, 12]
    if not safe_presence(landmarks, needed, 0.5):
        return {
            "ok": False,
            "status": "front not visible",
            "warning": "",
            "color": (140, 140, 140),
            "bad": False
        }

    ear_width = abs(landmarks[7].x - landmarks[8].x)
    left_d = landmarks[11].y - landmarks[7].y
    right_d = landmarks[12].y - landmarks[8].y
    delta_y = (left_d + right_d) / 2

    is_raised = delta_y < FRONT_SHOULDER_RATIO * ref_delta_y
    is_close = ear_width > FRONT_HEAD_FORWARD_RATIO * ref_ear_width

    warning_parts = []
    if is_raised:
        warning_parts.append("shoulders too high")
    if is_close:
        warning_parts.append("head too forward")

    bad = is_raised or is_close

    return {
        "ok": True,
        "status": "front bad" if bad else "front ok",
        "warning": " | ".join(warning_parts),
        "color": (0, 0, 255) if bad else (0, 220, 0),
        "bad": bad
    }

def analyze_side_view(landmarks, ref_angle=None):
    needed = [8, 12, 24]
    if not safe_presence(landmarks, needed, 0.55):
        return {
            "ok": False,
            "status": "side not visible",
            "warning": "",
            "color": (140, 140, 140),
            "bad": False
        }

    ear = landmarks[8]
    shoulder = landmarks[12]
    hip = landmarks[24]

    v1 = np.array([shoulder.x - ear.x, shoulder.y - ear.y], dtype=np.float32)
    v2 = np.array([hip.x - shoulder.x, hip.y - shoulder.y], dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return {
            "ok": False,
            "status": "side unclear",
            "warning": "",
            "color": (140, 140, 140),
            "bad": False
        }

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    if ref_angle is not None:
        angle_diff = abs(angle - ref_angle)
        if angle > 20:  # Allow some tolerance
            return {
                "ok": True,
                "status": f"side bad ({int(angle)} deg)",
                "warning": "clear forward head",
                "color": (0, 0, 255),
                "bad": True
            }
        else:
            return {
                "ok": True,
                "status": f"side ok ({int(angle)} deg)",
                "warning": "",
                "color": (0, 220, 0),
                "bad": False
            }

    if angle > SIDE_GOOD_ANGLE:
        return {
            "ok": True,
            "status": f"side ok ({int(angle)} deg)",
            "warning": "",
            "color": (0, 220, 0),
            "bad": False
        }
    elif angle > SIDE_WARN_ANGLE:
        return {
            "ok": True,
            "status": f"side warning ({int(angle)} deg)",
            "warning": "mild forward head",
            "color": (0, 180, 255),
            "bad": True
        }
    else:
        return {
            "ok": True,
            "status": f"side bad ({int(angle)} deg)",
            "warning": "clear forward head",
            "color": (0, 0, 255),
            "bad": True
        }

def resize_for_processing(frame):
    return cv2.resize(frame, (PROCESS_W, PROCESS_H))

def resize_for_display(frame, dual_camera_mode=False):
    if dual_camera_mode:
        return cv2.resize(frame, (DISPLAY_W_DUAL, DISPLAY_H_DUAL))
    else:
        return cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

def detect_pose(landmarker, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_img, timestamp_ms)

def put_multiline_text(img, lines, start_xy=(40, 60), line_gap=40,
                       font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.9, color=(255,255,255), thick=2):
    x, y = start_xy
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, thick)
        y += line_gap

def show_screen(title, lines, width=900, height=600):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)

    while True:
        img = np.zeros((height, width, 3), np.uint8)
        cv2.putText(img, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 50), 3)
        put_multiline_text(img, lines, start_xy=(40, 140), line_gap=45, scale=0.9, color=(255,255,255), thick=2)
        cv2.imshow(title, img)
        key = cv2.waitKey(0) & 0xFF
        if key != 255:
            break

    cv2.destroyWindow(title)
    return key

# --------------------------------------------------
# Menus
# --------------------------------------------------
def select_camera_mode():
    title = "Posture Checker - Start"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 900, 550)

    while True:
        img = np.zeros((550, 900, 3), np.uint8)

        cv2.putText(img, "POSTURE CHECKER", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (220, 220, 50), 4)
        cv2.putText(img, "Choose camera mode", (280, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(img, "1 = One camera", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "    Front webcam only", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)

        cv2.putText(img, "2 = Two cameras", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "    Front webcam + side camera", (150, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)
        cv2.putText(img, f"    Side camera index now: {SIDE_CAMERA_INDEX}", (150, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)

        cv2.putText(img, "3 = Exit", (100, 490), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

        cv2.imshow(title, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            cv2.destroyWindow(title)
            return 1
        elif key == ord('2'):
            cv2.destroyWindow(title)
            return 2
        elif key == ord('3') or key == 27:
            cv2.destroyWindow(title)
            return 0

def select_action_menu(camera_mode):
    title = "Posture Checker - Menu"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 900, 550)

    cam_text = "One camera mode" if camera_mode == 1 else "Two camera mode"

    while True:
        img = np.zeros((550, 900, 3), np.uint8)

        cv2.putText(img, "POSTURE CHECKER", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (220, 220, 50), 4)
        cv2.putText(img, cam_text, (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(img, "1 = Alert", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "    Beep when posture becomes bad", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)

        cv2.putText(img, "2 = Tracking", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "    Count bad posture seconds", (150, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 1)

        cv2.putText(img, "3 = Exit", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

        cv2.imshow(title, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            cv2.destroyWindow(title)
            return 1
        elif key == ord('2'):
            cv2.destroyWindow(title)
            return 2
        elif key == ord('3') or key == 27:
            cv2.destroyWindow(title)
            return 0

# --------------------------------------------------
# Camera open
# --------------------------------------------------
def open_cameras(camera_mode):
    cap_front = cv2.VideoCapture(0)
    if not cap_front.isOpened():
        print("Front camera did not open.")
        return None, None, 0

    cap_side = None
    actual_mode = camera_mode

    if camera_mode == 2:
        cap_side = cv2.VideoCapture(SIDE_CAMERA_INDEX)
        if not cap_side.isOpened():
            print("Side camera did not open. Falling back to one camera mode.")
            cap_side = None
            actual_mode = 1

    return cap_front, cap_side, actual_mode

# --------------------------------------------------
# Core runner
# --------------------------------------------------
def run_mode(camera_mode=1, action_mode="alert"):
    cap_front, cap_side, actual_mode = open_cameras(camera_mode)
    if cap_front is None:
        return "menu"

    landmarker_front = create_landmarker()
    landmarker_side = create_landmarker() if actual_mode == 2 else None

    calibrating = False
    ref_delta_y = None
    ref_ear_width = None
    calib_deltas = []
    calib_ear_widths = []

    last_beep = 0.0
    prev_loop_time = time.time()

    total_time = 0.0
    bad_time = 0.0

    front_status_hist = []
    side_status_hist = []

    title = f"Posture Checker - {'Alert' if action_mode == 'alert' else 'Tracking'}"
    
    # Set up window size based on camera mode
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if actual_mode == 2:
        # Two camera mode: 480+480=960 width, plus some margin
        cv2.resizeWindow(title, 1000, 580)
    else:
        # Single camera mode
        cv2.resizeWindow(title, 980, 570)

    while True:
        frame_start = time.time()

        ret_f, frame_f = cap_front.read()
        if not ret_f:
            print("Front camera stream ended.")
            break

        frame_f = cv2.flip(frame_f, 1)
        frame_f_proc = resize_for_processing(frame_f)
        frame_f_show = resize_for_display(frame_f, dual_camera_mode=(actual_mode == 2))

        frame_s_proc = None
        frame_s_show = None
        result_s = None

        if actual_mode == 2:
            ret_s, frame_s = cap_side.read()
            if not ret_s:
                print("Side camera stream ended.")
                break
            frame_s = cv2.flip(frame_s, 1)
            frame_s_proc = resize_for_processing(frame_s)
            frame_s_show = resize_for_display(frame_s, dual_camera_mode=True)

        timestamp_ms = int(time.monotonic() * 1000)

        result_f = detect_pose(landmarker_front, frame_f_proc, timestamp_ms)
        if actual_mode == 2:
            result_s = detect_pose(landmarker_side, frame_s_proc, timestamp_ms + 1)

        # Draw on display frames using detections from processing frames
        draw_landmarks(frame_f_show, result_f, side_mode=False)
        if actual_mode == 2 and result_s is not None:
            draw_landmarks(frame_s_show, result_s, side_mode=True)

        front_info = {
            "ok": False,
            "status": "front no calib",
            "warning": "",
            "color": (0, 255, 255),
            "bad": False
        }

        side_info = {
            "ok": False,
            "status": "side idle",
            "warning": "",
            "color": (180, 180, 180),
            "bad": False
        }

        # Calibration uses front camera only
        if result_f.pose_landmarks:
            lm_f = result_f.pose_landmarks[0]

            if calibrating:
                if safe_presence(lm_f, [7, 8, 11, 12], 0.5):
                    ear_w = abs(lm_f[7].x - lm_f[8].x)
                    d_y = ((lm_f[11].y - lm_f[7].y) + (lm_f[12].y - lm_f[8].y)) / 2
                    calib_deltas.append(d_y)
                    calib_ear_widths.append(ear_w)

                cv2.putText(frame_f_show, f"CALIBRATING {len(calib_deltas)}/{CALIBRATION_FRAMES}",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                if len(calib_deltas) >= CALIBRATION_FRAMES:
                    ref_delta_y = float(np.mean(calib_deltas))
                    ref_ear_width = float(np.mean(calib_ear_widths))
                    calibrating = False
                    print("Calibration finished.")

            elif ref_delta_y is not None:
                front_info = analyze_front_view(lm_f, ref_delta_y, ref_ear_width)

        if actual_mode == 2 and result_s and result_s.pose_landmarks:
            lm_s = result_s.pose_landmarks[0]
            side_info = analyze_side_view(lm_s, ref_angle=SIDE_CALIBRATION_ANGLE)

        # Smoothing
        front_status_hist.append(front_info["bad"])
        if len(front_status_hist) > STATUS_BUFFER_LEN:
            front_status_hist.pop(0)

        smoothed_front_bad = sum(front_status_hist) >= (STATUS_BUFFER_LEN // 2 + 1)

        if actual_mode == 2:
            side_status_hist.append(side_info["bad"])
            if len(side_status_hist) > STATUS_BUFFER_LEN:
                side_status_hist.pop(0)
            smoothed_side_bad = sum(side_status_hist) >= (STATUS_BUFFER_LEN // 2 + 1)
        else:
            smoothed_side_bad = False

        overall_bad = smoothed_front_bad or smoothed_side_bad if actual_mode == 2 else smoothed_front_bad

        # Time stats
        now = time.time()
        dt = now - prev_loop_time
        prev_loop_time = now

        total_time += dt
        if ref_delta_y is not None and overall_bad:
            bad_time += dt

        # Alert
        if action_mode == "alert" and ref_delta_y is not None and overall_bad:
            if time.time() - last_beep >= BEEP_INTERVAL_SEC:
                beep_async()
                last_beep = time.time()

        # Display
        if actual_mode == 1:
            vis = frame_f_show
        else:
            vis = np.hstack((frame_f_show, frame_s_show))

        # Top-left: Front camera status
        y_pos = 35
        cv2.putText(vis, f"Front: {front_info['status']}", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, front_info["color"], 2)
        y_pos += 32
        if front_info["warning"]:
            cv2.putText(vis, front_info["warning"], (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)
            y_pos += 28

        # Top-right: Side camera status or Tracking stats
        if actual_mode == 2:
            # Dual camera mode: show side camera info on the right
            offset = frame_f_show.shape[1]
            y_pos = 35
            cv2.putText(vis, f"Side: {side_info['status']}", (offset + 15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, side_info["color"], 2)
            y_pos += 32
            if side_info["warning"]:
                cv2.putText(vis, side_info["warning"], (offset + 15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)
        else:
            # Single camera mode: show tracking stats on the right if in tracking mode
            if action_mode == "tracking":
                percent = int((bad_time / total_time) * 100) if total_time > 0 else 0
                y_pos = 35
                cv2.putText(vis, f"Tracking: {int(total_time)} sec", (vis.shape[1] - 260, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 150), 2)
                y_pos += 32
                cv2.putText(vis, f"Bad: {int(bad_time)} sec ({percent}%)", (vis.shape[1] - 260, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

        # Center: Overall status (large text)
        if ref_delta_y is None and not calibrating:
            overall_text = "Press C to calibrate"
            overall_color = (0, 255, 255)
        elif calibrating:
            overall_text = "Calibrating..."
            overall_color = (0, 255, 255)
        else:
            overall_text = "BAD POSTURE" if overall_bad else "POSTURE OK"
            overall_color = (0, 0, 255) if overall_bad else (0, 220, 0)

        text_size = cv2.getTextSize(overall_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        text_x = (vis.shape[1] - text_size[0]) // 2
        text_y = (vis.shape[0] // 2) + 20
        cv2.putText(vis, overall_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, overall_color, 3)

        if overall_bad and ref_delta_y is not None:
            fix_text = "Fix your posture"
            fix_size = cv2.getTextSize(fix_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
            fix_x = (vis.shape[1] - fix_size[0]) // 2
            cv2.putText(vis, fix_text, (fix_x, text_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        # Tracking stats for dual camera mode
        if actual_mode == 2 and action_mode == "tracking":
            percent = int((bad_time / total_time) * 100) if total_time > 0 else 0
            y_pos = vis.shape[0] - 95
            cv2.putText(vis, f"Tracking: {int(total_time)} sec", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 150), 2)
            cv2.putText(vis, f"Bad: {int(bad_time)} sec ({percent}%)", (15, y_pos + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)

        # Bottom-right: FPS
        fps = 1.0 / max(1e-6, time.time() - frame_start)
        cv2.putText(vis, f"FPS: {int(fps)}", (vis.shape[1] - 130, vis.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom: Instructions
        cv2.putText(vis, "C=calibrate  B=back  Q=exit", (15, vis.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 200), 2)

        cv2.imshow(title, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cleanup(cap_front, cap_side, landmarker_front, landmarker_side)
            cv2.destroyAllWindows()
            return "quit"

        elif key == ord('b'):
            cleanup(cap_front, cap_side, landmarker_front, landmarker_side)
            cv2.destroyAllWindows()
            if action_mode == "tracking":
                show_stats(total_time, bad_time)
            return "menu"

        elif key == ord('c'):
            calibrating = True
            calib_deltas.clear()
            calib_ear_widths.clear()
            ref_delta_y = None
            ref_ear_width = None
            front_status_hist.clear()
            side_status_hist.clear()
            print("Calibration started. Sit straight and stay still for a moment.")

    cleanup(cap_front, cap_side, landmarker_front, landmarker_side)
    cv2.destroyAllWindows()

    if action_mode == "tracking":
        show_stats(total_time, bad_time)

    return "menu"

def cleanup(cap_front, cap_side, landmarker_front, landmarker_side):
    try:
        if cap_front is not None:
            cap_front.release()
    except:
        pass

    try:
        if cap_side is not None:
            cap_side.release()
    except:
        pass

    try:
        if landmarker_front is not None:
            landmarker_front.close()
    except:
        pass

    try:
        if landmarker_side is not None:
            landmarker_side.close()
    except:
        pass

def show_stats(total_time, bad_time):
    """Näytä seurannan tilastot."""
    if total_time <= 0:
        return
    pct = int(bad_time / total_time * 100)
    img = np.zeros((380, 720, 3), np.uint8)
    cv2.putText(img, "SEURANTA PÄÄTTYI",            (80, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 200), 3)
    cv2.putText(img, f"Kokonaisaika:  {int(total_time)} s", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, f"Huono ryhti:   {int(bad_time)} s ({pct} %)", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
    cv2.putText(img, "Paina mitä tahansa jatkaaksesi", (80, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 1)
    cv2.namedWindow("Seurannan tulokset", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seurannan tulokset", 720, 380)
    cv2.imshow("Seurannan tulokset", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Seurannan tulokset")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Posture Checker starting...")

    while True:
        camera_mode = select_camera_mode()

        if camera_mode == 0:
            break

        action = select_action_menu(camera_mode)

        if action == 0:
            break
        elif action == 1:
            result = run_mode(camera_mode, action_mode="alert")
            if result == "quit":
                break
        elif action == 2:
            result = run_mode(camera_mode, action_mode="tracking")
            if result == "quit":
                break

    cv2.destroyAllWindows()
    print("Program closed.")

if __name__ == "__main__":
    main()