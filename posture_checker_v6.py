import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os
import threading

# ───────────────────────────────────────────────
# Asetukset
# ───────────────────────────────────────────────

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print(f"Mallitiedostoa EI loydy: {MODEL_PATH}")
    print("Lataa: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    exit(1)

# ───────────────────────────────────────────────
# Aanihälytys (cross-platform)
# ───────────────────────────────────────────────

def beep():
    try:
        import winsound
        winsound.Beep(900, 180)
    except ImportError:
        try:
            import subprocess
            subprocess.Popen(["beep", "-f", "900", "-l", "180"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            print('\a', end='', flush=True)

def beep_async():
    t = threading.Thread(target=beep, daemon=True)
    t.start()

# ───────────────────────────────────────────────
# Landmarker-tehdasfunktio
# ───────────────────────────────────────────────

def create_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    return PoseLandmarker.create_from_options(options)

# ───────────────────────────────────────────────
# Apufunktiot
# ───────────────────────────────────────────────

# Yhteydet ilman jalkoja (sivukamera)
SIDE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,12),(11,23),(12,24),(23,24)
]

# Kaikki yhteydet (etukamera)
ALL_CONNECTIONS = SIDE_CONNECTIONS + [
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32)
]

def draw_landmarks(image, detection_result, connections=ALL_CONNECTIONS):
    if not detection_result.pose_landmarks:
        return
    for landmark_list in detection_result.pose_landmarks:
        for lm in landmark_list:
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
        for a, b in connections:
            if a < len(landmark_list) and b < len(landmark_list):
                p1 = landmark_list[a]
                p2 = landmark_list[b]
                x1, y1 = int(p1.x * image.shape[1]), int(p1.y * image.shape[0])
                x2, y2 = int(p2.x * image.shape[1]), int(p2.y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (200, 200, 200), 2)


def analyze_front(landmarks, ref_delta_y, ref_ear_width):
    if not (landmarks[7].presence > 0.5 and landmarks[8].presence > 0.5 and
            landmarks[11].presence > 0.5 and landmarks[12].presence > 0.5):
        return False, "ei tarpeeksi nakyvyytta", (128, 128, 128), ""
    ear_width = abs(landmarks[7].x - landmarks[8].x)
    delta_y = ((landmarks[11].y - landmarks[7].y) + (landmarks[12].y - landmarks[8].y)) / 2
    is_raised = delta_y < 0.78 * ref_delta_y
    is_close  = ear_width > 1.22 * ref_ear_width
    warn = ""
    color = (0, 255, 0)
    is_bad = False
    if is_raised:
        warn += "hartiat koholla "
        color = (0, 120, 255)
        is_bad = True
    if is_close:
        warn += "paa edessa "
        color = (0, 0, 255)
        is_bad = True
    status = "HUONO RYHTI!" if is_bad else "ryhti ok"
    return is_bad, status, color, warn


def analyze_side(landmarks):
    key_points = [7, 8, 11, 12, 23, 24]
    if any(landmarks[i].presence < 0.6 for i in key_points):
        return False, "huono nakyvyys", (100, 100, 255), ""
    ear    = landmarks[8]
    should = landmarks[12]
    hip    = landmarks[24]
    v1 = np.array([should.x - ear.x,    should.y - ear.y])
    v2 = np.array([hip.x - should.x,    hip.y - should.y])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    if angle > 160:
        return False, "hyva profiili", (0, 220, 0), f"kulma ~{int(angle)}"
    elif angle > 135:
        return True, "lieva eteentaivutus", (0, 180, 255), f"kulma ~{int(angle)}"
    else:
        return True, "selva forward head!", (0, 0, 255), f"kulma {int(angle)} - korjaa!"

# ───────────────────────────────────────────────
# Valikkoikkunat
# ───────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX

def select_camera_mode():
    """Valitse 1 tai 2 kameraa."""
    cv2.namedWindow("Posture Checker - Kameravalinta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker - Kameravalinta", 720, 420)
    mode = None
    while mode is None:
        img = np.zeros((420, 720, 3), np.uint8)
        cv2.putText(img, "POSTURE CHECKER", (100, 70),  FONT, 1.6, (220, 220, 50), 3)
        cv2.putText(img, "Valitse kameroiden maara:", (80, 120), FONT, 0.95, (255, 255, 255), 2)
        cv2.putText(img, "1  =  Vain etukamera (webcam)", (80, 200), FONT, 0.9, (0, 255, 200), 2)
        cv2.putText(img, "2  =  Etukamera + sivukamera", (80, 260), FONT, 0.9, (0, 255, 200), 2)
        cv2.putText(img, "     (sivukameraksi suositellaan DroidCam)", (110, 295), FONT, 0.65, (180, 180, 255), 1)
        cv2.putText(img, "ESC = lopeta", (80, 370), FONT, 0.75, (100, 100, 255), 1)
        cv2.putText(img, "Paina 1 tai 2", (250, 400), FONT, 0.8, (255, 255, 100), 2)
        cv2.imshow("Posture Checker - Kameravalinta", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'):
            mode = 1
        elif k == ord('2'):
            mode = 2
        elif k == 27:
            cv2.destroyAllWindows()
            return None
    cv2.destroyWindow("Posture Checker - Kameravalinta")
    return mode


def show_action_menu():
    """Valitse Halytys / Seuranta / Lopeta."""
    cv2.namedWindow("Posture Checker - Toiminto", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker - Toiminto", 720, 450)
    choice = None
    while choice is None:
        img = np.zeros((450, 720, 3), np.uint8)
        cv2.putText(img, "POSTURE CHECKER", (100, 70),  FONT, 1.6, (220, 220, 50), 3)
        cv2.putText(img, "Valitse toiminto:", (80, 130), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "1  =  Halytys  (aani kun ryhti huononee)", (80, 210), FONT, 0.9, (0, 255, 200), 2)
        cv2.putText(img, "2  =  Seuranta (laskee huonon ryhdin ajan)", (80, 270), FONT, 0.9, (0, 255, 200), 2)
        cv2.putText(img, "3  =  Lopeta ohjelma", (80, 330), FONT, 0.9, (0, 100, 255), 2)
        cv2.putText(img, "Paina 1, 2 tai 3", (230, 410), FONT, 0.8, (255, 255, 100), 2)
        cv2.imshow("Posture Checker - Toiminto", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'):
            choice = 1
        elif k == ord('2'):
            choice = 2
        elif k == ord('3') or k == 27:
            choice = 3
    cv2.destroyWindow("Posture Checker - Toiminto")
    return choice


def show_stats(total_time, bad_time):
    """Nayta seurannan tilastot."""
    if total_time <= 0:
        return
    pct = int(bad_time / total_time * 100)
    img = np.zeros((380, 720, 3), np.uint8)
    cv2.putText(img, "SEURANTA PAATTYI",            (80, 80),  FONT, 1.5, (0, 255, 200), 3)
    cv2.putText(img, f"Kokonaisaika:  {int(total_time)} s", (80, 170), FONT, 1.0, (255, 255, 255), 2)
    cv2.putText(img, f"Huono ryhti:   {int(bad_time)} s ({pct} %)", (80, 220), FONT, 1.0, (0, 180, 255), 2)
    cv2.putText(img, "Paina mitä tahansa jatkaaksesi", (80, 310), FONT, 0.8, (180, 180, 255), 1)
    cv2.namedWindow("Seurannan tulokset", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seurannan tulokset", 720, 380)
    cv2.imshow("Seurannan tulokset", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Seurannan tulokset")

# ───────────────────────────────────────────────
# Yhteinen kamerasessio (yksi tai kaksi kameraa)
# ───────────────────────────────────────────────

def run_session(mode, action):
    """
    mode   : 1 = yksi kamera, 2 = kaksi kameraa
    action : 1 = halytys,     2 = seuranta
    """
    # Kamerat
    cap_front = cv2.VideoCapture(0)
    if not cap_front.isOpened():
        print("Etukamera ei aukea!")
        return

    cap_side = None
    if mode == 2:
        cap_side = cv2.VideoCapture(1)
        if not cap_side.isOpened():
            print("Sivukamera ei aukea - jatketaan yhdella kameralla")
            mode = 1

    lm_front = create_landmarker()
    lm_side  = create_landmarker() if mode == 2 else None

    # Tila
    calibrating   = False
    ref_delta_y   = None
    ref_ear_width = None
    calib_deltas, calib_ears = [], []

    bad_counter   = 0
    prev_time     = time.time()
    last_beep     = 0.0
    total_time    = 0.0
    bad_time      = 0.0

    ts_f = 0
    ts_s = 0

    # Frame-skip sivukameralle: prosessoidaan joka toinen ruutu
    side_frame_skip = 0
    last_result_s   = None

    win_title = "Posture Checker - Halytys" if action == 1 else "Posture Checker - Seuranta"

    while True:
        ret_f, frame_f = cap_front.read()
        if not ret_f:
            break

        frame_f = cv2.flip(frame_f, 1)
        frame_f = cv2.resize(frame_f, (800, 450))   # pienempi -> nopeampi

        frame_s = None
        if mode == 2 and cap_side is not None:
            ret_s, frame_s_raw = cap_side.read()
            if ret_s:
                frame_s = cv2.flip(frame_s_raw, 1)
                frame_s = cv2.resize(frame_s, (800, 450))

        # ── MediaPipe etukamera ───────────────────────────────────────────
        rgb_f   = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        mp_f    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
        result_f = lm_front.detect_for_video(mp_f, ts_f)
        ts_f += 1

        # ── MediaPipe sivukamera (joka toinen ruutu) ──────────────────────
        result_s = last_result_s
        if mode == 2 and frame_s is not None:
            side_frame_skip += 1
            if side_frame_skip % 2 == 0:
                rgb_s    = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
                mp_s     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_s)
                result_s = lm_side.detect_for_video(mp_s, ts_s)
                last_result_s = result_s
            ts_s += 1

        # ── Piirretään merkit ─────────────────────────────────────────────
        draw_landmarks(frame_f, result_f, ALL_CONNECTIONS)
        if mode == 2 and frame_s is not None and result_s is not None:
            draw_landmarks(frame_s, result_s, SIDE_CONNECTIONS)

        # ── Kalibrointi & analyysi ─────────────────────────────────────────
        is_bad_f  = False
        is_bad_s  = False
        status_f  = "front: ?"
        color_f   = (180, 180, 180)
        warn_f    = ""
        status_s  = "side: ?"
        color_s   = (180, 180, 180)
        warn_s    = ""

        if result_f.pose_landmarks:
            lm = result_f.pose_landmarks[0]

            if calibrating:
                if lm[7].presence > 0.5 and lm[11].presence > 0.5:
                    ear_w = abs(lm[7].x - lm[8].x)
                    d_y   = ((lm[11].y - lm[7].y) + (lm[12].y - lm[8].y)) / 2
                    calib_deltas.append(d_y)
                    calib_ears.append(ear_w)
                cv2.putText(frame_f, f"KALIBROINTI {len(calib_deltas)}/60",
                            (40, 80), FONT, 1.1, (0, 255, 255), 3)
                if len(calib_deltas) >= 60:
                    ref_delta_y   = np.mean(calib_deltas)
                    ref_ear_width = np.mean(calib_ears)
                    calibrating   = False
                    print(f"Kalibrointi valmis -> delta_y={ref_delta_y:.4f}, ear={ref_ear_width:.4f}")

            elif ref_delta_y is not None:
                is_bad_f, status_f, color_f, warn_f = analyze_front(lm, ref_delta_y, ref_ear_width)
            else:
                cv2.putText(frame_f, "Paina C kalibrointia varten",
                            (40, 80), FONT, 0.9, (0, 255, 255), 2)
                status_f = "kalibroi ensin"

        if result_s is not None and result_s.pose_landmarks:
            lm_s = result_s.pose_landmarks[0]
            is_bad_s, status_s, color_s, warn_s = analyze_side(lm_s)

        is_bad = is_bad_f or is_bad_s

        # ── Ajanotto ──────────────────────────────────────────────────────
        delta      = time.time() - prev_time
        prev_time  = time.time()
        if ref_delta_y is not None:
            total_time += delta
            if is_bad:
                bad_time += delta

        # ── Aanihälytys ───────────────────────────────────────────────────
        if action == 1 and is_bad and (time.time() - last_beep) > 1.5:
            beep_async()
            last_beep = time.time()

        # ── Näyttö ────────────────────────────────────────────────────────
        if mode == 1:
            vis = frame_f
        else:
            vis = np.hstack((frame_f, frame_s if frame_s is not None else np.zeros_like(frame_f)))

        h_vis, w_vis = vis.shape[:2]
        offset = frame_f.shape[1] if mode == 2 else 0

        # Etukameran teksti
        cv2.putText(vis, f"Etu: {status_f}", (30, 55), FONT, 1.0, color_f, 2)
        if warn_f:
            cv2.putText(vis, warn_f, (30, 100), FONT, 0.85, (0, 0, 255), 2)

        # Vahva varoitus
        if bad_counter > 35:
            cx = w_vis // 2 - (offset // 2 if mode == 2 else 0)
            cv2.putText(vis, "KORJAA RYHTI", (cx - 200, h_vis // 2),
                        FONT, 2.0, (0, 0, 255), 5)
        bad_counter = (bad_counter + 1) if is_bad else max(0, bad_counter - 1)

        # Sivukameran teksti
        if mode == 2:
            cv2.putText(vis, f"Sivu: {status_s}", (30 + offset, 55), FONT, 1.0, color_s, 2)
            if warn_s:
                cv2.putText(vis, warn_s, (30 + offset, 100), FONT, 0.85, (0, 0, 255), 2)

        # Seuranta-tilastot
        if action == 2 and total_time > 0:
            pct = int(bad_time / total_time * 100)
            cv2.putText(vis, f"Aika: {int(total_time)} s", (30, h_vis - 80),
                        FONT, 0.75, (255, 255, 100), 2)
            cv2.putText(vis, f"Huono: {int(bad_time)} s ({pct}%)", (30, h_vis - 55),
                        FONT, 0.75, (0, 180, 255), 2)

        # FPS
        fps = 1.0 / (delta + 1e-9)
        cv2.putText(vis, f"FPS: {int(fps)}", (w_vis - 160, 40), FONT, 0.75, (200, 255, 200), 2)

        # Ohjeet
        hint = "C=kalibroi  B=valikko  Q=lopeta"
        cv2.putText(vis, hint, (30, h_vis - 20), FONT, 0.65, (255, 255, 200), 1)

        cv2.imshow(win_title, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Paaohjelma lopettaa kokonaan
            cap_front.release()
            if cap_side:
                cap_side.release()
            lm_front.close()
            if lm_side:
                lm_side.close()
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('b'):
            break
        elif key == ord('c') and not calibrating:
            calibrating = True
            calib_deltas.clear()
            calib_ears.clear()
            print("Kalibrointi alkaa - istu suorassa!")

    # Siivous
    cap_front.release()
    if cap_side:
        cap_side.release()
    lm_front.close()
    if lm_side:
        lm_side.close()
    cv2.destroyAllWindows()

    # Seurannan tulokset
    if action == 2:
        show_stats(total_time, bad_time)


# ───────────────────────────────────────────────
# Paaohjelma
# ───────────────────────────────────────────────

print("Posture Checker kaynnistyy...")

cam_mode = select_camera_mode()
if cam_mode is None:
    print("Ei valintaa - lopetetaan")
    exit(0)

while True:
    action = show_action_menu()
    if action == 3:
        break
    run_session(cam_mode, action)

print("Ohjelma sulkeutui.")
