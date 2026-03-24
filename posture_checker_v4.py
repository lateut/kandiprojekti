import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os

# ───────────────────────────────────────────────
# Asetukset
# ───────────────────────────────────────────────

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print(f"Mallitiedostoa EI löydy: {MODEL_PATH}")
    print("Lataa: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    exit(1)

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

def draw_landmarks(image, detection_result):
    if not detection_result.pose_landmarks:
        return
    for landmark_list in detection_result.pose_landmarks:
        # Piirtää pisteet
        for lm in landmark_list:
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # Yhteydet (33 landmarkia)
        connections = [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),       # pää
            (9,10),
            (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
            (11,12),(11,23),(12,24),(23,24),                        # vartalo
            (23,25),(25,27),(27,29),(27,31),(25,27),                # vasen jalka
            (24,26),(26,28),(28,30),(28,32),(26,28)                 # oikea jalka
        ]
        for a, b in connections:
            if a < len(landmark_list) and b < len(landmark_list):
                p1 = landmark_list[a]
                p2 = landmark_list[b]
                x1,y1 = int(p1.x*image.shape[1]), int(p1.y*image.shape[0])
                x2,y2 = int(p2.x*image.shape[1]), int(p2.y*image.shape[0])
                cv2.line(image, (x1,y1), (x2,y2), (200,200,200), 2)


def analyze_front_view(landmarks, ref_delta_y, ref_ear_width):
    if not (landmarks[7].presence > 0.5 and landmarks[8].presence > 0.5 and
            landmarks[11].presence > 0.5 and landmarks[12].presence > 0.5):
        return "ei tarpeeksi näkyvyyttä", (128,128,128), ""

    ear_width = abs(landmarks[7].x - landmarks[8].x)
    left_d  = landmarks[11].y - landmarks[7].y
    right_d = landmarks[12].y - landmarks[8].y
    delta_y = (left_d + right_d) / 2

    warning = ""
    color = (0,255,0)
    status = "ok"

    is_raised = delta_y < 0.78 * ref_delta_y
    is_close  = ear_width > 1.22 * ref_ear_width

    if is_raised:
        warning += "hartiat koholla "
        color = (0,120,255)
        status = "huono"
    if is_close:
        warning += "pää edessä "
        color = (0,0,255)
        status = "huono"

    return status, color, warning


def analyze_side_view(landmarks):
    key_points = [7, 8, 11, 12, 23, 24]
    if any(landmarks[i].presence < 0.6 for i in key_points):
        return "huono näkyvyys sivusta", (100,100,255), ""

    ear    = landmarks[8]   # oikea korva
    should = landmarks[12]  # oikea olkapää
    hip    = landmarks[24]  # oikea lantio

    v1 = np.array([should.x - ear.x, should.y - ear.y])
    v2 = np.array([hip.x - should.x, hip.y - should.y])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    if angle > 160:
        return "hyvä profiili", (0,220,0), f"kulma ~{int(angle)}°"
    elif angle > 135:
        return "lievä eteentaivutus", (0,180,255), f"kulma ~{int(angle)}°"
    else:
        return "selvä forward head!", (0,0,255), f"kulma {int(angle)}° – korjaa!"


# ───────────────────────────────────────────────
# Kamera-valinta ikkuna (ensimmäinen ruutu)
# ───────────────────────────────────────────────

def select_mode():
    cv2.namedWindow("Posture Checker – Kameroiden valinta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker – Kameroiden valinta", 700, 400)

    font = cv2.FONT_HERSHEY_SIMPLEX
    mode = None

    while mode is None:
        img = np.zeros((400, 700, 3), np.uint8)
        cv2.putText(img, "Posture Checker", (80, 60), font, 1.3, (220, 220, 50), 3)
        cv2.putText(img, "Valitse kameroiden määrä", (80, 110), font, 1.0, (255, 255, 255), 2)

        cv2.putText(img, "1 = Vain etukamera (webcam)", (80, 190), font, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "   → Nykyinen yksikameratoteutus", (120, 220), font, 0.7, (180, 180, 255), 1)

        cv2.putText(img, "2 = Etukamera + sivukamera", (80, 270), font, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "   → Sivukameraksi suositellaan DroidCam (puhelin)", (120, 300), font, 0.7, (180, 180, 255), 1)
        cv2.putText(img, "   (DroidCam yleensä kamera-indeksi 1 tai 2 – vaihda tarvittaessa)", (120, 330), font, 0.7, (180, 180, 255), 1)

        cv2.putText(img, "Paina 1 tai 2 → Enter sulkee ikkunan", (80, 370), font, 0.75, (100, 255, 100), 1)

        cv2.imshow("Posture Checker – Kameroiden valinta", img)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'):
            mode = 1
        elif k == ord('2'):
            mode = 2
        elif k == 27:  # ESC
            return None

    cv2.destroyWindow("Posture Checker – Kameroiden valinta")
    return mode


# ───────────────────────────────────────────────
# Pääohjelma
# ───────────────────────────────────────────────

mode = select_mode()
if mode is None:
    print("Ei valintaa → lopetetaan")
    exit(0)

# ── Kameroiden käynnistys (otettu toimivasta versiosta) ─────────────────────
cap_front = cv2.VideoCapture(0)           # etukamera

if mode == 2:
    cap_side = cv2.VideoCapture(1)        # sivukamera (DroidCam)
    print("Käytetään kahta kameraa → etu (0) + sivu (1)")
    print("   Jos sivukamera ei aukea, vaihda indeksi cap_side = cv2.VideoCapture(X)")
else:
    cap_side = None

if not cap_front.isOpened():
    print("Etukamera ei aukea!")
    exit(1)

if mode == 2 and (cap_side is None or not cap_side.isOpened()):
    print("Sivukamera ei aukea → jatketaan vain yhdellä kameralla")
    mode = 1

# ── Eri landmarkerit kummallekin kameralle (tärkein muutos) ───────────────
landmarker_front = create_landmarker()
landmarker_side = None
if mode == 2:
    landmarker_side = create_landmarker()

# Ikkunan alustus
cv2.namedWindow("Posture Checker", cv2.WINDOW_NORMAL)
if mode == 1:
    cv2.resizeWindow("Posture Checker", 960, 540)
else:
    cv2.resizeWindow("Posture Checker", 1920, 540)

# Kalibrointimuuttujat (vain front view)
calibrating = False
ref_delta_y = None
ref_ear_width = None
calib_deltas = []
calib_ear_widths = []

prev_time = time.time()
timestamp_front = 0
timestamp_side = 0

while True:
    # ── Kameroiden lukeminen (toimiva metodi toisesta versiosta) ───────────
    ret_f, frame_f = cap_front.read()
    if not ret_f:
        print("Etukamera katkesi")
        break

    frame_f = cv2.flip(frame_f, 1)
    frame_f = cv2.resize(frame_f, (960, 540))

    frame_s = None
    if mode == 2:
        ret_s, frame_s = cap_side.read()
        if not ret_s:
            print("Sivukamera katkesi → lopetetaan kaksoistila")
            break
        frame_s = cv2.flip(frame_s, 1)
        frame_s = cv2.resize(frame_s, (960, 540))

    # ── MediaPipe käsittely (erilliset landmarkerit + timestamp) ───────────
    rgb_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
    mp_img_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
    result_f = landmarker_front.detect_for_video(mp_img_f, timestamp_front)
    timestamp_front += 1

    result_s = None
    if mode == 2 and frame_s is not None:
        rgb_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
        mp_img_s = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_s)
        result_s = landmarker_side.detect_for_video(mp_img_s, timestamp_side)
        timestamp_side += 1

    # ── Piirrä & analysoi ───────────────────────────────────────────────────
    draw_landmarks(frame_f, result_f)

    status_f = "front: ?"
    color_f  = (180,180,180)
    warn_f   = ""

    if result_f.pose_landmarks:
        lm_f = result_f.pose_landmarks[0]

        if calibrating:
            if lm_f[7].presence > 0.5 and lm_f[11].presence > 0.5:
                ear_w = abs(lm_f[7].x - lm_f[8].x)
                d_y   = (lm_f[11].y - lm_f[7].y + lm_f[12].y - lm_f[8].y) / 2
                calib_deltas.append(d_y)
                calib_ear_widths.append(ear_w)

            cv2.putText(frame_f, f"KALIBROINTI {len(calib_deltas)}/60", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

            if len(calib_deltas) >= 60:
                ref_delta_y = np.mean(calib_deltas)
                ref_ear_width = np.mean(calib_ear_widths)
                calibrating = False
                print(f"Kalibrointi valmis → delta_y: {ref_delta_y:.4f}, ear_width: {ref_ear_width:.4f}")

        elif ref_delta_y is not None:
            status_f, color_f, warn_f = analyze_front_view(lm_f, ref_delta_y, ref_ear_width)

        else:
            cv2.putText(frame_f, "Paina C kalibroidaksesi etunäkymä", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # Sivunäkymä
    status_s = "side: ?"
    color_s  = (180,180,180)
    warn_s   = ""

    if result_s and result_s.pose_landmarks:
        draw_landmarks(frame_s, result_s)
        lm_s = result_s.pose_landmarks[0]
        status_s, color_s, warn_s = analyze_side_view(lm_s)

    # ── Näyttö ──────────────────────────────────────────────────────────────
    if mode == 1:
        vis = frame_f
    else:
        vis = np.hstack((frame_f, frame_s))

    cv2.putText(vis, f"Front: {status_f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_f, 3)
    if warn_f:
        cv2.putText(vis, warn_f, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    if mode == 2:
        offset = frame_f.shape[1]
        cv2.putText(vis, f"Side (DroidCam): {status_s}", (30+offset, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_s, 3)
        if warn_s:
            cv2.putText(vis, warn_s, (30+offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # FPS + ohjeet
    fps = 1 / (time.time() - prev_time + 1e-9)
    prev_time = time.time()
    cv2.putText(vis, f"FPS: {int(fps)}", (vis.shape[1]-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
    cv2.putText(vis, "C = kalibroi front | Q = quit", (30, vis.shape[0]-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)

    cv2.imshow("Posture Checker", vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrating = True
        ref_delta_y = None
        ref_ear_width = None
        calib_deltas.clear()
        calib_ear_widths.clear()
        print("Kalibrointi käynnistyi – istu suorassa edestä!")

# ── Lopetus ───────────────────────────────────────────────────────────────
cap_front.release()
if mode == 2 and cap_side is not None:
    cap_side.release()

landmarker_front.close()
if mode == 2 and landmarker_side is not None:
    landmarker_side.close()

cv2.destroyAllWindows()