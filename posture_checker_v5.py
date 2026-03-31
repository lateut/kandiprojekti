import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os

# Windows-aani (valinnainen)
try:
    import winsound
except ImportError:
    winsound = None

# ───────────────────────────────────────────────
# ASETUKSET
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
# Apufunktiot (jalkojen viivat poistettu)
# ───────────────────────────────────────────────

def draw_landmarks(image, detection_result):
    if not detection_result.pose_landmarks:
        return
    for landmark_list in detection_result.pose_landmarks:
        # Pisteet
        for lm in landmark_list:
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # Yhteydet – ILMAN jalkoja
        connections = [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),       # paa
            (9,10),
            (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
            (11,12),(11,23),(12,24),(23,24)                         # vartalo + olkapaat
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
        return "ei tarpeeksi nakyvyytta", (128,128,128), ""

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
        warning += "paa edessa "
        color = (0,0,255)
        status = "huono"

    return status, color, warning

def analyze_side_view(landmarks):
    key_points = [7, 8, 11, 12, 23, 24]
    if any(landmarks[i].presence < 0.6 for i in key_points):
        return "huono nakyvyys sivusta", (100,100,255), ""

    ear    = landmarks[8]
    should = landmarks[12]
    hip    = landmarks[24]

    v1 = np.array([should.x - ear.x, should.y - ear.y])
    v2 = np.array([hip.x - should.x, hip.y - should.y])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    if angle > 160:
        return "hyva profiili", (0,220,0), f"kulma ~{int(angle)}°"
    elif angle > 135:
        return "lieva eteentaivutus", (0,180,255), f"kulma ~{int(angle)}°"
    else:
        return "selva forward head!", (0,0,255), f"kulma {int(angle)}° – korjaa!"

# ───────────────────────────────────────────────
# Kameroiden valinta (ensimmainen ruutu)
# ───────────────────────────────────────────────

def select_mode():
    cv2.namedWindow("Posture Checker – Kameroiden valinta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker – Kameroiden valinta", 700, 400)

    font = cv2.FONT_HERSHEY_SIMPLEX
    mode = None

    while mode is None:
        img = np.zeros((400, 700, 3), np.uint8)
        cv2.putText(img, "Posture Checker", (80, 60), font, 1.3, (220, 220, 50), 3)
        cv2.putText(img, "Valitse kameroiden maara", (80, 110), font, 1.0, (255, 255, 255), 2)

        cv2.putText(img, "1 = Vain etukamera (webcam)", (80, 190), font, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "   -> Nykyinen yksikameratoteutus", (120, 220), font, 0.7, (180, 180, 255), 1)

        cv2.putText(img, "2 = Etukamera + sivukamera", (80, 270), font, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "   -> Sivukameraksi suositellaan DroidCam", (120, 300), font, 0.7, (180, 180, 255), 1)

        cv2.putText(img, "Paina 1 tai 2 -> Enter sulkee ikkunan", (80, 370), font, 0.75, (100, 255, 100), 1)

        cv2.imshow("Posture Checker – Kameroiden valinta", img)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'):
            mode = 1
        elif k == ord('2'):
            mode = 2
        elif k == 27:
            return None

    cv2.destroyWindow("Posture Checker – Kameroiden valinta")
    return mode

# ───────────────────────────────────────────────
# Paavalikko (Halytys / Seuranta / Ohjeet / Lopeta)
# ───────────────────────────────────────────────

def show_main_menu():
    cv2.namedWindow("Posture Checker – Paavalikko", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker – Paavalikko", 800, 600)

    font = cv2.FONT_HERSHEY_SIMPLEX
    choice = None

    while choice is None:
        img = np.zeros((600, 800, 3), np.uint8)
        
        cv2.putText(img, "POSTURE CHECKER", (120, 80), font, 1.8, (220, 220, 50), 4)
        cv2.putText(img, "Valitse toiminto", (220, 130), font, 1.0, (255, 255, 255), 2)

        cv2.putText(img, "1 -> Halytys (aani kun ryhti huononee)", (80, 220), font, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "2 -> Seuranta (laskee huonon ryhdin ajan)", (80, 270), font, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "3 -> Ohjeet", (80, 320), font, 1.0, (0, 255, 200), 2)
        cv2.putText(img, "4 -> Lopeta ohjelma", (80, 370), font, 1.0, (0, 100, 255), 2)

        cv2.putText(img, "Paina numero 1-4", (250, 520), font, 0.8, (180, 180, 255), 2)

        cv2.imshow("Posture Checker – Paavalikko", img)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'): choice = 1
        elif k == ord('2'): choice = 2
        elif k == ord('3'): choice = 3
        elif k == ord('4') or k == 27: choice = 4

    cv2.destroyWindow("Posture Checker – Paavalikko")
    return choice

# ───────────────────────────────────────────────
# Ohjeet
# ───────────────────────────────────────────────

def show_instructions():
    cv2.namedWindow("Posture Checker – Ohjeet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Checker – Ohjeet", 900, 600)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.zeros((600, 900, 3), np.uint8)

    lines = [
        "OHJEET",
        "",
        "• Istu kameraan nakyviin (etaisyys noin 50-80 cm)",
        "• Paina C -> kalibroi (istu suorassa noin 2 sekuntia)",
        "• Kalibroinnin jalkeen ohjelma tunnistaa ryhtisi",
        "",
        "Halytys-tila:",
        "   -> Pieni aanimerkki kun ryhti menee huonoksi",
        "",
        "Seuranta-tila:",
        "   -> Seuraa kuinka kauan ryhti on huonona",
        "   -> Paina B -> lopeta seuranta ja nae tilastot",
        "",
        "Yleiset nappaimet:",
        "   C = kalibroi",
        "   B = takaisin paavalikkoon",
        "   Q = lopeta koko ohjelma",
        "",
        "Paina mita tahansa nappainta palataksesi paavalikkoon"
    ]

    y = 60
    for line in lines:
        cv2.putText(img, line, (50, y), font, 0.85 if "•" in line or "Halytys" in line or "Seuranta" in line else 1.0,
                    (255, 255, 255), 2 if "OHJEET" in line else 1)
        y += 45

    cv2.imshow("Posture Checker – Ohjeet", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Posture Checker – Ohjeet")

# ───────────────────────────────────────────────
# HALYTYS-tila (toimii 1 tai 2 kameralla)
# ───────────────────────────────────────────────

def run_alert_mode(mode):
    TARGET_W, TARGET_H = 640, 360   # nopeuden takia

    cap_front = cv2.VideoCapture(0)
    cap_side = None
    landmarker_front = create_landmarker()
    landmarker_side = None

    if mode == 2:
        cap_side = cv2.VideoCapture(1)
        landmarker_side = create_landmarker()
        print("Käytetään kahta kameraa → etu (0) + sivu (1)")

    if not cap_front.isOpened():
        print("Etukamera ei aukea!")
        return

    if mode == 2 and (cap_side is None or not cap_side.isOpened()):
        print("Sivukamera ei aukea → jatketaan vain yhdella kameralla")
        mode = 1

    calibrating = False
    ref_delta_y = None
    ref_ear_width = None
    calib_deltas = []
    calib_ear_widths = []
    bad_counter = 0
    prev_time = time.time()
    last_beep = 0
    timestamp_f = 0
    timestamp_s = 0

    print("Halytys-tila kaynnistetty – paina B palataksesi valikkoon")

    while True:
        ret_f, frame_f = cap_front.read()
        if not ret_f:
            break
        frame_f = cv2.flip(frame_f, 1)
        frame_f = cv2.resize(frame_f, (TARGET_W, TARGET_H))

        frame_s = None
        if mode == 2 and cap_side:
            ret_s, frame_s = cap_side.read()
            if not ret_s:
                break
            frame_s = cv2.flip(frame_s, 1)
            frame_s = cv2.resize(frame_s, (TARGET_W, TARGET_H))

        # MediaPipe
        rgb_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        mp_img_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
        result_f = landmarker_front.detect_for_video(mp_img_f, timestamp_f)
        timestamp_f += 1

        result_s = None
        if mode == 2 and frame_s is not None:
            rgb_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
            mp_img_s = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_s)
            result_s = landmarker_side.detect_for_video(mp_img_s, timestamp_s)
            timestamp_s += 1

        # Analyysit
        status_f = "front: ?"
        color_f = (180,180,180)
        warn_f = ""
        is_bad_front = False

        if result_f.pose_landmarks:
            draw_landmarks(frame_f, result_f)
            lm_f = result_f.pose_landmarks[0]

            if calibrating:
                if lm_f[7].presence > 0.5 and lm_f[11].presence > 0.5:
                    ear_w = abs(lm_f[7].x - lm_f[8].x)
                    d_y = (lm_f[11].y - lm_f[7].y + lm_f[12].y - lm_f[8].y) / 2
                    calib_deltas.append(d_y)
                    calib_ear_widths.append(ear_w)
                cv2.putText(frame_f, f"KALIBROINTI {len(calib_deltas)}/60", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

                if len(calib_deltas) >= 60:
                    ref_delta_y = np.mean(calib_deltas)
                    ref_ear_width = np.mean(calib_ear_widths)
                    calibrating = False
                    print(f"Kalibrointi valmis → delta_y: {ref_delta_y:.4f}")

            elif ref_delta_y is not None:
                status_f, color_f, warn_f = analyze_front_view(lm_f, ref_delta_y, ref_ear_width)
                is_bad_front = (status_f == "huono")
            else:
                cv2.putText(frame_f, "Paina C kalibroidaksesi etunakyma", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # Sivunakyma
        status_s = "side: ?"
        color_s = (180,180,180)
        warn_s = ""
        is_bad_side = False

        if result_s and result_s.pose_landmarks:
            draw_landmarks(frame_s, result_s)
            lm_s = result_s.pose_landmarks[0]
            status_s, color_s, warn_s = analyze_side_view(lm_s)
            is_bad_side = ("forward" in status_s.lower() or "eteentaivutus" in status_s.lower() or "huono" in status_s.lower())

        is_bad_now = is_bad_front or is_bad_side

        # Aani (max 1x sekunnissa)
        if is_bad_now and winsound and time.time() - last_beep > 1.0:
            try:
                winsound.Beep(900, 180)
            except:
                pass
            last_beep = time.time()

        if is_bad_now:
            bad_counter += 1
        else:
            bad_counter = max(0, bad_counter - 1)

        # Naytto
        if mode == 1:
            vis = frame_f
        else:
            vis = np.hstack((frame_f, frame_s))

        cv2.putText(vis, f"Front: {status_f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_f, 3)
        if warn_f:
            cv2.putText(vis, warn_f, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if mode == 2:
            offset = TARGET_W
            cv2.putText(vis, f"Side: {status_s}", (30+offset, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_s, 3)
            if warn_s:
                cv2.putText(vis, warn_s, (30+offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if bad_counter > 35:
            cv2.putText(vis, "KORJAA RYHTI", (vis.shape[1]//2 - 220, vis.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 6)

        fps = 1 / (time.time() - prev_time + 1e-9)
        prev_time = time.time()
        cv2.putText(vis, f"FPS: {int(fps)}", (vis.shape[1]-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
        cv2.putText(vis, "C=kalibroi  B=valikkoon  Q=lopeta", (30, vis.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)

        cv2.imshow("Posture Checker – Halytys", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if ref_delta_y is None:
                calibrating = True
                calib_deltas.clear()
                calib_ear_widths.clear()
                print("Kalibrointi kaynnistyi – istu suorassa edesta!")
        elif key == ord('b'):
            break

    # Lopetus
    cap_front.release()
    if mode == 2 and cap_side:
        cap_side.release()
    landmarker_front.close()
    if mode == 2 and landmarker_side:
        landmarker_side.close()
    cv2.destroyAllWindows()

# ───────────────────────────────────────────────
# SEURANTA-tila (toimii 1 tai 2 kameralla)
# ───────────────────────────────────────────────

def run_track_mode(mode):
    TARGET_W, TARGET_H = 640, 360

    cap_front = cv2.VideoCapture(0)
    cap_side = None
    landmarker_front = create_landmarker()
    landmarker_side = None

    if mode == 2:
        cap_side = cv2.VideoCapture(1)
        landmarker_side = create_landmarker()
        print("Käytetään kahta kameraa → etu (0) + sivu (1)")

    if not cap_front.isOpened():
        print("Etukamera ei aukea!")
        return

    if mode == 2 and (cap_side is None or not cap_side.isOpened()):
        print("Sivukamera ei aukea → jatketaan vain yhdella kameralla")
        mode = 1

    calibrating = False
    ref_delta_y = None
    ref_ear_width = None
    calib_deltas = []
    calib_ear_widths = []
    bad_counter = 0
    prev_time = time.time()
    timestamp_f = 0
    timestamp_s = 0

    total_time = 0.0
    bad_time = 0.0

    print("Seuranta-tila kaynnistetty – paina B lopettaaksesi seurannan")

    while True:
        ret_f, frame_f = cap_front.read()
        if not ret_f:
            break
        frame_f = cv2.flip(frame_f, 1)
        frame_f = cv2.resize(frame_f, (TARGET_W, TARGET_H))

        frame_s = None
        if mode == 2 and cap_side:
            ret_s, frame_s = cap_side.read()
            if not ret_s:
                break
            frame_s = cv2.flip(frame_s, 1)
            frame_s = cv2.resize(frame_s, (TARGET_W, TARGET_H))

        # MediaPipe
        rgb_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        mp_img_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
        result_f = landmarker_front.detect_for_video(mp_img_f, timestamp_f)
        timestamp_f += 1

        result_s = None
        if mode == 2 and frame_s is not None:
            rgb_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
            mp_img_s = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_s)
            result_s = landmarker_side.detect_for_video(mp_img_s, timestamp_s)
            timestamp_s += 1

        # Analyysit
        status_f = "front: ?"
        color_f = (180,180,180)
        warn_f = ""
        is_bad_front = False

        if result_f.pose_landmarks:
            draw_landmarks(frame_f, result_f)
            lm_f = result_f.pose_landmarks[0]

            if calibrating:
                if lm_f[7].presence > 0.5 and lm_f[11].presence > 0.5:
                    ear_w = abs(lm_f[7].x - lm_f[8].x)
                    d_y = (lm_f[11].y - lm_f[7].y + lm_f[12].y - lm_f[8].y) / 2
                    calib_deltas.append(d_y)
                    calib_ear_widths.append(ear_w)
                cv2.putText(frame_f, f"KALIBROINTI {len(calib_deltas)}/60", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

                if len(calib_deltas) >= 60:
                    ref_delta_y = np.mean(calib_deltas)
                    ref_ear_width = np.mean(calib_ear_widths)
                    calibrating = False
                    print(f"Kalibrointi valmis → delta_y: {ref_delta_y:.4f}")

            elif ref_delta_y is not None:
                status_f, color_f, warn_f = analyze_front_view(lm_f, ref_delta_y, ref_ear_width)
                is_bad_front = (status_f == "huono")
            else:
                cv2.putText(frame_f, "Paina C kalibroidaksesi etunakyma", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        status_s = "side: ?"
        color_s = (180,180,180)
        warn_s = ""
        is_bad_side = False

        if result_s and result_s.pose_landmarks:
            draw_landmarks(frame_s, result_s)
            lm_s = result_s.pose_landmarks[0]
            status_s, color_s, warn_s = analyze_side_view(lm_s)
            is_bad_side = ("forward" in status_s.lower() or "eteentaivutus" in status_s.lower() or "huono" in status_s.lower())

        is_bad_now = is_bad_front or is_bad_side

        if is_bad_now:
            bad_counter += 1
        else:
            bad_counter = max(0, bad_counter - 1)

        # Aikatilastot
        delta = time.time() - prev_time
        prev_time = time.time()
        total_time += delta
        if is_bad_now:
            bad_time += delta

        # Naytto
        if mode == 1:
            vis = frame_f
        else:
            vis = np.hstack((frame_f, frame_s))

        cv2.putText(vis, f"Front: {status_f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_f, 3)
        if warn_f:
            cv2.putText(vis, warn_f, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if mode == 2:
            offset = TARGET_W
            cv2.putText(vis, f"Side: {status_s}", (30+offset, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_s, 3)
            if warn_s:
                cv2.putText(vis, warn_s, (30+offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if bad_counter > 35:
            cv2.putText(vis, "KORJAA RYHTI", (vis.shape[1]//2 - 220, vis.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 6)

        # Tilastot
        cv2.putText(vis, f"Seuranta: {int(total_time)} s", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,100), 2)
        percent = int(bad_time/total_time*100) if total_time > 0 else 0
        cv2.putText(vis, f"Huono ryhti: {int(bad_time)} s ({percent} %)", (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)

        fps = 1 / (time.time() - prev_time + 1e-9)
        cv2.putText(vis, f"FPS: {int(fps)}", (vis.shape[1]-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
        cv2.putText(vis, "C=kalibroi  B=lopeta seuranta  Q=lopeta", (30, vis.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)

        cv2.imshow("Posture Checker – Seuranta", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if ref_delta_y is None:
                calibrating = True
                calib_deltas.clear()
                calib_ear_widths.clear()
                print("Kalibrointi kaynnistyi – istu suorassa edesta!")
        elif key == ord('b'):
            break

    # Lopetus + tulokset
    cap_front.release()
    if mode == 2 and cap_side:
        cap_side.release()
    landmarker_front.close()
    if mode == 2 and landmarker_side:
        landmarker_side.close()
    cv2.destroyAllWindows()

    if total_time > 0:
        result_img = np.zeros((400, 700, 3), np.uint8)
        cv2.putText(result_img, "SEURANTA PAATTYI", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 200), 3)
        cv2.putText(result_img, f"Kokonaisaika: {int(total_time)} sekuntia", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        percent = int(bad_time/total_time*100)
        cv2.putText(result_img, f"Huono ryhti: {int(bad_time)} sekuntia ({percent} %)", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 255), 2)
        cv2.putText(result_img, "Paina mita tahansa nappainta palataksesi paavalikkoon", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 1)

        cv2.namedWindow("Seurannan tulokset", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Seurannan tulokset", 700, 400)
        cv2.imshow("Seurannan tulokset", result_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Seurannan tulokset")

# ───────────────────────────────────────────────
# Paaohjelma
# ───────────────────────────────────────────────

print("Posture Checker kaynnistyy...")

mode = select_mode()
if mode is None:
    print("Ei valintaa → lopetetaan")
    exit(0)

while True:
    choice = show_main_menu()
    
    if choice == 1:
        run_alert_mode(mode)
    elif choice == 2:
        run_track_mode(mode)
    elif choice == 3:
        show_instructions()
    elif choice == 4:
        break

print("Ohjelma sulkeutui. Hyvaa paivaa!")