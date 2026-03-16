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
    print("Lataa se täältä ja laita samaan kansioon:")
    print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    exit(1)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_segmentation_masks=False
)

landmarker = PoseLandmarker.create_from_options(options)

# ... loppu koodi pysyy samanlaisena ...

# ───────────────────────────────────────────────
# Apufunktiot
# ───────────────────────────────────────────────

def draw_landmarks(image, detection_result):
    if detection_result.pose_landmarks:
        for landmark_list in detection_result.pose_landmarks:
            for landmark in landmark_list:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
            # Piirrä yhteydet (POSE_CONNECTIONS ei enää suoraan saatavilla, joten manuaalinen lista)
            connections = [
                (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8),     # pää
                (9,10), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
                (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),
                (11,12), (11,23), (12,24), (23,24),                         # vartalo
                (23,25), (25,27), (27,29), (27,31), (25,27),                # vasen jalka
                (24,26), (26,28), (28,30), (28,32), (26,28)                 # oikea jalka
            ]
            for conn in connections:
                if conn[0] < len(landmark_list) and conn[1] < len(landmark_list):
                    p1 = landmark_list[conn[0]]
                    p2 = landmark_list[conn[1]]
                    x1, y1 = int(p1.x * image.shape[1]), int(p1.y * image.shape[0])
                    x2, y2 = int(p2.x * image.shape[1]), int(p2.y * image.shape[0])
                    cv2.line(image, (x1,y1), (x2,y2), (255, 255, 255), 2)


# ───────────────────────────────────────────────
# Pääsilmukka
# ───────────────────────────────────────────────

cap = cv2.VideoCapture(0)

calibrating = False
ref_delta_y = None
ref_ear_width = None
calib_deltas = []
calib_ear_widths = []
bad_counter = 0
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera ei toimi!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Skaalaa kuva parempaan kokoon
    frame = cv2.resize(frame, (1024, 576))
    h, w = 576, 1024

    # Muunna MediaPipe Imageksi
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Aikaleima millisekunteina (tärkeä VIDEO-tilassa)
    timestamp_ms = int(time.time() * 1000)

    # Suorita detektio
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    status_text = "ryhti ok"
    color = (0, 255, 0)
    warning = ""

    if detection_result.pose_landmarks:
        draw_landmarks(frame, detection_result)

        landmarks = detection_result.pose_landmarks[0]  # ensimmäinen henkilö

        # Tarkistetaan näkyvyys tärkeimmille pisteille
        if (landmarks[7].presence > 0.5 and landmarks[8].presence > 0.5 and
            landmarks[11].presence > 0.5 and landmarks[12].presence > 0.5):

            # Korvien välinen leveys (normalisoitu [0,1])
            ear_width_norm = abs(landmarks[7].x - landmarks[8].x)

            # Olkapäiden ja korvien keskim. pystyetäisyys
            left_delta  = landmarks[11].y - landmarks[7].y
            right_delta = landmarks[12].y - landmarks[8].y
            delta_y_norm = (left_delta + right_delta) / 2

            if calibrating:
                calib_deltas.append(delta_y_norm)
                calib_ear_widths.append(ear_width_norm)
                cv2.putText(frame, f"KALIBROIDAAN... {len(calib_deltas)}/60",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                if len(calib_deltas) >= 60:
                    ref_delta_y = np.mean(calib_deltas)
                    ref_ear_width = np.mean(calib_ear_widths)
                    calibrating = False
                    print("✅ Kalibrointi valmis!")

            elif ref_delta_y is not None:
                is_raised = delta_y_norm < 0.78 * ref_delta_y   # olkapäät koholla
                is_close  = ear_width_norm > 1.22 * ref_ear_width  # pää lähellä

                if is_raised or is_close:
                    bad_counter += 1
                    color = (0, 0, 255)
                    status_text = "HUONO RYHTI!"
                    # if is_raised: warning += "hartiat liian koholla! "
                    if is_close:  warning += "paea liian lahellae! "
                else:
                    bad_counter = max(0, bad_counter - 1)

    else:
        cv2.putText(frame, "Istu kameraan näkyviin", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if ref_delta_y is None and not calibrating:
        cv2.putText(frame, "Paina 'C' aloittaaksesi kalibroinnin", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Näytä tila
    cv2.putText(frame, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    if warning and bad_counter <= 35:
        cv2.putText(frame, warning, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if bad_counter > 35:  # ~1–1.5 s huonoa ryhtiä
        cv2.putText(frame, "korjaa ryhti", (w//2 - 280, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)

    # Näytä takaisin-nappi
    # cv2.putText(frame, "Takaisin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS + ohjeet
    fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "C = kalibroi | Q = lopeta | B = takaisin", (50, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.namedWindow('Posture Checker – MediaPipe Tasks', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Posture Checker – MediaPipe Tasks', 1024, 576)
    cv2.imshow('Posture Checker – MediaPipe Tasks', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrating = True
        calib_deltas.clear()
        calib_ear_widths.clear()
        print("Kalibrointi aloitettu – istu hyvässä ryhdissä ~2 sekuntia!")
    elif key == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()