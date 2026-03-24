import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os
import simple


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


# Update the side view processing to include posture analysis logic and warnings

def process_side_view(frame, detector, timestamp):
    """Process the side view frame with posture analysis logic."""
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp)

    annotated_frame = frame.copy()
    simple.draw_posture_axes(annotated_frame, result, w, h, is_side_view=True)
    simple.draw_head_forward_warning(annotated_frame, result, w, h)
    simple.draw_shoulder_forward_warning(annotated_frame, result, w, h)
    return annotated_frame, result


# ───────────────────────────────────────────────
# Pääsilmukka
# ───────────────────────────────────────────────

"""
#Yksi kamera

cap_front = cv2.VideoCapture(0)  # Etukamera

detector_front = PoseLandmarker.create_from_options(options)

timestamp_front = 0

while cap_front.isOpened() :
    ret_f, frame_front = cap_front.read()

    if not ret_f:
        print("Kamera ei toimi! Tarkista yhteydet.")
        break

    # Process the front camera frame
    frame_front = cv2.flip(frame_front, 1)
    rgb_frame_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
    mp_image_front = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_front)
    detection_result_front = detector_front.detect_for_video(mp_image_front, timestamp_front)
    timestamp_front += 1

    draw_landmarks(frame_front, detection_result_front)


    # Display the combined view
    cv2.namedWindow('Posture Checker – Dual View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Posture Checker – Dual View', 1280, 720)
    cv2.imshow('Posture Checker – Dual View', frame_front)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap_front.release()
cv2.destroyAllWindows()
landmarker.close()
"""

cap_front = cv2.VideoCapture(0)  # Etukamera
cap_side = cv2.VideoCapture(1)   # Sivukamera

detector_front = PoseLandmarker.create_from_options(options)
detector_side = PoseLandmarker.create_from_options(options)

timestamp_front = 0
timestamp_side = 0

while cap_front.isOpened() and cap_side.isOpened():
    ret_f, frame_front = cap_front.read()
    ret_s, frame_side = cap_side.read()

    if not ret_f or not ret_s:
        print("Kamera ei toimi! Tarkista yhteydet.")
        break

    # Process the front camera frame
    frame_front = cv2.flip(frame_front, 1)
    rgb_frame_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
    mp_image_front = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_front)
    detection_result_front = detector_front.detect_for_video(mp_image_front, timestamp_front)
    timestamp_front += 1

    draw_landmarks(frame_front, detection_result_front)

    # Process the side camera frame with posture analysis
    frame_side = cv2.flip(frame_side, 1)
    annotated_side, _ = process_side_view(frame_side, detector_side, timestamp_side)
    timestamp_side += 1

    # Combine the frames side by side
    combined_frame = np.hstack((frame_front, annotated_side))

    # Display the combined view
    cv2.namedWindow('Posture Checker – Dual View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Posture Checker – Dual View', 1280, 720)
    cv2.imshow('Posture Checker – Dual View', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap_front.release()
cap_side.release()
cv2.destroyAllWindows()
landmarker.close()
