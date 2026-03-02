import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np

# ==========================================
# FUNKTIOT
# ==========================================

def create_pose_detector(model_path='pose_landmarker_heavy.task'):
    """Luo ja palauttaa MediaPipe PoseLandmarker -detektorin."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

def draw_landmarks(rgb_image, detection_result):
    """Piirtää kaikki MediaPipe landmarkit kuvaan."""
    annotated_image = np.copy(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks

    landmark_style = drawing_styles.get_default_pose_landmarks_style()
    connection_style = drawing_utils.DrawingSpec(color=(0,255,0), thickness=2)

    if pose_landmarks_list:
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
    return annotated_image

def draw_spine_and_head_lines(frame, detection_result, w, h):
    """Piirtää selkärangan ja pään linjan sivukameraan."""
    if detection_result.pose_landmarks:
        lm = detection_result.pose_landmarks[0]

        # Hartiat ja lantio
        left_sh = lm[vision.PoseLandmark.LEFT_SHOULDER]
        right_sh = lm[vision.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[vision.PoseLandmark.LEFT_HIP]
        right_hip = lm[vision.PoseLandmark.RIGHT_HIP]

        mid_sh = ((left_sh.x + right_sh.x)/2, (left_sh.y + right_sh.y)/2)
        mid_hip = ((left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2)

        mid_sh_px = (int(mid_sh[0]*w), int(mid_sh[1]*h))
        mid_hp_px = (int(mid_hip[0]*w), int(mid_hip[1]*h))

        # Selkäranka (vihreä)
        cv2.line(frame, mid_hp_px, mid_sh_px, (0,255,0), 3)

        # Pään linja: hartioiden keskikohta → korvien keskipiste (punainen)
        left_ear = lm[vision.PoseLandmark.LEFT_EAR]
        right_ear = lm[vision.PoseLandmark.RIGHT_EAR]
        mid_ear = ((left_ear.x + right_ear.x)/2, (left_ear.y + right_ear.y)/2)
        mid_ear_px = (int(mid_ear[0]*w), int(mid_ear[1]*h))

        cv2.line(frame, mid_ear_px, mid_sh_px, (0,0,255), 2)

def combine_frames(frame1, frame2):
    """Yhdistää kaksi kuvaa vierekkäin."""
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape
    new_h = max(h1, h2)
    new_w = w1 + w2
    combined = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    combined[0:h1, 0:w1] = frame1
    combined[0:h2, w1:w1+w2] = frame2
    return combined

def main():
    """Päälooppi, joka pyörittää front- ja side-kameroita."""
    # Kamerat
    cap_front = cv2.VideoCapture(0)
    cap_side  = cv2.VideoCapture(1)

    # Detektorit
    detector_front = create_pose_detector()
    detector_side  = create_pose_detector()

    timestamp_front = 0
    timestamp_side  = 0

    while cap_front.isOpened() and cap_side.isOpened():
        ret_f, frame_front = cap_front.read()
        ret_s, frame_side  = cap_side.read()
        if not ret_f or not ret_s:
            break

        h1, w1, _ = frame_front.shape
        h2, w2, _ = frame_side.shape

        # ---------- Front camera ----------
        rgb_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        mp_img_front = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_front)
        result_front = detector_front.detect_for_video(mp_img_front, timestamp_front)
        timestamp_front += 1
        annotated_front = cv2.cvtColor(draw_landmarks(rgb_front, result_front), cv2.COLOR_RGB2BGR)

        # ---------- Side camera ----------
        rgb_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
        mp_img_side = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_side)
        result_side = detector_side.detect_for_video(mp_img_side, timestamp_side)
        timestamp_side += 1
        annotated_side = cv2.cvtColor(draw_landmarks(rgb_side, result_side), cv2.COLOR_RGB2BGR)

        # Piirrä selkäranka ja pää sivukameraan
        draw_spine_and_head_lines(annotated_side, result_side, w2, h2)

        # Yhdistä kuvat
        combined = combine_frames(annotated_front, annotated_side)

        cv2.imshow("Pose Landmarks + Spine/Head Lines", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Pose Landmarks + Spine/Head Lines", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap_front.release()
    cap_side.release()
    cv2.destroyAllWindows()


# ==========================================
# KÄYNNISTYS
# ==========================================
if __name__ == "__main__":
    main()
