import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ==========================================
# VAKIOT JA KONFIGURAATIO
# ==========================================
DEFAULT_MODEL = 'pose_landmarker_lite.task'

# Varoituskynnykset
SHOULDER_HEIGHT_THRESHOLD = 0.02
SPINE_SLOPE_THRESHOLD = 1.0
SHOULDER_ROTATION_THRESHOLD = 20  # asteet
SHOULDER_FORWARD_THRESHOLD = -30  # Olkapäiden eteenpäin kääntymisen kynnys (negatiivinen = eteenpäin)
HEAD_TILT_THRESHOLD = 0.03  # Korvien y-koordinaattien ero
HEAD_FORWARD_THRESHOLD = 0.75  # Pään eteenpäin kääntymisen kynnys (sivunäkymä)

# Värit (BGR)
COLOR_SPINE = (0, 255, 0)
COLOR_HEAD_FORWARD = (0, 0, 255)
COLOR_HEAD_TILT = (255, 0, 0)
COLOR_SHOULDER_LINE = (0, 255, 255)
COLOR_VERTICAL = (200, 200, 200)
COLOR_ROTATION_GOOD = (0, 255, 0)
COLOR_ROTATION_BAD = (0, 0, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_VERTICAL_LINE_THICKNESS = 1

# Viivapaksuudet
LINE_THICK = 3
LINE_MEDIUM = 2
ARROW_TIP_LENGTH = 0.3

# Teksti
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.8
TEXT_THICK = 2
TEXT_Y_START = 30
TEXT_Y_STEP = 30
TEXT_X_START = 10

# ==========================================
# DETEKTORIN LUONTI
# ==========================================
def create_pose_detector(model_path=DEFAULT_MODEL):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

# ==========================================
# APUFUNKTIOT
# ==========================================
def get_landmark(landmarks, landmark_type):
    """Hakee yksittäisen landmarkin annetuista landmarks-datasta."""
    return landmarks[landmark_type]

def pixel_coords(point, w, h):
    """Muuntaa normalisoidut koordinaatit pikselikoordinaatiksi."""
    return (int(point[0] * w), int(point[1] * h))

def midpoint_normalized(point1, point2):
    """Laskee kahden pisteen keskipisteen normalisoiduissa koordinaateissa."""
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def extract_pose_data(detection_result):
    """Poimii kaikki tarvittavat landmarkit detektiotuloksesta."""
    if not detection_result.pose_landmarks:
        return None
    
    lm = detection_result.pose_landmarks[0]
    return {
        'left_shoulder': get_landmark(lm, vision.PoseLandmark.LEFT_SHOULDER),
        'right_shoulder': get_landmark(lm, vision.PoseLandmark.RIGHT_SHOULDER),
        'left_hip': get_landmark(lm, vision.PoseLandmark.LEFT_HIP),
        'right_hip': get_landmark(lm, vision.PoseLandmark.RIGHT_HIP),
        'left_ear': get_landmark(lm, vision.PoseLandmark.LEFT_EAR),
        'right_ear': get_landmark(lm, vision.PoseLandmark.RIGHT_EAR),
        'left_elbow': get_landmark(lm, vision.PoseLandmark.LEFT_ELBOW),
        'right_elbow': get_landmark(lm, vision.PoseLandmark.RIGHT_ELBOW),
    }

def calculate_rotation_angle(shoulder, elbow, hip):
    """
    Laskee olkapään rotaatiokulman lantion suuntaan nähden.
    Vertailee hartia-kyynärpää vektoria hartia-lantio vektoriin.
    Positiivinen = kiertynyt taaksepäin (hyvä), negatiivinen = kiertynyt eteenpäin (huono).
    """
    # Referenssivektori: hartia → lantio (selkärangan suunta)
    ref_dx = hip.x - shoulder.x
    ref_dy = hip.y - shoulder.y
    
    # Mittausvektori: hartia → kyynärpää
    arm_dx = elbow.x - shoulder.x
    arm_dy = elbow.y - shoulder.y
    
    # Laske vektorien välinen kulma
    # Käytä atan2 näiden välille
    ref_angle = np.arctan2(ref_dx, ref_dy)
    arm_angle = np.arctan2(arm_dx, arm_dy)
    
    # Kulman ero radiaaneissa, muuta asteiksi
    angle_diff_rad = arm_angle - ref_angle
    angle_deg = np.degrees(angle_diff_rad)
    
    # Normalisoi välille -180 to 180
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    
    return angle_deg

# ==========================================
# RYHTIAKSELIEN PIIRTO
# ==========================================
def draw_posture_axes(frame, detection_result, w, h, is_side_view=False):
    """Piirtää analyysilinjat ja sivukameralla hartioiden eteenpäin nuolen."""
    pose_data = extract_pose_data(detection_result)
    if not pose_data:
        return

    # Lasketaan keskipisteet
    mid_shoulder = midpoint_normalized(
        (pose_data['left_shoulder'].x, pose_data['left_shoulder'].y),
        (pose_data['right_shoulder'].x, pose_data['right_shoulder'].y)
    )
    mid_hip = midpoint_normalized(
        (pose_data['left_hip'].x, pose_data['left_hip'].y),
        (pose_data['right_hip'].x, pose_data['right_hip'].y)
    )
    mid_ear = midpoint_normalized(
        (pose_data['left_ear'].x, pose_data['left_ear'].y),
        (pose_data['right_ear'].x, pose_data['right_ear'].y)
    )

    # Muunnetaan pikselioordinaatiksi
    mid_sh_px = pixel_coords(mid_shoulder, w, h)
    mid_hip_px = pixel_coords(mid_hip, w, h)
    mid_ear_px = pixel_coords(mid_ear, w, h)
    left_sh_px = pixel_coords((pose_data['left_shoulder'].x, pose_data['left_shoulder'].y), w, h)
    right_sh_px = pixel_coords((pose_data['right_shoulder'].x, pose_data['right_shoulder'].y), w, h)
    left_ear_px = pixel_coords((pose_data['left_ear'].x, pose_data['left_ear'].y), w, h)
    right_ear_px = pixel_coords((pose_data['right_ear'].x, pose_data['right_ear'].y), w, h)

    # ==========================================
    # PIIRRETTÄVÄT AKSELIT
    # ==========================================
    cv2.line(frame, mid_hip_px, mid_sh_px, COLOR_SPINE, LINE_THICK)
    cv2.line(frame, mid_sh_px, mid_ear_px, COLOR_HEAD_FORWARD, LINE_MEDIUM)
    cv2.line(frame, left_ear_px, right_ear_px, COLOR_HEAD_TILT, LINE_MEDIUM)
    cv2.line(frame, left_sh_px, right_sh_px, COLOR_SHOULDER_LINE, LINE_MEDIUM)
    
    vertical_line_end = (mid_sh_px[0], h)
    cv2.line(frame, mid_sh_px, vertical_line_end, COLOR_VERTICAL, COLOR_VERTICAL_LINE_THICKNESS)

    # ==========================================
    # Sivukamera: hartioiden rotaation nuoli
    # ==========================================
    if is_side_view:
        # Laske rotaatiokulmat molemmille puolille verrattuna lantioon
        left_rotation = calculate_rotation_angle(
            pose_data['left_shoulder'], 
            pose_data['left_elbow'],
            pose_data['left_hip']
        )
        right_rotation = calculate_rotation_angle(
            pose_data['right_shoulder'], 
            pose_data['right_elbow'],
            pose_data['right_hip']
        )
        
        # Keskiarvo
        avg_rotation = (left_rotation + right_rotation) / 2
        
        # Piirrä nuoli rotaation perusteella
        # Positiivinen = taaksepäin (hyvä), negatiivinen = eteenpäin (huono)
        arrow_length = int(np.clip(avg_rotation, -50, 50) * 1.5)
        start_point = mid_sh_px
        end_point = (mid_sh_px[0] + arrow_length, mid_sh_px[1])
        
        # Väri: vihreä jos taaksepäin (>5°), punainen jos eteenpäin (<-5°)
        if avg_rotation > 5:
            color = COLOR_ROTATION_GOOD
        elif avg_rotation < -5:
            color = COLOR_ROTATION_BAD
        else:
            color = (255, 150, 0)  # Oranssi = neutraali
        
        cv2.arrowedLine(frame, start_point, end_point, color, LINE_MEDIUM, tipLength=ARROW_TIP_LENGTH)
        
        # Näytä kulma numeroina
        angle_text = f"Rot: {avg_rotation:.1f}°"
        cv2.putText(frame, angle_text, (mid_sh_px[0] + 20, mid_sh_px[1] - 20),
                    TEXT_FONT, 0.6, color, TEXT_THICK)


# ==========================================
# PÄÄN KALLISTUKSEN ILMOITUS
# ==========================================
def draw_head_tilt_warning(frame, detection_result, w, h):
    """Näyttää ilmoituksen pään sivuttaiskallistuksesta."""
    pose_data = extract_pose_data(detection_result)
    if not pose_data:
        return
    
    # Laske korvien y-koordinaattien ero
    head_tilt = abs(pose_data['left_ear'].y - pose_data['right_ear'].y)
    
    if head_tilt > HEAD_TILT_THRESHOLD:
        message = "Pää kallistunut sivulle!"
        cv2.putText(frame, message, (TEXT_X_START, TEXT_Y_START),
                    TEXT_FONT, TEXT_SIZE, COLOR_ERROR, TEXT_THICK)


# ==========================================
# HARTIOIDEN TASAISUUDEN ILMOITUS
# ==========================================
def draw_shoulder_level_warning(frame, detection_result, w, h):
    """Näyttää ilmoituksen nousseista hartioista."""
    pose_data = extract_pose_data(detection_result)
    if not pose_data:
        return
    
    left_shoulder_y = pose_data['left_shoulder'].y
    right_shoulder_y = pose_data['right_shoulder'].y
    left_hip_y = pose_data['left_hip'].y
    right_hip_y = pose_data['right_hip'].y
    
    # Laske keskimääräinen hartioiden ja lantion y-koordinaatti
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    
    # Tarkista hartioiden korkeusero
    shoulder_height_diff = abs(left_shoulder_y - right_shoulder_y)
    
    if shoulder_height_diff > SHOULDER_HEIGHT_THRESHOLD:
        if left_shoulder_y < right_shoulder_y:
            message = "Vasen hartia noussut!"
        else:
            message = "Oikea hartia noussut!"
        
        y_pos = TEXT_Y_START + TEXT_Y_STEP
        cv2.putText(frame, message, (TEXT_X_START, y_pos),
                    TEXT_FONT, TEXT_SIZE, COLOR_ERROR, TEXT_THICK)


# ==========================================
# PÄÄN ETEENPÄIN KÄÄNTYMISEN ILMOITUS (SIVUNÄKYMÄ)
# ==========================================
def draw_head_forward_warning(frame, detection_result, w, h):
    """Näyttää ilmoituksen, jos pää on kääntynyt eteenpäin (sivunäkymä)."""
    pose_data = extract_pose_data(detection_result)
    if not pose_data:
        return
    
    # Laske keskipisteet
    mid_shoulder = midpoint_normalized(
        (pose_data['left_shoulder'].x, pose_data['left_shoulder'].y),
        (pose_data['right_shoulder'].x, pose_data['right_shoulder'].y)
    )
    mid_ear = midpoint_normalized(
        (pose_data['left_ear'].x, pose_data['left_ear'].y),
        (pose_data['right_ear'].x, pose_data['right_ear'].y)
    )
    
    # Laske pään eteenpäin vektori (hartioilta korviin)
    head_forward_dx = mid_ear[0] - mid_shoulder[0]
    head_forward_dy = mid_ear[1] - mid_shoulder[1]
    
    # Jos hartia on vasemmalla ja korva on oikealla (positiivinen dx), pää on kääntynyt eteenpäin
    # Laske pään etäisyys vertikaalista
    if head_forward_dy != 0:
        head_forward_angle = abs(head_forward_dx) / abs(head_forward_dy)
        
        if head_forward_angle > HEAD_FORWARD_THRESHOLD:
            message = "Pää kääntynyt eteenpäin!"
            y_pos = TEXT_Y_START
            cv2.putText(frame, message, (TEXT_X_START, y_pos),
                        TEXT_FONT, TEXT_SIZE, COLOR_ERROR, TEXT_THICK)


# ==========================================
# OLKAPÄIDEN ETEENPÄIN KÄÄNTYMISEN ILMOITUS (SIVUNÄKYMÄ)
# ==========================================
def draw_shoulder_forward_warning(frame, detection_result, w, h):
    """Näyttää ilmoituksen, jos olkapäät ovat kääntyneet eteenpäin (sivunäkymä)."""
    pose_data = extract_pose_data(detection_result)
    if not pose_data:
        return
    
    # Laske rotaatiokulmat (sama kuin draw_posture_axes:issa)
    left_rotation = calculate_rotation_angle(
        pose_data['left_shoulder'], 
        pose_data['left_elbow'],
        pose_data['left_hip']
    )
    right_rotation = calculate_rotation_angle(
        pose_data['right_shoulder'], 
        pose_data['right_elbow'],
        pose_data['right_hip']
    )
    
    # Keskiarvo
    avg_rotation = (left_rotation + right_rotation) / 2
    
    # Jos olkapäät ovat liian paljon eteenpäin (negatiivinen = eteenpäin)
    if avg_rotation < SHOULDER_FORWARD_THRESHOLD:
        message = "Olkapäät eteenpäin!"
        y_pos = TEXT_Y_START + TEXT_Y_STEP
        cv2.putText(frame, message, (TEXT_X_START, y_pos),
                    TEXT_FONT, TEXT_SIZE, COLOR_ERROR, TEXT_THICK)


def combine_frames(frame1, frame2):
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape
    new_h = max(h1, h2)
    new_w = w1 + w2
    combined = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    combined[0:h1, 0:w1] = frame1
    combined[0:h2, w1:w1+w2] = frame2
    return combined

def process_camera_frame(frame, detector, timestamp, is_side_view=False):
    """Käsittelee yksittäisen kamerakehyksen."""
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp)
    
    annotated_frame = frame.copy()
    draw_posture_axes(annotated_frame, result, w, h, is_side_view=is_side_view)
    
    # Näytä ilmoitukset etukameralla
    if not is_side_view:
        draw_head_tilt_warning(annotated_frame, result, w, h)
        draw_shoulder_level_warning(annotated_frame, result, w, h)
    # Näytä ilmoitukset sivukameralla
    else:
        draw_head_forward_warning(annotated_frame, result, w, h)
        draw_shoulder_forward_warning(annotated_frame, result, w, h)
    
    return annotated_frame, result

# ==========================================
# PÄÄLOOPPI
# ==========================================
def main():
    cap_front = cv2.VideoCapture(0)
    cap_side = cv2.VideoCapture(1)

    detector_front = create_pose_detector()
    detector_side = create_pose_detector()

    timestamp_front = 0
    timestamp_side = 0

    while cap_front.isOpened() and cap_side.isOpened():
        ret_f, frame_front = cap_front.read()
        ret_s, frame_side = cap_side.read()
        if not ret_f or not ret_s:
            break

        # Käsitellään molemmat kamerat
        annotated_front, _ = process_camera_frame(
            frame_front, detector_front, timestamp_front, is_side_view=False
        )
        annotated_side, _ = process_camera_frame(
            frame_side, detector_side, timestamp_side, is_side_view=True
        )
        
        timestamp_front += 1
        timestamp_side += 1

        # Yhdistetään ja näytetään
        combined = combine_frames(annotated_front, annotated_side)
        cv2.imshow("Posture Analysis (EdgeAI)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Posture Analysis (EdgeAI)", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap_front.release()
    cap_side.release()
    cv2.destroyAllWindows()


# ==========================================
# KÄYNNISTYS
# ==========================================
if __name__ == "__main__":
    main()