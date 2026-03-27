import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Solutions (Legacy API for stability)
mp_hands = mp.solutions.hands
mp_pose_solutions = mp.solutions.pose

# MediaPipe Tasks (New API for multi-person pose)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

def get_distance(p1, p2):
    """Calculates Euclidean distance between two points (normalized)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_face_bbox_from_pose(landmarks):
    """
    Estimates a face bounding box from pose landmarks (nose, eyes, ears).
    Returns (xmin, ymin, width, height) in normalized coordinates.
    """
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    xs = [landmarks[i].x for i in face_indices]
    ys = [landmarks[i].y for i in face_indices]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    width = xmax - xmin
    height = ymax - ymin
    
    padding_x = width * 0.5
    padding_y = height * 1.5 
    
    return (
        max(0, xmin - padding_x/2),
        max(0, ymin - padding_y/2),
        min(1, width + padding_x),
        min(1, height + padding_y)
    )

def get_chest_bbox_from_pose(landmarks):
    """
    Estimates a chest bounding box from pose landmarks (shoulders, hips).
    Returns (xmin, ymin, width, height) in normalized coordinates.
    """
    # Shoulder indices: 11 (Left), 12 (Right)
    # Hip indices: 23 (Left), 24 (Right)
    shoulder_l = landmarks[11]
    shoulder_r = landmarks[12]
    hip_l = landmarks[23]
    hip_r = landmarks[24]
    
    # Calculate bounds of the upper torso
    xmin = min(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    xmax = max(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    ymin = min(shoulder_l.y, shoulder_r.y)
    ymax = max(hip_l.y, hip_r.y)
    
    # Restrict chest height to upper half of the torso for "chest" specifically
    torso_height = ymax - ymin
    chest_ymax = ymin + (torso_height * 0.6) # Only upper 60% of torso
    
    return (
        xmin,
        ymin,
        xmax - xmin,
        chest_ymax - ymin
    )

def is_point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def main():
    # 1. Initialize Hand Tracking (Solutions API)
    hands_detector = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=4
    )

    # 2. Initialize Pose Landmarker (Tasks API for Multi-Person)
    with open('pose_landmarker.task', 'rb') as f:
        model_data = f.read()

    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )
    pose_landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    PERSON_COLORS = [
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255)  # Magenta
    ]

    print("Multi-Person Face Touch Detection Started.")
    print("Press 'q' to quit.")

    tracked_bodies = []

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        pose_result = pose_landmarker.detect(mp_image)
        hand_result = hands_detector.process(rgb_image)

        current_poses = []
        if pose_result.pose_landmarks:
            for lms in pose_result.pose_landmarks:
                cx = sum([lm.x for lm in lms]) / len(lms)
                cy = sum([lm.y for lm in lms]) / len(lms)
                current_poses.append({
                    'center': (cx, cy),
                    'landmarks': lms,
                    'matched': False
                })

        for tb in tracked_bodies:
            best_dist = 0.3
            best_pose = None
            for cp in current_poses:
                if cp['matched']: continue
                d = get_distance(tb['pose_center'], cp['center'])
                if d < best_dist:
                    best_dist = d
                    best_pose = cp
            if best_pose:
                best_pose['matched'] = True
                tb['pose_center'] = best_pose['center']
                tb['landmarks'] = best_pose['landmarks']
                tb['missing_frames'] = 0
            else:
                tb['missing_frames'] += 1

        tracked_bodies = [tb for tb in tracked_bodies if tb['missing_frames'] < 10]

        for cp in current_poses:
            if not cp['matched']:
                tb_id = 'A'
                if any(tb['id'] == 'A' for tb in tracked_bodies):
                    tb_id = 'B'
                if any(tb['id'] == tb_id for tb in tracked_bodies):
                    continue
                tracked_bodies.append({
                    'id': tb_id,
                    'pose_center': cp['center'],
                    'missing_frames': 0,
                    'landmarks': cp['landmarks']
                })

        for tb in tracked_bodies:
            if tb['missing_frames'] > 0: continue
            landmarks = tb['landmarks']
            
            color = (0, 200, 0) if tb['id'] == 'A' else (0, 100, 255) # Green for A, Orange for B

            face_bbox = get_face_bbox_from_pose(landmarks)
            chest_bbox = get_chest_bbox_from_pose(landmarks)
            left_wrist = (landmarks[15].x, landmarks[15].y)
            right_wrist = (landmarks[16].x, landmarks[16].y)
            tb['face_bbox'] = face_bbox
            tb['chest_bbox'] = chest_bbox
            tb['wrists'] = [left_wrist, right_wrist]
            
            for connection in mp_pose_solutions.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if landmarks[start_idx].presence > 0.5 and landmarks[end_idx].presence > 0.5:
                    start_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    end_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(image, start_pt, end_pt, color, 2)
            
            fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
            cv2.rectangle(image, (fox, foy), (fox+fow, foy+foh), color, 2)
            cx, cy, cw_rect, ch_rect = int(chest_bbox[0]*w), int(chest_bbox[1]*h), int(chest_bbox[2]*w), int(chest_bbox[3]*h)
            cv2.rectangle(image, (cx, cy), (cx+cw_rect, cy+ch_rect), color, 1, cv2.LINE_4)
            cv2.putText(image, f"Subject {tb['id']}", (fox, foy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        alerts = [] 
        
        subject_a = next((tb for tb in tracked_bodies if tb['id'] == 'A' and tb['missing_frames'] == 0), None)
        subject_b = next((tb for tb in tracked_bodies if tb['id'] == 'B' and tb['missing_frames'] == 0), None)

        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                hand_wrist_normalized = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
                color = (0, 200, 0) # Subject A's color
                
                if subject_a:
                    bw15 = subject_a['wrists'][0]
                    bw16 = subject_a['wrists'][1]
                    d15 = get_distance(hand_wrist_normalized, bw15)
                    d16 = get_distance(hand_wrist_normalized, bw16)
                    closest_w = bw15 if d15 < d16 else bw16
                    cv2.line(image, (int(hand_wrist_normalized[0]*w), int(hand_wrist_normalized[1]*h)), (int(closest_w[0]*w), int(closest_w[1]*h)), color, 2)

                for lm in hand_landmarks.landmark:
                    lx, ly = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (lx, ly), 4, color, -1)
                
                if subject_b:
                    pt = hand_wrist_normalized
                    if is_point_in_rect(pt, subject_b['face_bbox']):
                        alerts.append(("A", "B", "Face"))
                    elif is_point_in_rect(pt, subject_b['chest_bbox']):
                        alerts.append(("A", "B", "Chest"))

        if alerts:
            # unique_alerts = set(alerts)
            alert_text = "ALERTS: " + ", ".join([f"P{b} touches P{a}'s {area}!" for a, b, area in set(alerts)])
            cv2.putText(image, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

        cv2.imshow('Multi-Person Face Touch Detection', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    pose_landmarker.close()

if __name__ == "__main__":
    main()
