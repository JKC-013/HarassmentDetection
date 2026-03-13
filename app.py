import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# Page Config
st.set_page_config(page_title="Multi-Person Pose Detection", layout="wide")

st.title("🏃 Multi-Person Pose Detection")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press the **"Start"** button below to begin.
3. Once started, the webcam feed will appear.
4. Alerts will show when one person's hand touches another person's face or chest.
5. **Press 'q' to stop** (or use the Stop button in the sidebar/widget).
""")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }
)

# Pose connections (standard indices for MediaPipe Pose)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
    (9, 10), # Mouth
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders to Wrists
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (24, 26), (26, 28), # Hips to Ankles
    (15, 17), (15, 19), (15, 21), (17, 19), # Left Hand
    (16, 18), (16, 20), (16, 22), (18, 20)  # Right Hand
]

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_face_bbox_from_pose(landmarks):
    # Landmarks 0-8 are face related
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    xs = [landmarks[i].x for i in face_indices]
    ys = [landmarks[i].y for i in face_indices]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    padding_x = width * 0.5
    padding_y = height * 1.5 
    return (max(0, xmin - padding_x/2), max(0, ymin - padding_y/2), min(1, width + padding_x), min(1, height + padding_y))

def get_chest_bbox_from_pose(landmarks):
    # Shoulders (11, 12) and Hips (23, 24)
    shoulder_l, shoulder_r = landmarks[11], landmarks[12]
    hip_l, hip_r = landmarks[23], landmarks[24]
    xmin = min(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    xmax = max(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    ymin = min(shoulder_l.y, shoulder_r.y)
    ymax = max(hip_l.y, hip_r.y)
    torso_height = ymax - ymin
    chest_ymax = ymin + (torso_height * 0.6) 
    return (xmin, ymin, xmax - xmin, chest_ymax - ymin)

def is_point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize Pose Landmarker
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
        self.pose_landmarker = PoseLandmarker.create_from_options(options)
        self.PERSON_COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        pose_result = self.pose_landmarker.detect(mp_image)

        bodies = []
        if pose_result.pose_landmarks:
            for i, landmarks in enumerate(pose_result.pose_landmarks):
                color = self.PERSON_COLORS[i % len(self.PERSON_COLORS)]
                face_bbox = get_face_bbox_from_pose(landmarks)
                chest_bbox = get_chest_bbox_from_pose(landmarks)
                
                # Hand points (index fingers and pinkies)
                hand_points = [
                    (landmarks[19].x, landmarks[19].y), (landmarks[20].x, landmarks[20].y),
                    (landmarks[17].x, landmarks[17].y), (landmarks[18].x, landmarks[18].y)
                ]
                
                bodies.append({
                    'id': i,
                    'face_bbox': face_bbox,
                    'chest_bbox': chest_bbox,
                    'hand_points': hand_points,
                    'color': color
                })
                
                # Draw connections
                for connection in POSE_CONNECTIONS:
                    s, e = connection
                    if landmarks[s].presence > 0.5 and landmarks[e].presence > 0.5:
                        cv2.line(image, (int(landmarks[s].x * w), int(landmarks[s].y * h)), (int(landmarks[e].x * w), int(landmarks[e].y * h)), color, 2)
                
                # Draw boxes
                fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                cv2.rectangle(image, (fox, foy), (fox+fow, foy+foh), color, 2)
                cv2.putText(image, f"P{i}", (fox, foy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        alerts = []
        for body in bodies:
            owner_id = body['id']
            color = body['color']
            
            # Show hand circles
            for pt in body['hand_points']:
                cv2.circle(image, (int(pt[0] * w), int(pt[1] * h)), 6, color, -1)
                
                # Check touch with other bodies
                for other_body in bodies:
                    if owner_id != other_body['id']:
                        if is_point_in_rect(pt, other_body['face_bbox']):
                            alerts.append((other_body['id'], owner_id, "Face"))
                        elif is_point_in_rect(pt, other_body['chest_bbox']):
                            alerts.append((other_body['id'], owner_id, "Chest"))

        if alerts:
            alert_text = "ALERTS: " + ", ".join([f"P{b} touches P{a}'s {area}!" for a, b, area in set(alerts)])
            cv2.putText(image, alert_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

        return image

webrtc_streamer(
    key="pose-detection",
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
