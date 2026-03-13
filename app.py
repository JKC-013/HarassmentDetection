import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import sys
import ctypes
import threading
import av

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# Page Config
st.set_page_config(page_title="Multi-Person Pose Detection", layout="wide")

st.title("🏃 Multi-Person Pose Detection")

# --- THREAD-SAFE GLOBAL LANDMARKER ---
# Using a Lock ensures multiple frames don't hit the landmarker at the exact same time.
LANDMARKER_LOCK = threading.Lock()

@st.cache_resource
def get_landmarker():
    if not os.path.exists('pose_landmarker.task'):
        return None
    try:
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
        return PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing landmarker: {e}")
        return None

# Load it once
LANDMARKER = get_landmarker()

st.sidebar.subheader("System Diagnostics")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Model State: {'✅ Ready' if LANDMARKER else '❌ Failed to load'}")

if not LANDMARKER:
    st.error("CRITICAL: Failed to load `pose_landmarker.task`. Ensure the file is at the root of your repository.")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press the **"Start"** button below.
3. Grant camera access. (Connection can take 10-20s on first load)
4. Alerts will show when one person's hand touches another person's face/chest.
5. **Press 'q' to stop** (or use the Stop button).
""")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20)
]

def get_face_bbox_from_pose(landmarks):
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    xs = [landmarks[i].x for i in face_indices]
    ys = [landmarks[i].y for i in face_indices]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    return (max(0, min(xs) - width*0.1), max(0, min(ys) - height*0.1), min(1, width*1.2), min(1, height*1.2))

def get_chest_bbox_from_pose(landmarks):
    xmin = min(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    xmax = max(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    ymin, ymax = min(landmarks[11].y, landmarks[12].y), max(landmarks[23].y, landmarks[24].y)
    return (xmin, ymin, xmax - xmin, (ymax - ymin) * 0.5)

def is_point_in_rect(point, rect):
    x, y = point
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.PERSON_COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # If model is not loaded, just return the raw frame
        if LANDMARKER is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            # Thread-safe detection
            with LANDMARKER_LOCK:
                pose_result = LANDMARKER.detect(mp_img)

            bodies = []
            if pose_result and pose_result.pose_landmarks:
                for i, landmarks in enumerate(pose_result.pose_landmarks):
                    color = self.PERSON_COLORS[i % len(self.PERSON_COLORS)]
                    face_bbox = get_face_bbox_from_pose(landmarks)
                    chest_bbox = get_chest_bbox_from_pose(landmarks)
                    hand_info = [(landmarks[j].x, landmarks[j].y) for j in [19, 20, 17, 18]]
                    bodies.append({'id': i, 'face': face_bbox, 'chest': chest_bbox, 'hands': hand_info, 'color': color})
                    
                    # Draw Poses
                    for s, e in POSE_CONNECTIONS:
                        if landmarks[s].presence > 0.5 and landmarks[e].presence > 0.5:
                            cv2.line(img, (int(landmarks[s].x * w), int(landmarks[s].y * h)), (int(landmarks[e].x * w), int(landmarks[e].y * h)), color, 2)
                    
                    fx, fy, fw, fh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), color, 2)

            alerts = []
            for b in bodies:
                for pt in b['hands']:
                    cv2.circle(img, (int(pt[0]*w), int(pt[1]*h)), 6, b['color'], -1)
                    for other in bodies:
                        if b['id'] != other['id']:
                            if is_point_in_rect(pt, other['face']): alerts.append(f"P{b['id']} touch P{other['id']} Face")
                            elif is_point_in_rect(pt, other['chest']): alerts.append(f"P{b['id']} touch P{other['id']} Chest")
            
            if alerts:
                # Basic flash and text
                cv2.putText(img, "ALERT: TOUCH DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        except Exception as e:
            # If processing fails once, just send the frame
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="pose-detection-final-v4",
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
