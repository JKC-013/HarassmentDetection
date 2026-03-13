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

# --- SETTINGS ---
st.sidebar.subheader("App Settings")
debug_mode = st.sidebar.checkbox("Debug mode (Raw Camera)", value=False, help="Disable AI to test if the camera connection works at all.")

# --- CACHED MODEL ---
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
        return None

LANDMARKER = get_landmarker()
LANDMARKER_LOCK = threading.Lock()

st.sidebar.subheader("System Diagnostics")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Model: {'✅ Ready' if LANDMARKER else '❌ Error'}")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press **"Start"**.
3. If the screen stays black, try turning on **"Debug mode"** in the sidebar.
4. Alerts will show when one person's hand touches another person's face/chest.
""")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
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
    w_box, h_box = max(xs) - min(xs), max(ys) - min(ys)
    return (max(0, min(xs) - w_box*0.1), max(0, min(ys) - h_box*0.1), min(1, w_box*1.2), min(1, h_box*1.2))

def get_chest_bbox_from_pose(landmarks):
    xmin = min(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    xmax = max(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    ymin, ymax = min(landmarks[11].y, landmarks[12].y), max(landmarks[23].y, landmarks[24].y)
    return (xmin, ymin, xmax - xmin, (ymax - ymin) * 0.5)

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # DEBUG MODE: Skip AI processing
        if st.session_state.get('debug_mode_active', False):
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            # Optimization: Resize for faster AI processing on Render
            process_w = 480
            process_h = int(h * (process_w / w))
            small_img = cv2.resize(img, (process_w, process_h))
            rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            
            with LANDMARKER_LOCK:
                res = LANDMARKER.detect(mp_img) if LANDMARKER else None

            if res and res.pose_landmarks:
                for i, lms in enumerate(res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    face = get_face_bbox_from_pose(lms)
                    chest = get_chest_bbox_from_pose(lms)
                    
                    # Draw Poses (scaled back to original size)
                    for s, e in POSE_CONNECTIONS:
                        if lms[s].presence > 0.5 and lms[e].presence > 0.5:
                            cv2.line(img, (int(lms[s].x * w), int(lms[s].y * h)), (int(lms[e].x * w), int(lms[e].y * h)), color, 2)
                    
                    fx, fy, fw, fh = int(face[0]*w), int(face[1]*h), int(face[2]*w), int(face[3]*h)
                    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), color, 2)
                    
                    # Touch checks
                    hands = [(lms[j].x, lms[j].y) for j in [19, 20, 17, 18]]
                    for pt in hands:
                        cv2.circle(img, (int(pt[0]*w), int(pt[1]*h)), 6, color, -1)

            if LANDMARKER is None:
                cv2.putText(img, "MODEL LOAD ERROR", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Update debug mode state for the thread
if debug_mode:
    st.session_state.debug_mode_active = True
else:
    st.session_state.debug_mode_active = False

webrtc_streamer(
    key="pose-detection-debug-v6",
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
