import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import sys
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

# --- CACHED MODEL ---
@st.cache_resource
def get_landmarker():
    p_path = 'pose_landmarker.task'
    if not os.path.exists(p_path):
        return None
    try:
        with open(p_path, 'rb') as f:
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
    except Exception:
        return None

LANDMARKER = get_landmarker()
LANDMARKER_LOCK = threading.Lock()

st.sidebar.subheader("System Diagnostics")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Model: {'✅ Ready' if LANDMARKER else '❌ Missing pose_landmarker.task'}")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press **"Start"** to open your camera.
3. This is the **Stable Version**.
""")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not LANDMARKER:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            with LANDMARKER_LOCK:
                res = LANDMARKER.detect(mp_img)

            if res and res.pose_landmarks:
                for i, lms in enumerate(res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    
                    # Draw Poses
                    conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                    for s, e in conns:
                        cv2.line(img, (int(lms[s].x*w), int(lms[s].y*h)), (int(lms[e].x*w), int(lms[e].y*h)), color, 2)
                    
                    # Individual wrist dots as basic hand indicators
                    for idx in [15, 16]: # Wrists
                        cv2.circle(img, (int(lms[idx].x*w), int(lms[idx].y*h)), 10, color, -1)
                        
                    # Face Box estimate
                    fx, fy = int(lms[0].x*w), int(lms[0].y*h)
                    cv2.rectangle(img, (fx-20, fy-40), (fx+20, fy+40), color, 2)
                    cv2.putText(img, f"P{i}", (fx-20, fy-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="pose-stable-rollback-final",
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
