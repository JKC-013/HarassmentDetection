import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import os
import av
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading
import time

# --- SHARED STATE FOR ASYNC AI (GLOBAL FOR THREAD-SAFE ACCESS) ---
class GlobalAIState:
    def __init__(self):
        self.pose_results = None
        self.hand_results = None
        self.processing = False
        self.lock = threading.Lock()

if 'GLOBAL_AI_STATE' not in globals():
    globals()['GLOBAL_AI_STATE'] = GlobalAIState()

ai_state = globals()['GLOBAL_AI_STATE']

# --- PAGE CONFIG ---
st.set_page_config(page_title="Harassment Detection AI", layout="wide")

st.title("🛡️ Harassment Detection AI (Stable Docker)")
st.markdown("If the camera 'spins' and closes, please try refreshing or checking your browser permissions.")

# --- MODEL LOADING (CACHED) ---

@st.cache_resource
def load_mediapipe_engines():
    # Use ABSOLUTE paths for Docker stability, fallback to local for dev
    p_path = '/app/pose_landmarker.task' if os.path.exists('/app/pose_landmarker.task') else 'pose_landmarker.task'
    h_path = '/app/hand_landmarker.task' if os.path.exists('/app/hand_landmarker.task') else 'hand_landmarker.task'
    
    pose_engine = None
    hand_engine = None
    status_msg = []
    
    try:
        if os.path.exists(p_path):
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=p_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2
            )
            pose_engine = vision.PoseLandmarker.create_from_options(options)
            status_msg.append("✅ Pose Engine Loaded")
        else:
            status_msg.append(f"❌ Pose Model Missing at {p_path}")
            
        if os.path.exists(h_path):
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=h_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=4
            )
            hand_engine = vision.HandLandmarker.create_from_options(options)
            status_msg.append("✅ Hand Engine Loaded")
        else:
            status_msg.append(f"❌ Hand Model Missing at {h_path}")
            
    except Exception as e:
        status_msg.append(f"⚠️ Engine Error: {str(e)}")
        
    return pose_engine, hand_engine, status_msg

# Pre-load models so user sees status immediately
pose_engine, hand_engine, engine_status = load_mediapipe_engines()

# --- CALLBACK FUNCTION ---

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Mirror for natural view
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # --- ASYNC INFERENCE LOGIC ---
    # Try to grab the lock. If AI is busy, we SKIP inference and just draw the PREVIOUS results.
    # THIS PREVENTS STALLING THE STREAM.
    if not ai_state.processing:
        def run_inference():
            ai_state.processing = True
            try:
                if pose_engine:
                    ai_state.pose_results = pose_engine.detect(mp_img)
                if hand_engine:
                    ai_state.hand_results = hand_engine.detect(mp_img)
            except Exception:
                pass
            finally:
                ai_state.processing = False
        
        # Start inference in a background thread if it's not already running
        threading.Thread(target=run_inference, daemon=True).start()

    # --- DRAWING (USING LATEST KNOWN RESULTS) ---
    with ai_state.lock:
        pres = ai_state.pose_results
        hres = ai_state.hand_results

    if pres and pres.pose_landmarks:
        for lms in pres.pose_landmarks:
            conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
            for s, e in conns:
                if s < len(lms) and e < len(lms):
                    p1 = (int(lms[s].x*w), int(lms[s].y*h))
                    p2 = (int(lms[e].x*w), int(lms[e].y*h))
                    cv2.line(img, p1, p2, (0, 255, 0), 4)
            for pt in lms:
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, (0, 255, 0), -1)

    if hres and hres.hand_landmarks:
        for hlms in hres.hand_landmarks:
            for pt in hlms:
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 5, (255, 255, 255), -1)

    cv2.putText(img, "AI ACTIVE (ASYNC)", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- APP UI ---

st.sidebar.title("AI Detection Dashboard")
for msg in engine_status:
    if "✅" in msg:
        st.sidebar.success(msg)
    elif "❌" in msg:
        st.sidebar.error(msg)
    else:
        st.sidebar.warning(msg)

st.sidebar.markdown("---")
st.sidebar.info("Status: Tracking Active")
st.sidebar.markdown("""
- **Person Detection**: Green Lines
- **Hand Landmarks**: White Dots
""")

# Simplified RTC Configuration (One STUN server often more stable on Cloud)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Final Glitch-Proof Configuration
webrtc_streamer(
    key="harassment-detection-stable-v100", # New key to force widget refresh
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
    async_processing=True,
)
