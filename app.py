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
import urllib.request

# --- MODEL DOWNLOAD HELPER ---
def download_model_if_missing(model_name, url):
    """Download MediaPipe model if not present locally."""
    if os.path.exists(model_name):
        print(f"✅ {model_name} already exists")
        return model_name
    
    try:
        print(f"📥 Downloading {model_name} from {url}...")
        urllib.request.urlretrieve(url, model_name)
        if os.path.exists(model_name):
            size = os.path.getsize(model_name) / (1024*1024)  # Size in MB
            print(f"✅ Downloaded {model_name} ({size:.1f} MB)")
            return model_name
        else:
            print(f"❌ Download failed: File not created")
            return None
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {str(e)}")
        return None

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
    # Download models if missing
    pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full.tflite"
    hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task"
    
    print("🔄 Starting model loading...")
    pose_file = download_model_if_missing("pose_landmarker.task", pose_url)
    hand_file = download_model_if_missing("hand_landmarker.task", hand_url)
    
    # Use ABSOLUTE paths for Docker stability, fallback to local for dev
    p_path = '/app/pose_landmarker.task' if os.path.exists('/app/pose_landmarker.task') else 'pose_landmarker.task'
    h_path = '/app/hand_landmarker.task' if os.path.exists('/app/hand_landmarker.task') else 'hand_landmarker.task'
    
    pose_engine = None
    hand_engine = None
    status_msg = []
    
    try:
        print(f"Checking for pose model at: {p_path}")
        if os.path.exists(p_path):
            print(f"Loading pose model...")
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=p_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2
            )
            pose_engine = vision.PoseLandmarker.create_from_options(options)
            status_msg.append("✅ Pose Engine Loaded")
            print("✅ Pose engine loaded successfully")
        else:
            status_msg.append(f"❌ Pose Model Missing at {p_path}")
            print(f"❌ Pose model not found at {p_path}")
            
        print(f"Checking for hand model at: {h_path}")
        if os.path.exists(h_path):
            print(f"Loading hand model...")
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=h_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=4
            )
            hand_engine = vision.HandLandmarker.create_from_options(options)
            status_msg.append("✅ Hand Engine Loaded")
            print("✅ Hand engine loaded successfully")
        else:
            status_msg.append(f"❌ Hand Model Missing at {h_path}")
            print(f"❌ Hand model not found at {h_path}")
            
    except Exception as e:
        status_msg.append(f"⚠️ Engine Error: {str(e)}")
        print(f"⚠️ Engine loading error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return pose_engine, hand_engine, status_msg

# Pre-load models so user sees status immediately
print("📦 Pre-loading MediaPipe models...")
pose_engine, hand_engine, engine_status = load_mediapipe_engines()
print(f"Model loading complete. Pose: {pose_engine is not None}, Hand: {hand_engine is not None}")

# --- CALLBACK FUNCTION ---

def video_frame_callback(frame):
    """Bare-bones callback - return frame as fast as possible."""
    try:
        # 1. Get image
        img = frame.to_ndarray(format="bgr24")
        # 2. Mirror it
        img = cv2.flip(img, 1)
        # 3. Return ASAP
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print(f"Frame error: {e}")
        return frame

# --- APP UI ---

st.sidebar.title("🤖 AI Detection Dashboard")
st.sidebar.subheader("Model Status")

for msg in engine_status:
    if "✅" in msg:
        st.sidebar.success(msg)
    elif "❌" in msg:
        st.sidebar.error(msg)
    else:
        st.sidebar.warning(msg)

# Show current state
if pose_engine is not None:
    st.sidebar.write("✅ **Pose Detection:** Ready")
else:
    st.sidebar.write("⏳ **Pose Detection:** Waiting for model file")
    
if hand_engine is not None:
    st.sidebar.write("✅ **Hand Detection:** Ready")
else:
    st.sidebar.write("⏳ **Hand Detection:** Waiting for model file")

st.sidebar.markdown("---")
st.sidebar.info("""
🎥 **Camera Instructions:**
1. Click START button
2. Allow camera permission
3. Grant browser access when prompted

❌ **If camera doesn't appear:**
- Refresh page (Ctrl+R)
- Try Chrome/Firefox
- Check browser console (F12)
""")

# RTC Configuration with multiple STUN and TURN servers
# TURN servers are more reliable than STUN in restricted environments like HF Spaces
try:
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    })
except Exception as e:
    print(f"RTC config error: {e}")
    RTC_CONFIG = None

st.write("---")
st.subheader("📹 Live Camera Feed")
st.write("Click START below, grant camera permission, and the feed should appear.")
st.info("💡 If camera doesn't appear: Refresh page → Check browser permissions → Check browser console (F12)")

try:
    webrtc_streamer(
        key="harassment-detection-v104",
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )
except Exception as webrtc_error:
    st.error(f"❌ WebRTC Connection Error")
    st.error(str(webrtc_error))
    st.warning("""
    **Troubleshooting Steps:**
    1. Click your browser's refresh button
    2. Make sure you granted camera permission
    3. Try a different browser (Chrome/Firefox work best)
    4. Check browser console (F12 → Console tab) for detailed errors
    5. Check your firewall/VPN settings
    """)

