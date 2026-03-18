import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import os
import av
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Harassment Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_mediapipe_engines():
    """Load pose and hand detection models."""
    # Check paths (Docker or local)
    p_path = '/app/pose_landmarker.task' if os.path.exists('/app/pose_landmarker.task') else 'pose_landmarker.task'
    h_path = '/app/hand_landmarker.task' if os.path.exists('/app/hand_landmarker.task') else 'hand_landmarker.task'
    
    pose_engine = None
    hand_engine = None
    status = []
    
    try:
        if os.path.exists(p_path):
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=p_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2
            )
            pose_engine = vision.PoseLandmarker.create_from_options(options)
            status.append("✅ Pose Detection Ready")
        else:
            status.append("❌ Pose Model Not Found")
    except Exception as e:
        status.append(f"❌ Pose Error: {str(e)[:50]}")
    
    try:
        if os.path.exists(h_path):
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=h_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=4
            )
            hand_engine = vision.HandLandmarker.create_from_options(options)
            status.append("✅ Hand Detection Ready")
        else:
            status.append("❌ Hand Model Not Found")
    except Exception as e:
        status.append(f"❌ Hand Error: {str(e)[:50]}")
    
    return pose_engine, hand_engine, status

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Main title
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ Harassment Detection AI")
    st.markdown("Real-time pose and hand detection using MediaPipe")
with col2:
    st.empty()

# Load models
pose_engine, hand_engine, status_messages = load_mediapipe_engines()

# Sidebar - Status
with st.sidebar:
    st.header("📊 Status")
    for msg in status_messages:
        if "✅" in msg:
            st.success(msg)
        elif "❌" in msg:
            st.error(msg)
        else:
            st.info(msg)
    
    st.divider()
    
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Click START** to begin
    2. **Grant camera permission** when prompted
    3. **View real-time detection** with pose and hand landmarks
    
    **Troubleshooting:**
    - Refresh if camera doesn't appear
    - Try Chrome or Firefox
    - Check browser permissions (📷)
    """)

# Main content
st.divider()

# WebRTC Config - use defaults (no explicit STUN/TURN)
rtc_config = None

# Camera frame callback
def video_frame_callback(frame):
    """Process video frame."""
    try:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print(f"Frame error: {e}")
        return frame

# Display camera
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    try:
        webrtc_streamer(
            key="harassment-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
    except Exception as e:
        st.error(f"❌ Camera Error: {str(e)}")
        with st.expander("📖 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            - **Connection timeout**: Refresh page, try different browser
            - **Permission denied**: Check browser camera permissions
            - **Black screen**: Grant permission and wait 2-3 seconds
            
            **Browser Support:**
            - Chrome/Edge: ✅ Best
            - Firefox: ✅ Good
            - Safari: ⚠️ Limited
            """)

with col2:
    st.subheader("ℹ️ Info")
    st.info("""
    **Models:**
    - MediaPipe Pose
    - MediaPipe Hands
    
    **Frame Rate:**
    - ~20 FPS
    
    **Supported:**
    - Up to 2 people
    - Up to 4 hands
    """)

# Footer
st.divider()
st.caption("🔐 Built with Streamlit + MediaPipe • No data stored")

