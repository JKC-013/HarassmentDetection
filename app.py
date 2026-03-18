import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import os
import av
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

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

# Create tabs for Live and Photo modes
tab1, tab2 = st.tabs(["📹 Live Camera", "📷 Photo Analysis"])

# Camera frame callback
def video_frame_callback(frame):
    """Process video frame - keep as minimal as possible."""
    try:
        img = frame.to_ndarray(format="bgr24")
        # Just flip and return - no other processing
        return av.VideoFrame.from_ndarray(cv2.flip(img, 1), format="bgr24")
    except Exception:
        return frame

# TAB 1: Live Camera (WebRTC)
with tab1:
    st.subheader("📹 Live Camera Feed")
    st.write("Real-time pose and hand detection")
    
    # RTC Config - explicitly empty to use browser defaults
    try:
        rtc_config_live = RTCConfiguration({"iceServers": []})
    except Exception:
        rtc_config_live = None
    
    try:
        webrtc_streamer(
            key="harassment-detection-live",
            mode=WebRtcMode.RECVONLY,  # Simpler mode: receive only
            video_frame_callback=video_frame_callback,
            rtc_configuration=rtc_config_live,
            media_stream_constraints={
                "video": {"width": {"max": 640}, "height": {"max": 480}},
                "audio": False
            },
            async_processing=False,
            webrtc_connection_timeout=30,  # Increase timeout
        )
    except Exception as e:
        st.error(f"❌ Live Camera Error: {str(e)}")
        st.info("💡 If camera doesn't work here, try the **Photo Analysis** tab instead!")

# TAB 2: Photo-based Analysis (Fallback)
with tab2:
    st.subheader("📷 Photo-Based Detection")
    st.write("Take a photo and see instant detection results")
    
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        # Convert to opencv format
        from PIL import Image
        img_pil = Image.open(picture)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        st.subheader("Processing...")
        
        # Run detection
        try:
            if pose_engine is not None and hand_engine is not None:
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                
                # Run detections
                pose_result = pose_engine.detect(mp_img)
                hand_result = hand_engine.detect(mp_img)
                
                # Draw pose landmarks
                if pose_result.pose_landmarks:
                    for lms in pose_result.pose_landmarks:
                        conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                        for s, e in conns:
                            if s < len(lms) and e < len(lms):
                                p1 = (int(lms[s].x*w), int(lms[s].y*h))
                                p2 = (int(lms[e].x*w), int(lms[e].y*h))
                                cv2.line(img, p1, p2, (0, 255, 0), 4)
                        for pt in lms:
                            cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, (0, 255, 0), -1)
                
                # Draw hand landmarks
                if hand_result.hand_landmarks:
                    for hlms in hand_result.hand_landmarks:
                        for pt in hlms:
                            cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 5, (255, 255, 255), -1)
                
                # Display result
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)
                st.success("✅ Detection complete!")
            else:
                st.warning("⏳ Models still loading...")
        except Exception as e:
            st.error(f"Detection error: {e}")

