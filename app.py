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
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

# Page Config
st.set_page_config(page_title="Multi-Person Pose Detection", layout="wide")

st.title("🏃 Multi-Person Pose Detection")

# --- INITIALIZATION STATUS ---
status_placeholder = st.empty()
with status_placeholder.status("🚀 Loading AI Engine...", expanded=True) as status:
    st.sidebar.subheader("System Diagnostics")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    
    @st.cache_resource
    def load_engines():
        # Pose Engine
        p_path = 'pose_landmarker.task'
        p_engine = None
        if os.path.exists(p_path):
            try:
                base_options = python.BaseOptions(model_asset_buffer=open(p_path, 'rb').read())
                options = PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.IMAGE,
                    num_poses=4,
                )
                p_engine = PoseLandmarker.create_from_options(options)
            except: pass
            
        # Optional Hand Engine (for 21 points)
        h_path = 'hand_landmarker.task'
        h_engine = None
        if os.path.exists(h_path):
            try:
                base_options = python.BaseOptions(model_asset_buffer=open(h_path, 'rb').read())
                options = HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.IMAGE,
                    num_hands=4,
                )
                h_engine = HandLandmarker.create_from_options(options)
            except: pass
            
        return p_engine, h_engine

    POSE_ENGINE, HAND_ENGINE = load_engines()
    
    st.sidebar.write(f"Pose AI: {'✅ Ready' if POSE_ENGINE else '❌ Missing pose_landmarker.task'}")
    st.sidebar.write(f"Hand AI: {'✅ Full (21 pts)' if HAND_ENGINE else '⚠️ Basic (4 pts)'}")
    
    if HAND_ENGINE:
        status.update(label="✅ Ready with 21-point tracking!", state="complete", expanded=False)
    elif POSE_ENGINE:
        status.update(label="✅ Ready with basic pose tracking!", state="complete", expanded=False)
    else:
        st.error("Missing model files! Please upload pose_landmarker.task to your GitHub.")

status_placeholder.empty()

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press **"Start"** below.
3. Hand dots will automatically use 21 points if `hand_landmarker.task` is uploaded.
""")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

DETECTOR_LOCK = threading.Lock()

def is_point_in_rect(pt, rect):
    return rect[0] <= pt[0] <= rect[0]+rect[2] and rect[1] <= pt[1] <= rect[1]+rect[3]

class DetectProcessor(VideoProcessorBase):
    def __init__(self):
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not POSE_ENGINE:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            with DETECTOR_LOCK:
                pose_res = POSE_ENGINE.detect(mp_img)
                hand_res = HAND_ENGINE.detect(mp_img) if HAND_ENGINE else None

            bodies = []
            if pose_res.pose_landmarks:
                for i, lms in enumerate(pose_res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    
                    # Boxes
                    face_pts = [lms[j] for j in range(9)]
                    fx, fy = [p.x for p in face_pts], [p.y for p in face_pts]
                    fw, fh = max(fx)-min(fx), max(fy)-min(fy)
                    face_r = (min(fx)-fw*0.1, min(fy)-fh*0.5, fw*1.2, fh*2.0)
                    
                    c_min_x = min(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    c_max_x = max(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    c_min_y = min(lms[11].y, lms[12].y)
                    c_max_y = max(lms[23].y, lms[24].y)
                    chest_r = (c_min_x, c_min_y, c_max_x-c_min_x, (c_max_y-c_min_y)*0.6)
                    
                    bodies.append({'id': i, 'face': face_r, 'chest': chest_r, 'wrists': [(lms[15].x, lms[15].y), (lms[16].x, lms[16].y)], 'color': color})
                    
                    # Draw Poses (skeleton)
                    # Use indices directly to avoid importing solutions
                    conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
                    for s, e in conns:
                        cv2.line(img, (int(lms[s].x*w), int(lms[s].y*h)), (int(lms[e].x*w), int(lms[e].y*h)), color, 2)
                    
                    bx, by, bw, bh = int(face_r[0]*w), int(face_r[1]*h), int(face_r[2]*w), int(face_r[3]*h)
                    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, 2)
                    cv2.putText(img, f"P{i}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            alerts = []
            # Option 1: Hand Model (21 points)
            if HAND_ENGINE and hand_res and hand_res.hand_landmarks:
                for hlms in hand_res.hand_landmarks:
                    hw = (hlms[0].x, hlms[0].y)
                    best_d, oid = 0.2, -1
                    for b in bodies:
                        for bw in b['wrists']:
                            d = np.sqrt((hw[0]-bw[0])**2 + (hw[1]-bw[1])**2)
                            if d < best_d: best_d, oid = d, b['id']
                    
                    hc = bodies[oid]['color'] if oid != -1 else (128, 128, 128)
                    for pt in hlms:
                        cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, hc, -1)
                        if oid != -1:
                            for other in bodies:
                                if oid != other['id']:
                                    if is_point_in_rect((pt.x, pt.y), other['face']): alerts.append((other['id'], oid, "Face"))
                                    elif is_point_in_rect((pt.x, pt.y), other['chest']): alerts.append((other['id'], oid, "Chest"))

            # Option 2: Fallback to Pose-based hand blobs (4 pts per hand, 15-22 landmarks)
            elif pose_res.pose_landmarks:
                for b in bodies:
                    # Show wrists as basic indicators
                    for wr in b['wrists']:
                        cv2.circle(img, (int(wr[0]*w), int(wr[1]*h)), 8, b['color'], -1)

            if alerts:
                msg = "TOUCH DETECTED: " + ", ".join([f"P{b}->P{a} {area}" for a, b, area in set(alerts)])
                cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        except: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="pose-final-fix-v9", video_processor_factory=DetectProcessor, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False})
