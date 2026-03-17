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
with status_placeholder.status("🚀 Initializing AI Engine...", expanded=True) as status:
    st.sidebar.subheader("System Diagnostics")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    
    @st.cache_resource
    def load_engines():
        # Pose Engine
        p_path = 'pose_landmarker.task'
        p_engine = None
        if os.path.exists(p_path):
            try:
                p_engine = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=p_path),
                    running_mode=RunningMode.IMAGE,
                    num_poses=4,
                ))
            except: pass
            
        # Hand Engine (21 finger dots)
        h_path = 'hand_landmarker.task'
        h_engine = None
        if os.path.exists(h_path):
            try:
                h_engine = HandLandmarker.create_from_options(HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=h_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=4,
                ))
            except: pass
            
        return p_engine, h_engine

    POSE_ENGINE, HAND_ENGINE = load_engines()
    
    st.sidebar.write(f"Pose AI: {'✅ Ready' if POSE_ENGINE else '❌ Missing pose_landmarker.task'}")
    st.sidebar.write(f"Hand AI: {'✅ 21 Dots Available' if HAND_ENGINE else '⚠️ Basic Tracking Only'}")
    
    if HAND_ENGINE:
        status.update(label="✅ Ready with 21 finger dots!", state="complete", expanded=False)
    elif POSE_ENGINE:
        status.update(label="✅ Ready with basic pose tracking!", state="complete", expanded=False)

status_placeholder.empty()

st.markdown("""
### Instructions:
1. **At least 2 people** for touch detection.
2. Press **"Start"** to open camera (Mirrored view).
3. Optimized for **Multiple Sessions** (AI engine shared via locking).
4. If hand dots missing, ensure `hand_landmarker.task` is in the root folder.
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
        
        # Mirror flip (matches local behavior)
        img = cv2.flip(img, 1)
        
        if not POSE_ENGINE:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # --- INFERENCE STEP (LOCK PROTECTED) ---
            with DETECTOR_LOCK:
                pose_res = POSE_ENGINE.detect(mp_img)
                hand_res = HAND_ENGINE.detect(mp_img) if HAND_ENGINE else None

            # --- POST-PROCESSING (NO LOCK NEEDED) ---
            bodies = []
            if pose_res and pose_res.pose_landmarks:
                for i, lms in enumerate(pose_res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    
                    # Face Box Estimate
                    pts = [lms[j] for j in range(9)]
                    fx, fy = [p.x for p in pts], [p.y for p in pts]
                    fw, fh = max(fx)-min(fx), max(fy)-min(fy)
                    face_r = (min(fx)-fw*0.2, min(fy)-fh*0.6, fw*1.4, fh*2.2)
                    
                    # Chest Box Estimate
                    cx = min(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    cw = max(lms[11].x, lms[12].x, lms[23].x, lms[24].x) - cx
                    cy = min(lms[11].y, lms[12].y)
                    ch = (max(lms[23].y, lms[24].y) - cy) * 0.6
                    chest_r = (cx, cy, cw, ch)
                    
                    bodies.append({'id': i, 'face': face_r, 'chest': chest_r, 'wrists': [(lms[15].x, lms[15].y), (lms[16].x, lms[16].y)], 'color': color})
                    
                    # Draw Pose skeleton
                    conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                    for s, e in conns:
                        cv2.line(img, (int(lms[s].x*w), int(lms[s].y*h)), (int(lms[e].x*w), int(lms[e].y*h)), color, 2)
                    
                    bx, by, bw, bh = int(face_r[0]*w), int(face_r[1]*h), int(face_r[2]*w), int(face_r[3]*h)
                    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, 2)
                    cv2.putText(img, f"Person {i}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            alerts = []
            # 21 Finger Dots Logic
            if HAND_ENGINE and hand_res and hand_res.hand_landmarks:
                for hlms in hand_res.hand_landmarks:
                    hw = (hlms[0].x, hlms[0].y) # wrist
                    best_d, oid = 0.2, -1
                    for b in bodies:
                        for bw in b['wrists']:
                            d = np.sqrt((hw[0]-bw[0])**2 + (hw[1]-bw[1])**2)
                            if d < best_d: best_d, oid = d, b['id']
                    
                    hc = bodies[oid]['color'] if (oid != -1 and oid < len(bodies)) else (128, 128, 128)
                    for pt in hlms:
                        cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, hc, -1)
                        if oid != -1 and oid < len(bodies):
                            for other in bodies:
                                if oid != other['id']:
                                    if is_point_in_rect((pt.x, pt.y), other['face']): alerts.append((other['id'], oid, "Face"))
                                    elif is_point_in_rect((pt.x, pt.y), other['chest']): alerts.append((other['id'], oid, "Chest"))

            # Fallback to basic wrist dots if hand model missing
            elif pose_res and pose_res.pose_landmarks:
                for b in bodies:
                    for wr in b['wrists']:
                        cv2.circle(img, (int(wr[0]*w), int(wr[1]*h)), 10, b['color'], -1)

            if alerts:
                msg = "TOUCH: " + ", ".join([f"P{b}->P{a} {area}" for a, b, area in set(alerts)])
                cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="pose-final-engine-v13-pro", video_processor_factory=DetectProcessor, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, async_processing=False)
