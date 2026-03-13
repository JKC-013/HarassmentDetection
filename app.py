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

# --- CACHED DETECTORS ---
@st.cache_resource
def load_detectors():
    # 1. Pose Model
    p_path = 'pose_landmarker.task'
    p_model = None
    if os.path.exists(p_path):
        try:
            with open(p_path, 'rb') as f:
                p_data = f.read()
            p_opts = PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=p_data),
                running_mode=RunningMode.IMAGE,
                num_poses=4
            )
            p_model = PoseLandmarker.create_from_options(p_opts)
        except: pass

    # 2. Hand Model (Using Task API for compatibility)
    # Note: Streamlit Cloud usually has hand_landmarker.task or we can use the default.
    # If we don't have the task file, we'll gracefully fallback.
    h_path = 'hand_landmarker.task' 
    h_model = None
    # For now, let's use the Pose Landmarker only and see if we can get hand points from it.
    # Actually, PoseLandmarker provides index 15-22 which are wrist/fingers, but not all 21.
    # Let's try to initialize the HandLandmarker if the file exists.
    if os.path.exists(h_path):
        try:
            with open(h_path, 'rb') as f:
                h_data = f.read()
            h_opts = HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=h_data),
                running_mode=RunningMode.IMAGE,
                num_hands=4
            )
            h_model = HandLandmarker.create_from_options(h_opts)
        except: pass
        
    return p_model, h_model

POSE_LANDMARKER, HAND_LANDMARKER = load_detectors()
DETECTOR_LOCK = threading.Lock()

st.sidebar.subheader("System Diagnostics")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Pose Model: {'✅' if POSE_LANDMARKER else '❌ (Missing pose_landmarker.task)'}")
st.sidebar.write(f"Hand Model: {'✅' if HAND_LANDMARKER else '⚠️ (hand_landmarker.task missing - showing basic points)'}")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press **"Start"** below.
3. If `hand_landmarker.task` is in the repo, you will see all **21 hand dots**. 
4. Otherwise, it will show the basic pose-provided wrist/finger points.
""")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def is_point_in_rect(pt, rect):
    return rect[0] <= pt[0] <= rect[0]+rect[2] and rect[1] <= pt[1] <= rect[1]+rect[3]

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not POSE_LANDMARKER:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            with DETECTOR_LOCK:
                pose_res = POSE_LANDMARKER.detect(mp_img)
                hand_res = HAND_LANDMARKER.detect(mp_img) if HAND_LANDMARKER else None

            bodies = []
            if pose_res and pose_res.pose_landmarks:
                for i, lms in enumerate(pose_res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    
                    # Face Bbox
                    face_pts = [lms[j] for j in range(9)]
                    fx, fy = [p.x for p in face_pts], [p.y for p in face_pts]
                    fw, fh = max(fx)-min(fx), max(fy)-min(fy)
                    face_rect = (min(fx)-fw*0.1, min(fy)-fh*0.5, fw*1.2, fh*2.0)
                    
                    # Chest Bbox
                    cx_min = min(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    cx_max = max(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    cy_min = min(lms[11].y, lms[12].y)
                    cy_max = max(lms[23].y, lms[24].y)
                    chest_rect = (cx_min, cy_min, cx_max-cx_min, (cy_max-cy_min)*0.6)
                    
                    bodies.append({'id': i, 'face': face_rect, 'chest': chest_rect, 'color': color, 'wrists': [(lms[15].x, lms[15].y), (lms[16].x, lms[16].y)]})
                    
                    # Draw Pose Connections
                    for conn in mp.solutions.pose.POSE_CONNECTIONS:
                        if lms[conn[0]].presence > 0.5 and lms[conn[1]].presence > 0.5:
                            cv2.line(img, (int(lms[conn[0]].x*w), int(lms[conn[0]].y*h)), (int(lms[conn[1]].x*w), int(lms[conn[1]].y*h)), color, 2)
                    
                    # Draw Boxes
                    bx, by, bw, bh = int(face_rect[0]*w), int(face_rect[1]*h), int(face_rect[2]*w), int(face_rect[3]*h)
                    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, 2)
                    cv2.putText(img, f"P{i}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw Hand Details (21 points)
            alerts = []
            if hand_res and hand_res.hand_landmarks:
                for h_lms in hand_res.hand_landmarks:
                    # Associate hand with person via wrist
                    hw = (h_lms[0].x, h_lms[0].y)
                    min_d, owner_id = 0.2, -1
                    for b in bodies:
                        for bw in b['wrists']:
                            d = np.sqrt((hw[0]-bw[0])**2 + (hw[1]-bw[1])**2)
                            if d < min_d: min_d, owner_id = d, b['id']
                    
                    h_color = bodies[owner_id]['color'] if owner_id != -1 else (128, 128, 128)
                    for pt in h_lms:
                        cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, h_color, -1)
                        if owner_id != -1:
                            for other in bodies:
                                if owner_id != other['id']:
                                    if is_point_in_rect((pt.x, pt.y), other['face']): alerts.append((other['id'], owner_id, "Face"))
                                    elif is_point_in_rect((pt.x, pt.y), other['chest']): alerts.append((other['id'], owner_id, "Chest"))
            
            # Fallback draw if hand model missing (dots from pose)
            elif not hand_res:
                for b in bodies:
                    for wr in b['wrists']:
                        cv2.circle(img, (int(wr[0]*w), int(wr[1]*h)), 8, b['color'], -1)

            if alerts:
                unique_alerts = set(alerts)
                msg = "ALERTS: " + ", ".join([f"P{b} touch P{a} {area}" for a, b, area in unique_alerts])
                cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        except Exception: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="pose-final-rollback", video_processor_factory=PoseTransformer, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False})
