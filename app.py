import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import sys
import threading
import av

# MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_pose_solutions = mp.solutions.pose

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# Page Config
st.set_page_config(page_title="Multi-Person Pose Detection", layout="wide")

st.title("🏃 Multi-Person Pose Detection")

# --- INITIALIZATION STATUS ---
# This shows the user what's happening during the long load time
status_placeholder = st.empty()
with status_placeholder.status("🚀 Initializing AI Models...", expanded=True) as status:
    st.write("Checking Python environment...")
    st.sidebar.subheader("System Diagnostics")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    
    st.write("Loading detectors (this may take a minute on first run)...")
    
    @st.cache_resource
    def load_detectors():
        # 1. Pose Landmarker (Multi-Person)
        p_path = 'pose_landmarker.task'
        if not os.path.exists(p_path):
            p_model = None
        else:
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
                p_model = PoseLandmarker.create_from_options(options)
            except:
                p_model = None
                
        # 2. Hand Detector (Legacy API for 21-points)
        try:
            h_model = mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=4
            )
        except:
            h_model = None
            
        return p_model, h_model

    POSE_LANDMARKER, HANDS_DETECTOR = load_detectors()
    
    st.sidebar.write(f"Pose Model: {'✅ Ready' if POSE_LANDMARKER else '❌ Error'}")
    st.sidebar.write(f"Hand Model: {'✅ Ready' if HANDS_DETECTOR else '❌ Error'}")
    
    if not POSE_LANDMARKER or not HANDS_DETECTOR:
        st.error("Wait! One of the AI models failed to start. Please check the sidebar.")
    else:
        status.update(label="✅ Models Loaded Successfully!", state="complete", expanded=False)

# Clear status placeholder after success
status_placeholder.empty()

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press **"Start"** below to open your camera.
3. The app will trace your pose and all **21 hand points**.
4. Alerts flash when hands touch a Face or Chest area.
""")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]}
)

DETECTOR_LOCK = threading.Lock()

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_point_in_rect(point, rect):
    rx, ry, rw, rh = rect
    return rx <= point[0] <= rx + rw and ry <= point[1] <= ry + rh

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
                hand_res = HANDS_DETECTOR.process(rgb_img) if HANDS_DETECTOR else None

            bodies = []
            if pose_res and pose_res.pose_landmarks:
                for i, lms in enumerate(pose_res.pose_landmarks):
                    color = self.colors[i % len(self.colors)]
                    
                    # Estimate Face Bbox
                    face_pts = [lms[j] for j in range(9)]
                    fx_s = [p.x for p in face_pts]
                    fy_s = [p.y for p in face_pts]
                    fw_raw, fh_raw = max(fx_s)-min(fx_s), max(fy_s)-min(fy_s)
                    face_bbox = (min(fx_s)-fw_raw*0.1, min(fy_s)-fh_raw*0.5, fw_raw*1.2, fh_raw*2.0)
                    
                    # Estimate Chest Bbox
                    cx_min = min(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    cx_max = max(lms[11].x, lms[12].x, lms[23].x, lms[24].x)
                    cy_min = min(lms[11].y, lms[12].y)
                    cy_max = max(lms[23].y, lms[24].y)
                    chest_bbox = (cx_min, cy_min, cx_max-cx_min, (cy_max-cy_min)*0.6)
                    
                    bodies.append({'id': i, 'face': face_bbox, 'chest': chest_bbox, 'wrists': [(lms[15].x, lms[15].y), (lms[16].x, lms[16].y)], 'color': color})
                    
                    # Draw Skeleton
                    for conn in mp_pose_solutions.POSE_CONNECTIONS:
                        if lms[conn[0]].presence > 0.5 and lms[conn[1]].presence > 0.5:
                            cv2.line(img, (int(lms[conn[0]].x*w), int(lms[conn[0]].y*h)), (int(lms[conn[1]].x*w), int(lms[conn[1]].y*h)), color, 2)
                    
                    fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                    cv2.rectangle(img, (fox, foy), (fox+fow, foy+foh), color, 2)
                    cv2.putText(img, f"Person {i}", (fox, foy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            alerts = []
            if hand_res and hand_res.multi_hand_landmarks:
                for hand_lms in hand_res.multi_hand_landmarks:
                    # Association by wrist
                    h_wrist = (hand_lms.landmark[0].x, hand_lms.landmark[0].y)
                    min_d, owner_id = 0.2, -1
                    for b in bodies:
                        for b_wrist in b['wrists']:
                            d = get_distance(h_wrist, b_wrist)
                            if d < min_d: min_d, owner_id = d, b['id']
                    
                    h_color = bodies[owner_id]['color'] if owner_id != -1 else (128, 128, 128)
                    for lm in hand_lms.landmark:
                        cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 4, h_color, -1)
                        if owner_id != -1:
                            for other in bodies:
                                if owner_id != other['id']:
                                    pt = (lm.x, lm.y)
                                    if is_point_in_rect(pt, other['face']): alerts.append((other['id'], owner_id, "Face"))
                                    elif is_point_in_rect(pt, other['chest']): alerts.append((other['id'], owner_id, "Chest"))

            if alerts:
                unique_alerts = set(alerts)
                alert_msg = "ALERTS: " + ", ".join([f"P{b} touch P{a}'s {area}" for a, b, area in unique_alerts])
                cv2.putText(img, alert_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="pose-detection-stable-v8",
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
