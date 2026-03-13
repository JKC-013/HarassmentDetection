import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import sys
import ctypes

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FIX: Monkeypatch MediaPipe's ctypes error for Python 3.14 (even if on 3.11 now, for safety)
try:
    from mediapipe.tasks.python.core import mediapipe_c_bindings
    if not hasattr(mediapipe_c_bindings, 'free'):
        try:
            libc = ctypes.CDLL("libc.so.6")
            mediapipe_c_bindings.free = libc.free
        except:
            pass
except:
    pass

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# Page Config
st.set_page_config(page_title="Multi-Person Pose Detection", layout="wide")

st.title("🏃 Multi-Person Pose Detection")

# --- CACHED MODEL INITIALIZATION ---
# This avoids re-initializing the model for every new connection, which is slow on Render.
@st.cache_resource
def load_pose_landmarker():
    if not os.path.exists('pose_landmarker.task'):
        return None
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

LANDMARKER = load_pose_landmarker()

st.sidebar.subheader("System Diagnostics")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Model Ready: {'✅' if LANDMARKER else '❌'}")

if not LANDMARKER:
    st.error("CRITICAL: `pose_landmarker.task` not found or failed to load. Please ensure it's in your GitHub repo!")

st.markdown("""
### Instructions:
1. **Required**: At least 2 people for this experiment.
2. Press the **"Start"** button below to begin.
3. Once started, the webcam feed will appear. (It may take a few seconds to connect)
4. Alerts will show when one person's hand touches another person's face or chest.
5. **Press 'q' to stop** (or use the Stop button below).
""")

# Robust STUN server list
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
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
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width, height = xmax - xmin, ymax - ymin
    return (max(0, xmin - width*0.25), max(0, ymin - height*0.75), min(1, width*1.5), min(1, height*2.5))

def get_chest_bbox_from_pose(landmarks):
    xmin = min(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    xmax = max(landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x)
    ymin, ymax = min(landmarks[11].y, landmarks[12].y), max(landmarks[23].y, landmarks[24].y)
    return (xmin, ymin, xmax - xmin, (ymax - ymin) * 0.6)

def is_point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.PERSON_COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if LANDMARKER is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            pose_result = LANDMARKER.detect(mp_img)

            bodies = []
            if pose_result.pose_landmarks:
                for i, landmarks in enumerate(pose_result.pose_landmarks):
                    color = self.PERSON_COLORS[i % len(self.PERSON_COLORS)]
                    face_bbox = get_face_bbox_from_pose(landmarks)
                    chest_bbox = get_chest_bbox_from_pose(landmarks)
                    hand_points = [(landmarks[j].x, landmarks[j].y) for j in [19, 20, 17, 18]]
                    
                    bodies.append({'id': i, 'face_bbox': face_bbox, 'chest_bbox': chest_bbox, 'hand_points': hand_points, 'color': color})
                    
                    for s, e in POSE_CONNECTIONS:
                        if landmarks[s].presence > 0.5 and landmarks[e].presence > 0.5:
                            cv2.line(img, (int(landmarks[s].x * w), int(landmarks[s].y * h)), (int(landmarks[e].x * w), int(landmarks[e].y * h)), color, 2)
                    
                    fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                    cv2.rectangle(img, (fox, foy), (fox+fow, foy+foh), color, 2)
                    cv2.putText(img, f"P{i}", (fox, foy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            alerts = []
            for body in bodies:
                for pt in body['hand_points']:
                    cv2.circle(img, (int(pt[0] * w), int(pt[1] * h)), 6, body['color'], -1)
                    for other in bodies:
                        if body['id'] != other['id']:
                            if is_point_in_rect(pt, other['face_bbox']): alerts.append((other['id'], body['id'], "Face"))
                            elif is_point_in_rect(pt, other['chest_bbox']): alerts.append((other['id'], body['id'], "Chest"))

            if alerts:
                alert_text = "ALERTS: " + ", ".join([f"P{b} touches P{a}'s {area}!" for a, b, area in set(alerts)])
                cv2.putText(img, alert_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img

import av
ctx = webrtc_streamer(
    key="pose-detection-v3", # New key to force fresh state
    video_processor_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Backend processing improves connection UI
)

if ctx.state.playing:
    st.write("✅ Connection established! Analyzing video...")
else:
    st.write("⌛ Waiting for connection... Please click 'Start' and grant camera access.")
