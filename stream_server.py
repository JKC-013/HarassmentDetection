"""
Server-side real-time camera streaming with MediaPipe detection.
This uses Flask + OpenCV + HTTP streaming (MJPEG) instead of WebRTC P2P.
Works across networks and firewalls.
"""

# Setup headless display for Render
import os
import subprocess
import sys

# Try to start virtual display (xvfb)
try:
    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '640x480x24'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ['DISPLAY'] = ':99'
    print("✅ Virtual display started (Xvfb)")
except (FileNotFoundError, Exception) as e:
    print(f"⚠️  Xvfb not available: {e}")
    os.environ['DISPLAY'] = ''
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
from flask import Flask, render_template_string, Response
import numpy as np

app = Flask(__name__)

PERSON_COLORS = [
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255)  # Magenta
]

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_face_bbox_from_pose(landmarks):
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xs = [landmarks[i].x for i in face_indices if i < len(landmarks)]
    ys = [landmarks[i].y for i in face_indices if i < len(landmarks)]
    if not xs: return (0, 0, 0, 0)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width, height = xmax - xmin, ymax - ymin
    padding_x, padding_y = width * 0.5, height * 1.5 
    return (max(0, xmin - padding_x/2), max(0, ymin - padding_y/2), min(1, width + padding_x), min(1, height + padding_y))

def get_chest_bbox_from_pose(landmarks):
    if len(landmarks) <= 24: return (0, 0, 0, 0)
    shoulder_l, shoulder_r = landmarks[11], landmarks[12]
    hip_l, hip_r = landmarks[23], landmarks[24]
    xmin = min(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    xmax = max(shoulder_l.x, shoulder_r.x, hip_l.x, hip_r.x)
    ymin = min(shoulder_l.y, shoulder_r.y)
    ymax = max(hip_l.y, hip_r.y)
    torso_height = ymax - ymin
    chest_ymax = ymin + (torso_height * 0.6)
    return (xmin, ymin, xmax - xmin, chest_ymax - ymin)

def is_point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

# Global state
class CameraState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.pose_engine = None
        self.hand_engine = None

state = CameraState()

# Load MediaPipe models
def load_models():
    """Load pose and hand detection models."""
    p_path = '/app/pose_landmarker.task' if os.path.exists('/app/pose_landmarker.task') else 'pose_landmarker.task'
    h_path = '/app/hand_landmarker.task' if os.path.exists('/app/hand_landmarker.task') else 'hand_landmarker.task'
    
    try:
        if os.path.exists(p_path):
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=p_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=2
            )
            state.pose_engine = vision.PoseLandmarker.create_from_options(options)
            print("✅ Pose engine loaded")
    except Exception as e:
        print(f"❌ Pose error: {e}")
    
    try:
        if os.path.exists(h_path):
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=h_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=4
            )
            state.hand_engine = vision.HandLandmarker.create_from_options(options)
            print("✅ Hand engine loaded")
    except Exception as e:
        print(f"❌ Hand error: {e}")

# Camera capture thread
def capture_frames():
    """Capture and process frames from camera."""
    cap = cv2.VideoCapture(0)
    
    # Check if camera is available
    if not cap.isOpened():
        print("⚠️  No camera found - using demo mode")
        demo_mode = True
    else:
        demo_mode = False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        print("✅ Camera opened")
    
    state.running = True
    frame_count = 0
    
    while state.running:
        try:
            if demo_mode:
                # Create a demo frame with text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "DEMO MODE", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, "No camera available on server", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "This works on your local machine!", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame_count += 1
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Camera read failed - falling back to demo")
                    demo_mode = True
                    cap.release()
                    continue
                
                frame = cv2.flip(frame, 1)  # Mirror
                h, w = frame.shape[:2]
                
                # Run MediaPipe detection
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    
                    bodies = []
                    
                    # Pose detection
                    if state.pose_engine is not None:
                        pose_result = state.pose_engine.detect(mp_img)
                        if pose_result.pose_landmarks:
                            for i, lms in enumerate(pose_result.pose_landmarks):
                                color = PERSON_COLORS[i % len(PERSON_COLORS)]
                                face_bbox = get_face_bbox_from_pose(lms)
                                chest_bbox = get_chest_bbox_from_pose(lms)
                                
                                left_wrist = (lms[15].x, lms[15].y) if 15 < len(lms) else (0, 0)
                                right_wrist = (lms[16].x, lms[16].y) if 16 < len(lms) else (0, 0)
                                
                                bodies.append({
                                    'id': i,
                                    'face_bbox': face_bbox,
                                    'chest_bbox': chest_bbox,
                                    'wrists': [left_wrist, right_wrist],
                                    'color': color
                                })
                                
                                # Draw skeleton
                                conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                                for s, e in conns:
                                    if s < len(lms) and e < len(lms):
                                        p1 = (int(lms[s].x*w), int(lms[s].y*h))
                                        p2 = (int(lms[e].x*w), int(lms[e].y*h))
                                        cv2.line(frame, p1, p2, color, 2)
                                # Draw joints
                                for pt in lms:
                                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, color, -1)
                                    
                                # Draw Bboxes & ID
                                fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                                cv2.rectangle(frame, (fox, foy), (fox+fow, foy+foh), color, 2)
                                cx, cy, cw_rect, ch_rect = int(chest_bbox[0]*w), int(chest_bbox[1]*h), int(chest_bbox[2]*w), int(chest_bbox[3]*h)
                                cv2.rectangle(frame, (cx, cy), (cx+cw_rect, cy+ch_rect), color, 1, cv2.LINE_4)
                                cv2.putText(frame, f"P{i}", (fox, max(0, foy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    alerts = []
                    
                    # Hand detection
                    if state.hand_engine is not None:
                        hand_result = state.hand_engine.detect(mp_img)
                        if hand_result.hand_landmarks:
                            for hlms in hand_result.hand_landmarks:
                                hand_wrist_normalized = (hlms[0].x, hlms[0].y) if hlms else (0, 0)
                                min_dist = 100.0
                                owner_id = -1
                                for body in bodies:
                                    for body_wrist in body['wrists']:
                                        dist = get_distance(hand_wrist_normalized, body_wrist)
                                        if dist < min_dist:
                                            min_dist = dist
                                            owner_id = body['id']
                                
                                if min_dist > 0.2:
                                    owner_id = -1
                                    
                                color = bodies[owner_id]['color'] if owner_id != -1 else (255, 255, 255)
                                for pt in hlms:
                                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, color, -1)
                                    
                                if owner_id != -1:
                                    for landmark in hlms:
                                        pt_norm = (landmark.x, landmark.y)
                                        for other_body in bodies:
                                            if owner_id != other_body['id']:
                                                if is_point_in_rect(pt_norm, other_body['face_bbox']):
                                                    alerts.append((other_body['id'], owner_id, "Face"))
                                                    break
                                                if is_point_in_rect(pt_norm, other_body['chest_bbox']):
                                                    alerts.append((other_body['id'], owner_id, "Chest"))
                                                    break
                    
                    if alerts:
                        alert_text = "ALERTS: " + ", ".join([f"P{b} touches P{a}'s {area}!" for a, b, area in set(alerts)])
                        cv2.putText(frame, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                except Exception as e:
                    print(f"Detection error: {e}")
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Streaming... ({frame_count} frames)")
        
        except Exception as e:
            print(f"Capture error: {e}")
            break
        
        # Store frame
        with state.lock:
            state.frame = frame.copy()
    
    cap.release()
    state.running = False

# Video stream generator (MJPEG format)
def generate_frames():
    """Generate video frames for streaming."""
    while True:
        with state.lock:
            if state.frame is None:
                continue
            frame = state.frame.copy()
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve streaming page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Detection</title>
        <style>
            body { font-family: Arial; display: flex; justify-content: center; align-items: center; height: 100vh; background: #1a1a1a; margin: 0; }
            .container { text-align: center; }
            h1 { color: white; }
            img { border: 2px solid #00ff00; border-radius: 5px; max-width: 90vw; max-height: 90vh; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Live Detection Stream</h1>
            <img src="/video_feed" alt="Video Stream">
            <p style="color: #888; margin-top: 20px;">Server-side streaming (works everywhere!)</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'models_loaded': state.pose_engine is not None and state.hand_engine is not None}

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    
    print("Starting camera capture thread...")
    camera_thread = threading.Thread(target=capture_frames, daemon=True)
    camera_thread.start()
    
    print("Starting Flask server on 0.0.0.0:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
