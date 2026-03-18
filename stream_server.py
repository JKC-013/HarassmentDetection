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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    state.running = True
    frame_count = 0
    
    while state.running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        h, w = frame.shape[:2]
        
        # Run MediaPipe detection
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Pose detection
            if state.pose_engine is not None:
                pose_result = state.pose_engine.detect(mp_img)
                if pose_result.pose_landmarks:
                    for lms in pose_result.pose_landmarks:
                        # Draw skeleton
                        conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                        for s, e in conns:
                            if s < len(lms) and e < len(lms):
                                p1 = (int(lms[s].x*w), int(lms[s].y*h))
                                p2 = (int(lms[e].x*w), int(lms[e].y*h))
                                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                        # Draw joints
                        for pt in lms:
                            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (0, 255, 0), -1)
            
            # Hand detection
            if state.hand_engine is not None:
                hand_result = state.hand_engine.detect(mp_img)
                if hand_result.hand_landmarks:
                    for hlms in hand_result.hand_landmarks:
                        for pt in hlms:
                            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (255, 255, 255), -1)
        except Exception as e:
            print(f"Detection error: {e}")
        
        # Add FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Streaming... ({frame_count} frames)")
        
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
