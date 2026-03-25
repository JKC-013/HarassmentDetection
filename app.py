import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import streamlit.components.v1 as components

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Harassment Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    except Exception:
        status.append("⚠️ Pose model unavailable in this environment")
    
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
    except Exception:
        status.append("⚠️ Hand model unavailable in this environment")
    
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


# Main content
st.divider()

# Create tabs for different modes
tab1, tab2 = st.tabs(["📷 Photo Analysis", "🎥 Live Detection"])

# TAB 1: Photo-based Analysis
with tab1:
    st.subheader("📷 Photo-Based Detection - RECOMMENDED FOR SHARING")
    st.write("Take a photo and see instant pose + hand detection")
    
    st.success("""
    ✅ **Works for everyone** - no special setup needed
    ✅ **Works on any device** - phone, tablet, computer
    ✅ **Works on Render** - no server limitations
    ✅ **Quick and reliable** - instant results
    
    👉 **Share this app link with friends** - they can use this feature immediately!
    """)
    
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
                
                bodies = []
                
                # Draw pose landmarks
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
                        
                        conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                        for s, e in conns:
                            if s < len(lms) and e < len(lms):
                                p1 = (int(lms[s].x*w), int(lms[s].y*h))
                                p2 = (int(lms[e].x*w), int(lms[e].y*h))
                                cv2.line(img, p1, p2, color, 4)
                        for pt in lms:
                            cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, color, -1)
                            
                        fox, foy, fow, foh = int(face_bbox[0]*w), int(face_bbox[1]*h), int(face_bbox[2]*w), int(face_bbox[3]*h)
                        cv2.rectangle(img, (fox, foy), (fox+fow, foy+foh), color, 2)
                        cx, cy, cw_rect, ch_rect = int(chest_bbox[0]*w), int(chest_bbox[1]*h), int(chest_bbox[2]*w), int(chest_bbox[3]*h)
                        cv2.rectangle(img, (cx, cy), (cx+cw_rect, cy+ch_rect), color, 1, cv2.LINE_4)
                        cv2.putText(img, f"P{i}", (fox, max(0, foy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                alerts = []
                
                # Draw hand landmarks
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
                            cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 5, color, -1)
                            
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
                    cv2.putText(img, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
                
                # Display result
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)
                st.success("✅ Detection complete!")
            else:
                st.warning("⏳ Models still loading...")
        except Exception as e:
            st.error(f"Detection error: {e}")

# TAB 2: Browser-Side Live Detection (MediaPipe JS / WASM)
with tab2:
    st.subheader("🎥 Live Detection — Runs in Your Browser")
    st.write("Real-time pose + hand detection powered by MediaPipe WebAssembly — no server camera needed.")

    st.success("""
    ✅ **Works on Render** — detection runs in your browser, not on the server
    ✅ **30+ FPS** with GPU/WebGL acceleration
    ✅ **Privacy** — your camera feed never leaves your device
    ✅ **Works on mobile** — any browser that supports getUserMedia
    """)

    LIVE_DETECTION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Live Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117;
    color: #fff;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px;
    min-height: 100vh;
  }
  #status {
    font-size: 14px;
    color: #aaa;
    margin-bottom: 10px;
    text-align: center;
    min-height: 20px;
  }
  #alert-banner {
    display: none;
    background: #ff3333;
    color: #fff;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    margin-bottom: 10px;
    animation: pulse 0.6s infinite alternate;
  }
  @keyframes pulse { from { opacity: 1; } to { opacity: 0.6; } }
  #video-wrap {
    position: relative;
    width: 640px;
    max-width: 100%;
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid #333;
    background: #111;
  }
  video, canvas {
    width: 100%;
    display: block;
  }
  canvas {
    position: absolute;
    top: 0; left: 0;
  }
  #controls {
    margin-top: 14px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }
  button {
    padding: 10px 28px;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.1s, opacity 0.2s;
  }
  button:active { transform: scale(0.97); }
  #btn-start { background: #00c853; color: #fff; }
  #btn-stop  { background: #e53935; color: #fff; display: none; }
  #stats {
    margin-top: 10px;
    font-size: 12px;
    color: #666;
    text-align: center;
  }
</style>
</head>
<body>

<div id="status">⏳ Loading MediaPipe models, please wait…</div>
<div id="alert-banner">⚠️ HARASSMENT DETECTED — Hand near face!</div>

<div id="video-wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
</div>

<div id="controls">
  <button id="btn-start">▶ Start Camera</button>
  <button id="btn-stop">⏹ Stop</button>
</div>
<div id="stats" id="stats"></div>

<script type="module">
import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const statusEl   = document.getElementById('status');
const alertBanner = document.getElementById('alert-banner');
const video      = document.getElementById('video');
const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const btnStart   = document.getElementById('btn-start');
const btnStop    = document.getElementById('btn-stop');
const statsEl    = document.getElementById('stats');

let poseLandmarker = null;
let handLandmarker = null;
let animFrame      = null;
let streaming      = false;
let lastFrameTime  = -1;
let fpsArr         = [];

// ── Load models ──────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU"
      },
      runningMode:    "VIDEO",
      numPoses:        2,
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence:      0.5,
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode:  "VIDEO",
      numHands:      4,
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence:      0.5,
    });

    statusEl.textContent = "✅ Models ready — click Start Camera";
    btnStart.disabled = false;
  } catch (err) {
    statusEl.textContent = "❌ Model load failed: " + err.message;
    console.error(err);
  }
}

// ── Skeleton connections ──────────────────────────────────────────────────────
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]
];

// Person colours (up to 2 people)
const PERSON_COLORS = ["#00e676", "#ff6d00"];

// Hand connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ── Harassment check ─────────────────────────────────────────────────────────
// Keypoints 0-10 are face/head region in pose landmarks
const FACE_KPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const CHEST_KPS = [11, 12, 23, 24]; // Shoulders and hips
const ALERT_DIST = 0.08;  // normalised distance threshold

function getDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx*dx + dy*dy);
}

function checkHarassment(poseResult, handResult) {
  if (!poseResult || poseResult.landmarks.length < 2) return false;
  if (!handResult  || handResult.landmarks.length === 0)  return false;

  // Build bounding regions per person
  const bodies = poseResult.landmarks.map(lms => {
    let fx = 0, fy = 0, fCount = 0;
    for (const ki of FACE_KPS) {
      if (ki < lms.length) { fx += lms[ki].x; fy += lms[ki].y; fCount++; }
    }
    const faceCenter = fCount ? { x: fx/fCount, y: fy/fCount } : null;

    let cx = 0, cy = 0, cCount = 0;
    for (const ki of CHEST_KPS) {
      if (ki < lms.length) { cx += lms[ki].x; cy += lms[ki].y; cCount++; }
    }
    const chestCenter = cCount ? { x: cx/cCount, y: cy/cCount - 0.05 } : null;

    const leftWrist = lms.length > 15 ? lms[15] : {x:0, y:0};
    const rightWrist = lms.length > 16 ? lms[16] : {x:0, y:0};

    return { faceCenter, chestCenter, wrists: [leftWrist, rightWrist] };
  });

  let harassementDetected = false;

  for (let hi = 0; hi < handResult.landmarks.length; hi++) {
    const wrist = handResult.landmarks[hi][0];
    let minDist = 100.0;
    let ownerIdx = -1;
    
    for (let pi = 0; pi < bodies.length; pi++) {
      for (const bw of bodies[pi].wrists) {
        const d = getDistance(wrist, bw);
        if (d < minDist) { minDist = d; ownerIdx = pi; }
      }
    }
    if (minDist > 0.2) ownerIdx = -1;
    
    handResult.landmarks[hi].ownerIdx = ownerIdx;

    if (ownerIdx !== -1) {
      for (let pi = 0; pi < bodies.length; pi++) {
        if (ownerIdx === pi) continue;
        const b = bodies[pi];
        if (b.faceCenter && getDistance(wrist, b.faceCenter) < ALERT_DIST) harassementDetected = true;
        if (b.chestCenter && getDistance(wrist, b.chestCenter) < ALERT_DIST + 0.04) harassementDetected = true;
      }
    }
  }
  return harassementDetected;
}

// ── Draw ─────────────────────────────────────────────────────────────────────
function draw(poseResult, handResult) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw pose skeletons
  if (poseResult && poseResult.landmarks) {
    poseResult.landmarks.forEach((lms, pi) => {
      const color = PERSON_COLORS[pi % PERSON_COLORS.length];
      const W = canvas.width, H = canvas.height;

      // Connections
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      for (const [s, e] of POSE_CONNECTIONS) {
        if (s < lms.length && e < lms.length) {
          ctx.beginPath();
          ctx.moveTo(lms[s].x * W, lms[s].y * H);
          ctx.lineTo(lms[e].x * W, lms[e].y * H);
          ctx.stroke();
        }
      }
      // Joints
      ctx.fillStyle = color;
      for (const lm of lms) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI*2);
        ctx.fill();
      }
    });
  }

  // Draw hands
  if (handResult && handResult.landmarks) {
    handResult.landmarks.forEach(hlms => {
      const W = canvas.width, H = canvas.height;
      const ownerIdx = hlms.ownerIdx;
      const color = (ownerIdx !== undefined && ownerIdx !== -1) 
        ? PERSON_COLORS[ownerIdx % PERSON_COLORS.length] 
        : "#ffffff";
        
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [s, e] of HAND_CONNECTIONS) {
        if (s < hlms.length && e < hlms.length) {
          ctx.beginPath();
          ctx.moveTo(hlms[s].x * W, hlms[s].y * H);
          ctx.lineTo(hlms[e].x * W, hlms[e].y * H);
          ctx.stroke();
        }
      }
      ctx.fillStyle = color;
      for (const lm of hlms) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 3, 0, Math.PI*2);
        ctx.fill();
      }
    });
  }
}

// ── Detection loop ────────────────────────────────────────────────────────────
function detectLoop(now) {
  if (!streaming) return;
  animFrame = requestAnimationFrame(detectLoop);

  if (video.readyState < 2) return;

  // Sync canvas size to video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const ts = now;
  if (ts === lastFrameTime) return;
  lastFrameTime = ts;

  let poseResult = null, handResult = null;
  try {
    poseResult = poseLandmarker.detectForVideo(video, ts);
    handResult = handLandmarker.detectForVideo(video, ts);
  } catch (e) { return; }

  // Harassment alert (also assigns ownerIdx to hands)
  const alert = checkHarassment(poseResult, handResult);
  alertBanner.style.display = alert ? 'block' : 'none';

  draw(poseResult, handResult);

  // FPS counter
  fpsArr.push(ts);
  fpsArr = fpsArr.filter(t => ts - t < 1000);
  statsEl.textContent = "FPS: " + fpsArr.length +
    " | Poses: " + (poseResult ? poseResult.landmarks.length : 0) +
    " | Hands: " + (handResult ? handResult.landmarks.length : 0);
}

// ── Camera ───────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  try {
    statusEl.textContent = "📷 Requesting camera access…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    streaming = true;
    btnStart.style.display = 'none';
    btnStop.style.display  = 'inline-block';
    statusEl.textContent   = "🟢 Live — detecting…";
    animFrame = requestAnimationFrame(detectLoop);
  } catch (err) {
    statusEl.textContent = "❌ Camera error: " + err.message;
  }
});

btnStop.addEventListener('click', () => {
  streaming = false;
  if (animFrame) cancelAnimationFrame(animFrame);
  if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  alertBanner.style.display = 'none';
  btnStart.style.display = 'inline-block';
  btnStop.style.display  = 'none';
  statusEl.textContent   = "⏹ Stopped";
  statsEl.textContent    = "";
});

// ── Init ─────────────────────────────────────────────────────────────────────
btnStart.disabled = true;
loadModels();
</script>
</body>
</html>
"""

    components.html(LIVE_DETECTION_HTML, height=680, scrolling=False)
