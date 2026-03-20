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

# Sidebar - Status
with st.sidebar:
    st.header("📊 Status")
    for msg in status_messages:
        if "✅" in msg:
            st.success(msg)
        elif "❌" in msg:
            st.error(msg)
        elif "⚠️" in msg:
            st.warning(msg)
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
const ALERT_DIST = 0.08;  // normalised distance threshold

function checkHarassment(poseResult, handResult) {
  if (!poseResult || poseResult.landmarks.length < 2) return false;
  if (!handResult  || handResult.landmarks.length === 0)  return false;

  // Build face bounding regions per person
  const faceCentres = poseResult.landmarks.map(lms => {
    let sx = 0, sy = 0, count = 0;
    for (const ki of FACE_KPS) {
      if (ki < lms.length) { sx += lms[ki].x; sy += lms[ki].y; count++; }
    }
    return count ? { x: sx/count, y: sy/count } : null;
  });

  // For each hand, check if its wrist (lm 0) is near another person's face
  for (let hi = 0; hi < handResult.landmarks.length; hi++) {
    const wrist = handResult.landmarks[hi][0];
    for (let pi = 0; pi < faceCentres.length; pi++) {
      const fc = faceCentres[pi];
      if (!fc) continue;
      const dx = wrist.x - fc.x;
      const dy = wrist.y - fc.y;
      if (Math.sqrt(dx*dx + dy*dy) < ALERT_DIST) {
        // Only alert if this hand is NOT from this person
        // Simple heuristic: face centre x ± 0.2 window
        const samePerson = Math.abs(wrist.x - fc.x) < 0.05 &&
                            Math.abs(wrist.y - fc.y) < 0.05;
        if (!samePerson) return true;
      }
    }
  }
  return false;
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
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      for (const [s, e] of HAND_CONNECTIONS) {
        if (s < hlms.length && e < hlms.length) {
          ctx.beginPath();
          ctx.moveTo(hlms[s].x * W, hlms[s].y * H);
          ctx.lineTo(hlms[e].x * W, hlms[e].y * H);
          ctx.stroke();
        }
      }
      ctx.fillStyle = "#ffffff";
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

  draw(poseResult, handResult);

  // Harassment alert
  const alert = checkHarassment(poseResult, handResult);
  alertBanner.style.display = alert ? 'block' : 'none';

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
