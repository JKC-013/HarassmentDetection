import streamlit as st
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

# Main content
st.divider()

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
    // Remove strict cutoff (if missing wrist is e.g. 1.0 apart, it's fine. We map to closest person)
    if (minDist > 0.6) ownerIdx = -1; // Keep a generous fallback to unowned only if absurdly far
    
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
