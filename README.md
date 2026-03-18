# 🛡️ Harassment Detection AI

Real-time pose and hand detection using MediaPipe and Streamlit. Detects people and their hand positions for harassment prevention.

## Features

✅ **Real-time Detection** - Live camera stream processing  
✅ **Multi-person Support** - Detects up to 2 people simultaneously  
✅ **Hand Tracking** - Tracks up to 4 hands per frame  
✅ **Low Latency** - ~20 FPS processing speed  
✅ **Easy Deployment** - Deploy on Render, Railway, or any Docker-compatible platform  

## Tech Stack

- **Frontend**: Streamlit + WebRTC
- **ML Models**: MediaPipe (Pose + Hand)
- **Video Processing**: OpenCV
- **Deployment**: Docker

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser to http://localhost:8501
```

## Deployment on Render

### Prerequisites
- GitHub account with this repo
- Render.com account (free tier available)

### Steps

1. **Connect Repository**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - **Name**: `harassment-detection`
   - **Environment**: `Python 3.11`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Instance Type**: `Standard` ($7/month) or `Free` (limited)

3. **Deploy**
   - Click "Create Web Service"
   - Wait 3-5 minutes for deploy
   - Access at `https://<service-name>.onrender.com`

## Usage

1. Click **START** to enable camera
2. **Grant camera permission** when prompted
3. View **real-time detection** with pose and hand landmarks
4. Landmarks appear as connected points on video

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera won't open | Refresh page, try Chrome/Firefox, check permissions |
| Connection timeout | Browser permissions issue, try different browser |
| Black screen | Wait 2-3 seconds after granting permission |
| High latency | Close other tabs, use wired connection |

## Model Info

- **Pose**: 33 landmarks per person (full body)
- **Hands**: Multi-hand detection (up to 4 hands)
- **FPS**: ~20 frames per second
- **Input**: 640x480 video stream

## Privacy

✅ All processing on device/server  
✅ No data storage  
✅ No external transmission  
✅ No analytics  

## File Structure

```
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── Procfile              # Deployment config
├── .streamlit/config.toml # Streamlit settings
├── Dockerfile            # Docker image
└── README.md             # This file
```

## Browser Support

- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ⚠️ Safari (Limited WebRTC support)

## License

MIT - Open source and free to use

## Support

Issues? Check:
1. Browser console (F12) for errors
2. Server logs on Render dashboard
3. Try Chrome browser
4. Test camera with native app first
