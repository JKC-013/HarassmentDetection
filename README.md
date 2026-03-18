---
title: Multi-Person Harassment Detection
emoji: 🏃
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Multi-Person Harassment Detection

This application uses MediaPipe to detect multiple people and hands, alerting when one person's hand touches another's face or chest.

## Deployment on Hugging Face Spaces

This repository is configured to run as a Streamlit app on Hugging Face Spaces.

### Key Files:
- `app.py`: Main application logic.
- `requirements.txt`: Python dependencies.
- `packages.txt`: System-level dependencies (OpenCV).
- `*.task`: MediaPipe model files.
