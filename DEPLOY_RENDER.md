# 🚀 Deployment Guide for Render.com

This guide walks you through deploying the Harassment Detection AI app on Render.com.

## Why Render?

✅ **Free tier available** (with some limitations)  
✅ **WebRTC works** (unlike HF Spaces)  
✅ **Easy GitHub integration** (auto-deploy on push)  
✅ **No credit card required** to start  
✅ **Scales easily** when needed  

## Step-by-Step Deployment

### 1. Create Render Account

- Go to [render.com](https://render.com)
- Sign up with GitHub (recommended)
- Authorize Render to access your GitHub account

### 2. Create New Web Service

- Click **"New +"** button (top right)
- Select **"Web Service"**
- Click **"Connect a repository"**
- Find and select **`HarassmentDetection`** (or your repo name)
- Click **"Connect"**

### 3. Configure Service

Fill out the form:

| Field | Value |
|-------|-------|
| **Name** | `harassment-detection` (or your choice) |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
| **Instance Type** | `Free` or `Standard` ($7/month) |
| **Auto-deploy** | Toggle ON |

### 4. Deploy

- Click **"Create Web Service"**
- Wait 3-5 minutes for first deploy
- You'll see a URL like: `https://harassment-detection.onrender.com`
- Click the URL to access your app!

### 5. Test

Once deployed:
1. Open the URL in your browser
2. Click **"START"** button
3. Allow camera permissions
4. Check if video stream appears

## Troubleshooting Render Deployment

### "Build failed"

**Problem**: Dependencies didn't install  
**Solution**: 
- Check `requirements.txt` for typos
- Ensure all packages are compatible
- Try: `pip freeze > requirements.txt` locally first

### "Camera still times out"

**Problem**: WebRTC connection issues  
**Solution**:
- Make sure you're using **Render**, not HF Spaces
- Try different browser (Chrome recommended)
- Check browser console (F12) for errors
- Refresh page and try again

### "Slow/high latency"

**Problem**: App too slow for real-time  
**Solution**:
- Upgrade to `Standard` instance ($7/month) for better CPU
- Close other browser tabs
- Use wired internet (not WiFi)
- Consider deploying to **Railway.app** instead (similar process)

## Cost Comparison

| Platform | Free Tier | Paid | WebRTC |
|----------|-----------|------|--------|
| **Render** | Yes (limited) | $7-25/mo | ✅ Works |
| **Railway** | $5 credits/mo | Pay as you go | ✅ Works |
| **DigitalOcean** | No | $6+/mo | ✅ Works |
| **AWS EC2** | 12 months free | Variable | ✅ Works |
| **HF Spaces** | Yes | Yes | ❌ Blocked |

## Auto-Deployment

After deploying once, Render will **automatically redeploy** whenever you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update app"
git push origin main

# Render automatically deploys! (takes 1-2 minutes)
```

## Advanced: Custom Domain

Want to use your own domain?

1. Go to your Render dashboard
2. Select your service
3. Go to **"Settings"** → **"Custom Domain"**
4. Add your domain
5. Point DNS to Render (instructions provided)

## Performance Optimization Tips

For better performance on Render:

1. **Upgrade to Standard Instance** ($7/month)
   - Better CPU performance
   - More reliable connection

2. **Enable Caching**
   - Models are cached after first load
   - Subsequent requests are faster

3. **Monitor Logs**
   - Render dashboard shows live logs
   - Check for errors: **Logs** tab

## Alternative: Deploy to Railway.app

Railway has a similar setup:

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"**
3. Select **"GitHub Repo"**
4. Connect repo
5. Add environment variable: `PORT=5173`
6. Deploy!

Railway is often faster and cheaper, with better WebRTC support.

## Need Help?

- **Render Support**: docs.render.com
- **Streamlit Docs**: docs.streamlit.io
- **WebRTC Issues**: Check browser console (F12)
- **Model Issues**: Check `requirements.txt` versions

## Next Steps

After successful deployment:

1. ✅ Share app URL with others
2. ✅ Monitor performance on Render dashboard
3. ✅ Collect feedback and improve
4. ✅ Consider upgrading to Standard if needed
5. ✅ Add custom domain (optional)

Good luck! 🚀
