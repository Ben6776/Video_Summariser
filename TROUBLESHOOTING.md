# Troubleshooting Guide

## "Failed to Fetch" Error

If you're getting a "failed to fetch" error when uploading a video, here are the steps to resolve it:

### 1. Check if the server is running
Open your browser and go to: `http://127.0.0.1:8000/health`
You should see: `{"status": "healthy", "message": "Video Summariser is running"}`

### 2. Check browser console for errors
1. Open your browser's Developer Tools (F12)
2. Go to the Console tab
3. Try uploading a video
4. Look for any error messages in the console

### 3. Common causes and solutions

#### A. Server not running
**Symptoms**: Cannot access `http://127.0.0.1:8000`
**Solution**: Start the server:
```bash
python main_api.py
```

#### B. Port already in use
**Symptoms**: "Address already in use" error
**Solution**: 
1. Find and kill the process using port 8000:
   ```bash
   netstat -ano | findstr :8000
   taskkill /F /PID <PID>
   ```
2. Restart the server

#### C. Video file too large
**Symptoms**: Upload times out or fails
**Solution**: Try with a smaller video file (under 100MB)

#### D. Unsupported video format
**Symptoms**: Processing error after upload
**Solution**: Use common formats like MP4, AVI, MOV

#### E. Missing dependencies
**Symptoms**: Import errors in server logs
**Solution**: Install missing packages:
```bash
pip install -r requirements.txt
```

### 4. Debug mode

The application now includes debug information. When you upload a video:
1. Check the browser console for detailed logs
2. Look for the debug information displayed on the page
3. Check the server console for processing logs

### 5. Test with a simple video

Try uploading a short (10-30 seconds) MP4 video with clear audio to test the system.

### 6. Check system resources

The application requires:
- At least 4GB RAM
- Sufficient disk space for temporary files
- CPU processing power (GPU recommended but not required)

### 7. Alternative testing

Run the test script to verify the server:
```bash
python test_upload.py
```

## Still having issues?

1. Check the server logs for detailed error messages
2. Try restarting the server
3. Clear your browser cache
4. Try a different browser
5. Check if your antivirus is blocking the connection

## Getting help

If you're still experiencing issues, please provide:
1. The exact error message from the browser console
2. The server logs (if any)
3. Your video file format and size
4. Your system specifications 