from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil
from typing import List, Optional
import logging

from moviepy.editor import VideoFileClip
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import whisper
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Directories and paths
UPLOAD_DIR = "temp"
VIDEO_PATH = os.path.join(UPLOAD_DIR, "video.mp4")
AUDIO_PATH = os.path.join(UPLOAD_DIR, "audio.wav")
FRAMES_DIR = os.path.join(UPLOAD_DIR, "frames")
SUMMARY_PATH = os.path.join(UPLOAD_DIR, "summary.txt")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure necessary directories exist
os.makedirs(FRAMES_DIR, exist_ok=True)

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists(VIDEO_PATH):
            os.remove(VIDEO_PATH)
        if os.path.exists(AUDIO_PATH):
            os.remove(AUDIO_PATH)
        if os.path.exists(SUMMARY_PATH):
            os.remove(SUMMARY_PATH)
        
        # Clean up frames
        if os.path.exists(FRAMES_DIR):
            for f in os.listdir(FRAMES_DIR):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(FRAMES_DIR, f))
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

def validate_video_file(file_path: str) -> bool:
    """Validate if the video file is properly formatted and readable"""
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.error(f"File {file_path} does not exist or is empty")
            return False
        
        # Try to open with OpenCV first (faster check)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"OpenCV cannot open file {file_path}")
            return False
        
        # Check if we can read at least one frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.error(f"Cannot read frames from {file_path}")
            return False
        
        # Additional check with MoviePy for duration
        try:
            video = VideoFileClip(file_path)
            duration = video.duration
            video.close()
            
            if duration <= 0:
                logger.error(f"Video duration is invalid: {duration}")
                return False
                
            logger.info(f"Video validation successful. Duration: {duration:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy validation failed: {e}")
            # If MoviePy fails but OpenCV works, we can still try to process
            # This handles some edge cases where MoviePy is overly strict
            logger.info("Falling back to OpenCV-only processing")
            return True
            
    except Exception as e:
        logger.error(f"Video validation error: {e}")
        return False

def extract_audio_and_frames(video_path: str):
    try:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        for f in os.listdir(FRAMES_DIR):
            if f.endswith('.jpg'):
                os.remove(os.path.join(FRAMES_DIR, f))

        has_audio = False
        
        # Try MoviePy first
        try:
            video = VideoFileClip(video_path)
            if video.audio is not None:
                try:
                    video.audio.write_audiofile(AUDIO_PATH, verbose=False, logger=None)
                    has_audio = True
                except Exception as e:
                    logger.warning(f"Could not extract audio with MoviePy: {e}")
            video.close()
        except Exception as e:
            logger.warning(f"MoviePy failed, using OpenCV fallback: {e}")

        # Use OpenCV for frame extraction (more reliable)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file with OpenCV")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        interval = max(1, int(fps * 2))  # one frame every 2 seconds, minimum 1
        count = 0
        frame_idx = 0
        max_frames = 10  # Limit to prevent memory issues

        while cap.isOpened() and count < max_frames:
            success, frame = cap.read()
            if not success:
                break
            if frame_idx % interval == 0:
                frame_path = os.path.join(FRAMES_DIR, f"frame_{count:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                count += 1
            frame_idx += 1

        cap.release()
        
        if count == 0:
            raise Exception("No frames could be extracted from the video")
            
        logger.info(f"Extracted {count} frames successfully")
        return has_audio
        
    except Exception as e:
        logger.error(f"Error extracting audio and frames: {e}")
        raise

def transcribe_audio(audio_path: str) -> str:
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

def generate_captions(frames_dir: str, max_frames=4) -> List[str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        captions = []
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])[:max_frames]
        
        if not files:
            return ["No visual content detected"]
            
        for fname in files:
            try:
                image_path = os.path.join(frames_dir, fname)
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)
                captions.append(caption)
            except Exception as e:
                logger.error(f"Error processing frame {fname}: {e}")
                captions.append("Frame processing error")

        return captions
    except Exception as e:
        logger.error(f"Error generating captions: {e}")
        return ["Error generating visual descriptions"]

def generate_summary(text: str) -> str:
    try:
        if not text.strip():
            return "No content available for summarization"
            
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary"

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Clean up any existing files
        cleanup_temp_files()
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Validate file type
        if not file.content_type.startswith('video/'):
            return JSONResponse({"error": "File must be a video"}, status_code=400)

        # Log file information for debugging
        logger.info(f"Uploading file: {file.filename}, size: {file.size}, type: {file.content_type}")

        # Save the uploaded file
        with open(VIDEO_PATH, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"File saved to {VIDEO_PATH}")

        # Check if file was saved correctly
        if not os.path.exists(VIDEO_PATH) or os.path.getsize(VIDEO_PATH) == 0:
            return JSONResponse({"error": "Failed to save video file"}, status_code=500)

        # Validate the video file before processing
        logger.info("Validating video file...")
        if not validate_video_file(VIDEO_PATH):
            return JSONResponse({
                "error": "Invalid or corrupted video file. Please ensure the file is a valid video format (MP4, AVI, MOV, etc.) and is not corrupted."
            }, status_code=400)

        logger.info("Starting video processing...")
        has_audio = extract_audio_and_frames(VIDEO_PATH)
        logger.info(f"Audio extraction completed. Has audio: {has_audio}")

        transcript = ""
        if has_audio:
            logger.info("Starting audio transcription...")
            transcript = transcribe_audio(AUDIO_PATH)
            logger.info(f"Transcription completed. Length: {len(transcript)}")

        logger.info("Starting visual analysis...")
        captions = generate_captions(FRAMES_DIR)
        logger.info(f"Visual analysis completed. Captions: {len(captions)}")

        combined = ""
        if transcript:
            combined += "Transcript: " + transcript + ". "
        if captions:
            combined += "Visuals: " + " ".join(captions) + "."

        logger.info(f"Combined content length: {len(combined)}")

        summary = generate_summary(combined) if combined else "No summary could be generated."
        logger.info(f"Summary generated: {len(summary)} characters")

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(summary)

        logger.info("Upload processing completed successfully")
        return {"summary": summary, "video_url": "/video"}

    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)

@app.get("/video")
def serve_video():
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type="video/mp4")
    return JSONResponse({"error": "Video not found"}, status_code=404)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Video Summariser is running"}

@app.get("/")
def index():
    return HTMLResponse("""
    <html><body>
    <h2>Video Summariser</h2>
    <form id="upload-form" enctype="multipart/form-data">
      <input type="file" name="file" accept="video/*" required />
      <button type="submit">Upload</button>
    </form>
    <br/>
    <video id="video" width="480" controls style="display:none;"></video>
    <h3>Summary:</h3>
    <div id="summary"></div>
    <div id="debug" style="color: red; font-size: 12px;"></div>
    <script>
    const form = document.getElementById('upload-form');
    const video = document.getElementById('video');
    const summaryDiv = document.getElementById('summary');
    const debugDiv = document.getElementById('debug');

    form.onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        summaryDiv.innerText = "Processing...";
        debugDiv.innerText = "";
        
        try {
            console.log('Sending request to /upload...');
            const res = await fetch('/upload', { 
                method: 'POST', 
                body: formData 
            });
            
            console.log('Response status:', res.status);
            console.log('Response headers:', res.headers);
            
            const data = await res.json();
            console.log('Response data:', data);
            
            if (res.ok) {
                if (data.video_url) {
                    video.src = data.video_url;
                    video.style.display = 'block';
                }
                summaryDiv.innerText = data.summary || 'Summary generated successfully';
                summaryDiv.style.color = 'black';
            } else {
                // Handle different error types
                let errorMessage = data.error || 'Unknown error occurred';
                
                if (res.status === 400) {
                    if (errorMessage.includes('File must be a video')) {
                        errorMessage = 'Please select a valid video file (MP4, AVI, MOV, etc.)';
                    } else if (errorMessage.includes('corrupted')) {
                        errorMessage = 'The video file appears to be corrupted. Please try a different video file.';
                    }
                } else if (res.status === 500) {
                    errorMessage = 'Server error occurred while processing the video. Please try again.';
                }
                
                summaryDiv.innerText = 'Error: ' + errorMessage;
                summaryDiv.style.color = 'red';
                debugDiv.innerText = 'Debug: HTTP ' + res.status + ' - ' + data.error;
            }
            
        } catch (error) {
            console.error('Fetch error:', error);
            debugDiv.innerText = 'Debug: ' + error.message;
            
            // Additional debugging info
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                summaryDiv.innerText = 'Network error: Please check if the server is running on http://127.0.0.1:8000';
            } else {
                summaryDiv.innerText = 'Error: ' + error.message;
            }
            summaryDiv.style.color = 'red';
        }
    };
    </script>
    </body></html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
