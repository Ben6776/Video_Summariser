# Video Summariser

A FastAPI-based web application that automatically generates summaries of uploaded videos by analyzing both audio (transcription) and visual content (frame captions).

## Features

- **Audio Transcription**: Uses OpenAI Whisper to transcribe video audio
- **Visual Analysis**: Extracts frames from video and generates captions using BLIP model
- **Smart Summarization**: Combines audio and visual content to create comprehensive summaries
- **Web Interface**: Simple HTML interface for uploading and viewing results
- **Error Handling**: Robust error handling and logging throughout the application

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. **Clone or download the project files**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Method 1: Using the startup script (Recommended)
```bash
python run.py
```

### Method 2: Direct execution
```bash
python main_api.py
```

### Method 3: Using uvicorn directly
```bash
uvicorn main_api:app --host 127.0.0.1 --port 8000
```

## Accessing the Application

1. Open your web browser
2. Navigate to: `http://127.0.0.1:8000`
3. Upload a video file using the web interface
4. Wait for processing (this may take a few minutes depending on video length)
5. View the generated summary and uploaded video

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload video file for processing
- `GET /video`: Serve the uploaded video file

## How It Works

1. **Video Upload**: User uploads a video file through the web interface
2. **Audio Extraction**: The application extracts audio from the video
3. **Frame Extraction**: Key frames are extracted from the video (one frame every 2 seconds)
4. **Audio Transcription**: Whisper model transcribes the audio to text
5. **Visual Captioning**: BLIP model generates captions for the extracted frames
6. **Content Combination**: Audio transcript and visual captions are combined
7. **Summarization**: BART model creates a final summary of the combined content
8. **Result Display**: Summary and video are displayed to the user

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: If you get import errors, make sure all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Issues**: If you have CUDA issues, the application will automatically fall back to CPU processing

3. **Memory Issues**: For large videos, you might need more RAM. The application processes videos in chunks to minimize memory usage

4. **Port Already in Use**: If port 8000 is busy, you can change it in the code or use a different port:
   ```bash
   uvicorn main_api:app --host 127.0.0.1 --port 8001
   ```

### Performance Tips

- Use a GPU for faster processing (CUDA-compatible)
- For very long videos, consider splitting them into smaller segments
- The application automatically limits frame extraction to prevent memory issues

## File Structure

```
video_summariser/
├── main_api.py          # Main FastAPI application
├── run.py              # Startup script with dependency checking
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── temp/              # Temporary files (created automatically)
│   └── frames/        # Extracted video frames
└── venv/              # Virtual environment (created during setup)
```

## Technical Details

- **Backend**: FastAPI with Uvicorn ASGI server
- **Audio Processing**: OpenAI Whisper for transcription
- **Image Processing**: OpenCV for frame extraction
- **Visual Analysis**: BLIP (Bootstrapping Language-Image Pre-training) for image captioning
- **Text Summarization**: BART (Bidirectional and Auto-Regressive Transformers) for text summarization
- **Frontend**: Simple HTML with JavaScript for file upload and display

## License

This project is open source and available under the MIT License. 