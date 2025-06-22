#!/usr/bin/env python3
"""
Video Summariser Startup Script
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required packages are installed"""
    # Mapping of package name (for pip) to import name (for python)
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'python-multipart': 'multipart',
        'moviepy': 'moviepy',
        'opencv-python': 'cv2',
        'transformers': 'transformers',
        'torch': 'torch',
        'Pillow': 'PIL',
        'openai-whisper': 'whisper'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please run the following command to install them:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("Video Summariser")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/frames", exist_ok=True)
    
    print("Starting server...")
    print("Open your browser and go to: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Import and run the FastAPI app
        from main_api import app
        import uvicorn
        
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 