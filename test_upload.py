#!/usr/bin/env python3
"""
Test script for the video upload endpoint
"""

import requests
import os

def test_server():
    """Test if the server is running and the upload endpoint works"""
    
    # Test if server is running
    try:
        response = requests.get("http://127.0.0.1:8000/")
        print(f"Server status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Server is running correctly")
        else:
            print("❌ Server returned unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    
    # Test upload endpoint with a dummy file
    try:
        # Create a dummy video file for testing
        dummy_video_path = "test_video.mp4"
        with open(dummy_video_path, "wb") as f:
            f.write(b"dummy video content")
        
        with open(dummy_video_path, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            response = requests.post("http://127.0.0.1:8000/upload", files=files)
        
        print(f"Upload endpoint status: {response.status_code}")
        print(f"Upload response: {response.text[:200]}...")
        
        # Clean up dummy file
        if os.path.exists(dummy_video_path):
            os.remove(dummy_video_path)
            
        if response.status_code == 200:
            print("✅ Upload endpoint is working")
            return True
        else:
            print("❌ Upload endpoint returned error")
            return False
            
    except Exception as e:
        print(f"❌ Error testing upload: {e}")
        return False

if __name__ == "__main__":
    print("Testing Video Summariser Server...")
    print("=" * 40)
    
    if test_server():
        print("\n✅ All tests passed! The server is working correctly.")
        print("You can now use the web interface at http://127.0.0.1:8000")
    else:
        print("\n❌ Some tests failed. Check the server logs for more details.") 