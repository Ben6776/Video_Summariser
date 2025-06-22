from moviepy import VideoFileClip

clip = VideoFileClip("testvideo.mp4")  # replace with your video filename
print(f"Video duration: {clip.duration} seconds")

