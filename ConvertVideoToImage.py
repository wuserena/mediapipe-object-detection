# Retrieve from: Converting Video to Images Using Python and OpenCV
# https://medium.com/@wayandadangunsri/converting-video-to-images-using-python-and-opencv-72b2ea66a692

import cv2
import os

# Function to extract frames from a video until reaching the desired frame count
def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)

    frame_rate = 1  # 1 frame every 0.5 seconds
    frame_count = 0

    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_directory}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = r"LRV 9028_TP CAMERA 07-12-2024_1111A.mp4"  # Replace with your video's name

    extract_frames(video_file)