import cv2
import requests
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 20  # pixels
ROW_SIZE = 20  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255)  # red

# Define MediaPipe object detection options
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./exported_model_384/model.tflite'),  # Update this path
    max_results=1,
    running_mode=VisionRunningMode.IMAGE  # Change to IMAGE mode for frame-by-frame detection
)
detector = ObjectDetector.create_from_options(options)

# URL of the Raspberry Pi video feed
URL = f"http:/10.109.17.224:5000/video_feed"

# Open the video stream using OpenCV
cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert OpenCV image (BGR) to MediaPipe image (RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    detection_result = detector.detect(mp_image)

    # Draw bounding boxes if detection results exist
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (bbox.origin_x + MARGIN, bbox.origin_y + MARGIN + ROW_SIZE)
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Display the frame
    cv2.imshow('Raspberry Pi Stream', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
