import cv2
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
    base_options=BaseOptions(model_asset_path='./exported_model_384/model.tflite'),  # Update this path # exported_model
    max_results=1,
    running_mode=VisionRunningMode.VIDEO
)

detector = ObjectDetector.create_from_options(options)

# Open the video file
video_path = "D:/Serena/TU/Independent Study/LRV 9028_TP CAMERA 07-12-2024_1043A.mp4"
cap = cv2.VideoCapture(video_path)

# Get FPS of the video
video_file_fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0  # Initialize frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    frame_timestamp_ms = int(1000 * frame_index / video_file_fps)  # Convert frame index to timestamp

    # Convert OpenCV frame (BGR) to MediaPipe Image (RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Draw bounding boxes (if detection results exist)
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Show the video
    cv2.imshow('Object Tracking', frame)

    frame_index += 1  # Increment frame index

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
