#first
import cv2
import numpy as np
import mediapipe as mp
from IPython.display import display
import ipywidgets as widgets
from threading import Thread

# Define MediaPipe object detection options
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./model_384_int8.tflite'),  # Update this path
    max_results=1,
    running_mode=VisionRunningMode.IMAGE  # Frame-by-frame detection
)
detector = ObjectDetector.create_from_options(options)

# Global variable to control video streaming
running = True

# Function to update the image widget with video frames
def update_image_widget(image_widget):
    global running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert OpenCV image (BGR) to MediaPipe image (RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Perform object detection
        detection_result = detector.detect(mp_image)

        # Draw bounding boxes if detection results exist
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 2)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (bbox.origin_x + 10, bbox.origin_y + 20)
            cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Convert frame to JPEG format
        _, frame_jpeg = cv2.imencode('.jpeg', frame)

        # Update image widget
        image_widget.value = frame_jpeg.tobytes()

    cap.release()

# Function to stop the video capture loop
def stop_camera(button):
    global running
    running = False
    stop_button.disabled = True  # Disable the stop button once clicked

# Create an Image widget to display video frames
image_widget = widgets.Image(format='jpeg', width=640, height=480)

# Create a Button widget to stop the video capture
stop_button = widgets.Button(description="Stop Camera")
stop_button.on_click(stop_camera)

# Display the image widget and stop button
display(image_widget, stop_button)

# Start a thread to update the image widget with video frames
thread = Thread(target=update_image_widget, args=(image_widget,))
thread.start()
