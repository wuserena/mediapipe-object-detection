#Retrieve from: https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/raspberry_pi
# Copyright 2023 The MediaPipe Authors. All Rights Reserved.

"""Raspberry Pi MediaPipe Object Detection with HTTP Streaming"""

import argparse
import sys
import time
import threading
import cv2
import mediapipe as mp
from flask import Flask, Response, render_template_string

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize

# Global variables for FPS calculation
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Global for streaming
output_frame = None
lock = threading.Lock()

# Flask app
app = Flask(__name__)

HTML_PAGE = """
<html>
<head><title>Raspberry Pi Object Detection Stream</title></head>
<body>
<h1>Live Object Detection Stream</h1>
<img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')

def run_detection(model: str, max_results: int, score_threshold: float,
                  camera_id: int, width: int, height: int) -> None:
    global output_frame, lock, COUNTER, FPS, START_TIME

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        sys.exit('ERROR: Unable to open camera.')

    detection_result_list = []
    fps_avg_frame_count = 10

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global COUNTER, FPS, START_TIME

        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        detection_result_list.append(result)
        COUNTER += 1

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results,
                                           score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    print("🟣 Streaming started at: http://<your_pi_ip>:5000")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Camera read failed")
            continue

        #image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        cv2.putText(image, fps_text, (24, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        if detection_result_list:
            image = visualize(image, detection_result_list[0])
            detection_result_list.clear()

        with lock:
            output_frame = image.copy()

    cap.release()
    detector.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet.tflite')
    parser.add_argument('--maxResults', default=5, type=int)
    parser.add_argument('--scoreThreshold', default=0.25, type=float)
    parser.add_argument('--cameraId', default=0, type=int)
    parser.add_argument('--frameWidth', default=1280, type=int)
    parser.add_argument('--frameHeight', default=720, type=int)
    args = parser.parse_args()

    # Start detector thread
    t = threading.Thread(target=run_detection, args=(args.model, args.maxResults, args.scoreThreshold,
                                                     args.cameraId, args.frameWidth, args.frameHeight))
    t.daemon = True
    t.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

if __name__ == '__main__':
    main()
