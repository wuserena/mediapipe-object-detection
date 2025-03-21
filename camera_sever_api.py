# Run the code on rasiberry pi

import cv2
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    # Open the camera once instead of reinitializing in the loop
    camera = cv2.VideoCapture(0)
    # Set resolution to avoid errors
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break  # Exit the loop if the camera fails

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
    finally:
        camera.release()  # Release the camera when the app stops
