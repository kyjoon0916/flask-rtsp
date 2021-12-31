# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
# Initialize the Flask app
app = Flask(__name__)
# Model
# model = torch.hub.load('../../yolov5', 'custom',
#                        path='../../model/weights/yolov5s.pt', source='local')  # local repo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# camera = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")
# camera = cv2.VideoCapture(
#     "rtsp://192.168.50.153:8554/test")
camera = cv2.VideoCapture(
    0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
threshold = 0.5


def inference(frame):

    # Inference
    results = model(frame)
    img = np.array(results.imgs)[0]
    names = results.names

    # Adding Box
    # xyxy의 결과값은 startx, starty, endx, endy, confidenc, class
    boxes = results.xyxy[0].cpu().numpy()
    # print(boxes.shape)

    for box in boxes:
        if box[4] > threshold:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (0, 0, 255), 2)
            title = names[int(box[5])]
            prob = round(box[4] * 100, 1)
            img = cv2.putText(img, title + ' ' + str(prob) + '%', (int(box[0]), int(box[1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    return img


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', inference(frame))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
