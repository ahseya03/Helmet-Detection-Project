
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import numpy as np
import base64
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Making sure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Constants
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5

# Paths to configuration, weights, and class names
config_path = "cfg/yolov3-helmet.cfg"
weights_path = "weights/yolov3-helmet.weights"
class_names_path = "data/helmet.names"

# Load class names
with open(class_names_path, 'r') as f:
    class_names = f.read().strip().split('\n')

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Define random colors for classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

def model_output(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layer_names)
    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and class_names[class_id] == 'Helmet':
                box = detection[0:4] * np.array([w, h, w, h])
                (x_center, y_center, width, height) = box.astype("int")
                x_min = int(x_center - (width / 2))
                y_min = int(y_center - (height / 2))
                boxes.append([x_min, y_min, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def process_image(image):
    boxes, confidences, class_ids = model_output(image)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x_min, y_min, width, height = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_image(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    return output_video_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logging.debug(f"Saving uploaded file to {file_path}")
    file.save(file_path)
    logging.debug(f"File saved: {file_path}")

    if file and filename.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
        image = cv2.imread(file_path)
        processed_image = process_image(image)
        
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
        cv2.imwrite(processed_image_path, processed_image)
        logging.debug(f"Processed image saved to {processed_image_path}")

        _, buffer = cv2.imencode('.png', processed_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        img_data = f"data:image/png;base64,{img_str}"

        os.remove(file_path)
        return render_template('result.html', image_data=img_data)

    elif file and filename.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv']:
        processed_video_path = process_video(file_path)
        os.remove(file_path)
        return render_template('result.html', video_path=processed_video_path)

    return redirect(url_for('index'))

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


