import os
import cv2
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. LOAD MODEL GLOBALLY (Saves RAM by not reloading)
model = YOLO('yolov8s.pt')

def get_signal_timing(vehicle_count):
    if vehicle_count < 10:
        return 30, "LOW", "success"
    elif 10 <= vehicle_count <= 25:
        return 60, "MEDIUM", "warning"
    else:
        return 90, "HIGH", "danger"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # iou=0.5 (default is 0.7). Lowering it helps with overlapping vehicles.
            results = model(filepath, conf=0.15, imgsz=640, iou=0.5)
            
            vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
            count = 0
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in vehicle_classes:
                        count += 1
            
            signal_time, density, status_color = get_signal_timing(count)

            return render_template('index.html', 
                                   image_name=file.filename, 
                                   count=count, 
                                   density=density, 
                                   signal_time=signal_time,
                                   status_color=status_color)

    return render_template('index.html')

if __name__ == '__main__':
    # Local run uses port 5000
    app.run(debug=True)