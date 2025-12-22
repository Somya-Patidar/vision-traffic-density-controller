import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
# We use os.path.join to ensure paths work on both Windows and Linux
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folders if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Pre-trained YOLOv8 Nano model (lightweight and fast)
model = YOLO('yolov8n.pt')

# --- LOGIC FUNCTION ---
def get_signal_timing(vehicle_count):
    """
    Logic:
    < 10 vehicles -> 30s (LOW)
    10-25 vehicles -> 60s (MEDIUM)
    > 25 vehicles -> 90s (HIGH)
    """
    if vehicle_count < 10:
        return 30, "LOW", "success" # Green color
    elif 10 <= vehicle_count <= 25:
        return 60, "MEDIUM", "warning" # Orange color
    else:
        return 90, "HIGH", "danger" # Red color

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400

        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Run YOLOv8 detection
            # imgsz=640 is standard; conf=0.25 ignores weak detections
            results = model(filepath, conf=0.25)
            
            # Filter for vehicles: Class IDs in COCO dataset for car(2), motorcycle(3), bus(5), truck(7)
            vehicle_classes = [2, 3, 5, 7]
            count = 0
            
            # Iterate through detections and count only vehicles
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in vehicle_classes:
                        count += 1
            
            # Get timing and density label
            signal_time, density, status_color = get_signal_timing(count)

            # Pass everything to the HTML front-end
            # We pass the filename specifically for the <img> tag
            return render_template('index.html', 
                                   image_name=file.filename, 
                                   count=count, 
                                   density=density, 
                                   signal_time=signal_time,
                                   status_color=status_color)

    return render_template('index.html')

if __name__ == '__main__':
    # Running on port 5000
    app.run(debug=True)