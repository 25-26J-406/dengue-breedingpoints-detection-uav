from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = YOLO('weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    # Save the file with a unique name
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run Model
    results = model.predict(source=filepath, conf=0.25)
    result = results[0]

    # If nothing found
    if len(result.boxes) == 0:
        return jsonify({"message": "No waste detected."})

    # Save Annotated Image
    output_filename = "res_" + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    res_plotted = result.plot()
    cv2.imwrite(output_path, res_plotted)

    # Prepare JSON Data
    detections = []
    for box in result.boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0])
        })

    return jsonify({
        "image_url": f"/static/uploads/{output_filename}",
        "detections": detections
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)