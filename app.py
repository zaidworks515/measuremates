import os
from flask import Flask, request, render_template, g, send_from_directory, make_response
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from flask_cors import cross_origin
import numpy as np


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/image'),'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.before_request
def load_model():
    model = YOLO(os.path.join(app.root_path, 'static/model/last.pt'))
    g.model = model

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['GET', 'POST'])
@cross_origin()
def detect_objects():
    model = getattr(g, 'model', None)

    if model is None:
        return {'error': 'Model not loaded'}

    if request.method == 'POST':
        try:
            loaded_image = request.files['image']

            # Read the image from the BytesIO object
            image_stream = loaded_image.stream
            image_bytes = bytearray(image_stream.read())
            image_np = cv2.imdecode(np.asarray(image_bytes), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(source=image_np, save=False, conf=0.6, task='pose')

            pred_image = results[0].plot()
            pred_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

            # Convert images to base64 format
            input_image_base64 = encode_image(image_rgb)
            output_image_base64 = encode_image(pred_rgb)
            
            # Store images in Flask context (g object) as they will be used in download_pdf function
            g.input_image_base64 = input_image_base64
            g.output_image_base64 = output_image_base64


            return render_template('result.html', input_image = input_image_base64, output_image = output_image_base64)

        except Exception as e:
            return {'error': str(e)}

    else:
        return render_template('index.html')


def encode_image(image):
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', threaded=True, port=8080)
