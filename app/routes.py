from flask import render_template, request, current_app
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

from app import app

@app.route('/', methods=['GET', 'POST'])
def detect_objects():
    # if request.method == 'POST':
    #     try:
    #         loaded_image = request.files['image']
    #         image = cv2.imdecode(np.frombuffer(loaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         # Use the pre-loaded YOLO model
    #         yolo_model = current_app.config['yolo_model']
    #         results = yolo_model(source=image, save=False, conf=0.6, task='pose')

    #         pred_image = results[0].plot()
    #         pred_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

    #         # Convert images to base64 format
    #         input_image_base64 = encode_image(image_rgb)
    #         output_image_base64 = encode_image(pred_rgb)

    #         return render_template('result.html', input_image_base64=input_image_base64, output_image_base64=output_image_base64)

    #     except Exception as e:
    #         return {'error': str(e)}
    # else:
    return render_template('index.html')

def encode_image(image):
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
