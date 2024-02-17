import os
from flask import Flask, request, render_template, g
import cv2
from ultralytics import YOLO
from werkzeug.exceptions import HTTPException
import logging
import threading
import numpy as np


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True


input_image_save = 'static/input/'
output_image_save = 'static/predicted_output/'

thread_lock = threading.Lock()


@app.before_request
def load_model():
    model = YOLO(os.path.join(app.root_path, 'static/model/last.pt'))
    g.model = model



@app.route('/')
def home():
    return render_template('index.html')

def model_implementation(image):
    try:
        # image = cv2.imread(image)
        model = g.model
        
        if model is None:
            return 'MODEL IS NOT LOADED', None
        
        results = model(source=image, save=False, conf=0.6, task='pose')
        predicted_image = results[0].plot()

        output_image_path = os.path.join(output_image_save, 'output_image.jpeg')
        cv2.imwrite(output_image_path, predicted_image)

        return image, output_image_path

    except Exception as e:
        logging.exception("An error occurred: %s", str(e))
        return None, None  # Return None in case of error
    
    

@app.route('/evaluate', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            with thread_lock:
                loaded_image = request.files['image']

                image_data = loaded_image.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
                
                input_image, output_image  = model_implementation(image)
            
            
            # threads = []
            
            # for image in loaded_image:
            #     thread = threading.Thread(target=model_implementation, args=(loaded_image,))
            #     threads.append(thread)
            
            # for thread in threads:
            #     thread.start()

            # # Wait for all threads to finish
            # for thread in threads:
            #     thread.join()


                
            return render_template('result.html', input_image=[input_image], output_image=[output_image])
        
        except HTTPException as e:
            logging.error("HTTP version not supported: %s", str(e))
            return render_template('error.html', error_message=str(e), code=e.code)

        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
            return render_template('error.html', error_message=str(e), code=500)            

    else:
        return render_template('index.html')


if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
