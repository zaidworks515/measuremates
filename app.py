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
        
        results = model(source=image, save=False, conf=0.8, task='pose')
        
        # predicted keypoints
        predicted_keypoints=[]

        kptss = results[0].keypoints.data
        for kpts in kptss:
            for kpt in kpts:
                predicted_keypoints.append(kpt)
        
        
        list_of_dicts = [{"x": kpt[0].item(), "y": kpt[1].item(), "z": kpt[2].item()} for kpt in predicted_keypoints]

        for d in list_of_dicts:
            if d['x'] == 0.0 and d['y'] == 0.0:   # If x,y is 0 which means the object is not present in the photo therefore, z=0
                d['z'] = 0.0

        # POSE DETECTION WORKING....
        pose= ''

        if (list_of_dicts[1].get('x') == 0 and list_of_dicts[2].get('x') != 0) and list_of_dicts[10].get('x') == 0:
            pose ='left'

        elif (list_of_dicts[1].get('x') != 0 and list_of_dicts[2].get('x') == 0) and list_of_dicts[10].get('x') == 0:
            pose ='right'
        else:
            print('Retake the photo')
                    
        # HEIGHT ESTIMATION WORKING....
        if pose == 'left':
            first_foot = list_of_dicts[4]  # left front foot
            second_foot = list_of_dicts[5] # left back foot
            
            height=''
            
            for h in [first_foot, second_foot]:
                if (h.get('x') and h.get('y')) > 800:
                    height='normal'
                    
                elif (h.get('x') or h.get('y')) == 0:
                    height = 'Please retake the image as one of the hoof is not clearly visible'
                
                else:
                    height='Not Normal'
            
        if pose == 'right':
            first_foot = list_of_dicts[6] # right front foot
            second_foot = list_of_dicts[7]  # right back foot

            for h in [first_foot, second_foot]:
                if (h.get('x') and h.get('y')) > 800:
                    height='normal'
                    
                elif (h.get('x') or h.get('y')) == 0:
                    height ='Please retake the image as one of the hoof is not clearly visible'
                
                else:
                    height='Not Normal'    
                    
        
        
        predicted_image = results[0].plot() # plotted image

        output_image_path = os.path.join(output_image_save, 'output_image.jpeg')
        input_image_path = os.path.join(input_image_save, 'input_image.jpeg')
        cv2.imwrite(output_image_path, predicted_image)
        cv2.imwrite(input_image_path, image)
        
        return input_image_path, output_image_path, height, pose

    except Exception as e:
        logging.exception("An error occurred: %s", str(e))
        return None, None  # Return None in case of error
    
    

@app.route('/evaluate', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            with thread_lock:
                loaded_image = request.files['image']
                image_stream = loaded_image.stream
                image_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                input_image, output_image, height, pose  = model_implementation(image)
                
                return render_template('result.html', input_image=[input_image], output_image=[output_image], height=height, pose=pose)
        
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
