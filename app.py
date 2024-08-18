import os
from flask import Flask, request, render_template, g, jsonify
from flask_cors import CORS
import cv2
import base64
from ultralytics import YOLO
from werkzeug.exceptions import HTTPException
import logging
import threading
import numpy as np
import json
import joblib

from weight_price import price_calculation
from pose_seg import model_implementation


class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ModelLoader(Singleton):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.pose_model = YOLO(os.path.join(app.root_path, 'static/models/last.pt'))
            # self.pose_model = YOLO(os.path.join(app.root_path, 'static/models/best_pose_model.pt'))
            self.seg_model = YOLO(os.path.join(app.root_path, 'static/models/best_seg_model.pt'))
            self.breed_model = YOLO(os.path.join(app.root_path, 'static/models/best_breed_classification_model.pt')) 
            # self.weight_model_path = os.path.join(app.root_path, 'static/models/cow_weight_predictor(ExtraTreesRegressor+Transformer).pkl')
            self.weight_model_path = os.path.join(app.root_path, 'static/models/cow_weight_predictor(stacked).pkl')
            self.weight_model = joblib.load(self.weight_model_path)
            self.initialized = True

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



result_file = 'assets/last_result.json'
thread_lock = threading.Lock()



def decode_image(file):
    # Read and decode image file directly from memory
    in_memory_file = file.read()
    np_array = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image



def preprocess_image(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    target_h, target_w = target_size
    aspect_ratio = min(target_h / h, target_w / w)

    if aspect_ratio < 1:
        resized_image = cv2.resize(image, None, fx=aspect_ratio, fy=aspect_ratio, interpolation=cv2.INTER_AREA)
    else:
        resized_image = cv2.resize(image, None, fx=aspect_ratio, fy=aspect_ratio, interpolation=cv2.INTER_LINEAR)

        if resized_image.shape[0] < target_h or resized_image.shape[1] < target_w:
            resized_image = cv2.resize(resized_image, target_size, interpolation=cv2.INTER_LINEAR)

    top_pad = (target_h - resized_image.shape[0]) // 2
    bottom_pad = target_h - resized_image.shape[0] - top_pad
    left_pad = (target_w - resized_image.shape[1]) // 2
    right_pad = target_w - resized_image.shape[1] - left_pad

    top_pad = max(0, top_pad)
    bottom_pad = max(0, bottom_pad)
    left_pad = max(0, left_pad)
    right_pad = max(0, right_pad)

    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return padded_image



@app.before_request
def load_models():
    g.model_loader = ModelLoader()
    g.pose_model = g.model_loader.pose_model
    g.seg_model = g.model_loader.seg_model
    g.breed_model = g.model_loader.breed_model
    g.weight_model = g.model_loader.weight_model


def save_last_result(result: dict) -> None:
    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)  
    except Exception as e:
        logging.exception("An error occurred while saving the result: %s", str(e))
        

def initialize_result_file() -> None:
    try:
        with open(result_file, 'w') as file:
            json.dump({}, file)  
    except Exception as e:
        logging.exception("An error occurred while initializing the result file: %s", str(e))

def load_last_result() -> dict:
    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.exception("An error occurred while loading the result: %s", str(e))
        return {}
    
    
    
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/evaluate_web', methods=['GET', 'POST'])
def prediction_web():
    if request.method == 'POST':
        try:
            with thread_lock:
                initialize_result_file()
                
                
                front_pose = request.files.get('front_pose')
                side_pose = request.files.get('side_pose')
                back_pose = request.files.get('back_pose')
                
                if not (front_pose and side_pose and back_pose):
                    return render_template('error.html', error_message='All pose images are required.')
                
                org_image_width = 3048
                org_image_height = 4064
                hw = (org_image_width, org_image_height)
                

                images = {
                    'front_pose': preprocess_image(decode_image(front_pose)),
                    'side_pose': preprocess_image(decode_image(side_pose)),
                    'back_pose': preprocess_image(decode_image(back_pose))
                }

                
                
                
                
                cv2.imwrite('FRONT_POSE.jpg',images['front_pose'])
                cv2.imwrite('SIDE_POSE.jpg',images['side_pose'])
                cv2.imwrite('BACK_POSE.jpg',images['back_pose'])
                

                results = {}

                for pose_name, image in images.items():
                    try:
                        pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption = model_implementation(image, f'{pose_name}.jpg', hw)
                        results[pose_name] = [pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption]
                    except Exception as e:
                        logging.exception(f"Error processing {pose_name}: {str(e)}")
                        results[pose_name] = [f'{pose_name} of cow is not detected', 'Unknown', 'Unknown', 'None', None, 0, 'Unknown', 'Unknown']

                total_weight = sum(result[5] for result in results.values())
                weight_in_mun = total_weight / 40
                
                
                # PRICE BASED ON STRUCTURE#
                

                str_total_weight = f"{int(total_weight)} ± {int(total_weight * 0.10)} kgs  ==  {weight_in_mun:.2f} ± {weight_in_mun * 0.10:.2f} mun"
                price = price_calculation(total_weight)
                
                side_pose_data = results.get('side_pose')
                
                def adjust_price(price, side_pose_data):
    
                    structure_rate = side_pose_data[7][-8]
                    if structure_rate == 'excellent':
                        price = price+(price / 100) * 7
                    elif structure_rate == 'good':
                        pass  
                    elif structure_rate == 'average':
                        price = price+(price / 100) * -5
                    elif structure_rate == 'bad':
                        price = price+(price / 100) * -10

                    beauty = side_pose_data[7][1]
                    if beauty == 'beautiful':
                        price = price+(price / 100) * 7
                    elif beauty == 'good':
                        pass 
                    elif beauty in ['average', 'below']:
                        price = price+(price / 100) * -5

                    return price
                
                adjuste_price = adjust_price(price,side_pose_data)

                str_price = f"PKR {int(adjuste_price * 0.9):,} to PKR {int(adjuste_price * 1.1):,}"
                


                result = {
                        'side_pose_complete_data': results.get('side_pose', None),
                        'front_pose_complete_data': results.get('front_pose', None),
                        'back_pose_complete_data': results.get('back_pose', None),
                        'total_weight': str_total_weight if str_total_weight is not None else None,
                        'price': str_price if str_price is not None else None
                    }
                
                

                save_last_result(result)

                return render_template('result_web.html', **result)

        except HTTPException as e:
            logging.error("HTTP version not supported: %s", str(e))
            return render_template('error.html', error_message=str(e), code=e.code)

        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
            return render_template('error.html', error_message=str(e), code=500)

    else:
        return render_template('index.html')




@app.route('/evaluate', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            with thread_lock:
                initialize_result_file()
                data = request.get_json()
                
                # org_image_width=data.get('width')
                # org_image_height=data.get('height')
                
                org_image_width = 3048
                org_image_height = 4064
                hw = (org_image_width, org_image_height)
                
                
                print(f"HEIGHT AND WIDTH: {hw}")

                front_pose_base64 = data.get('front_pose')
                side_pose_base64 = data.get('side_pose')
                back_pose_base64 = data.get('back_pose')

                if not all([front_pose_base64, side_pose_base64, back_pose_base64]):
                    return jsonify({'error': 'Missing images'}), 400

                def decode_base64_image(image_base64):
                    image_data = base64.b64decode(image_base64)
                    np_array = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    return image

                images = {
                    'front_pose': cv2.rotate(decode_base64_image(front_pose_base64), cv2.ROTATE_90_COUNTERCLOCKWISE),
                    'side_pose': cv2.rotate(decode_base64_image(side_pose_base64), cv2.ROTATE_90_COUNTERCLOCKWISE),
                    'back_pose': cv2.rotate(decode_base64_image(back_pose_base64), cv2.ROTATE_90_COUNTERCLOCKWISE)
                }
                
                # images = {
                #     'front_pose': decode_base64_image(front_pose_base64),
                #     'side_pose': decode_base64_image(side_pose_base64),
                #     'back_pose': decode_base64_image(back_pose_base64)
                # }
                
                
                cv2.imwrite('FRONT_POSE.jpg',images['front_pose'])
                cv2.imwrite('SIDE_POSE.jpg',images['side_pose'])
                cv2.imwrite('BACK_POSE.jpg',images['back_pose'])
                

                results = {}

                for pose_name, image in images.items():
                    try:
                        pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption = model_implementation(image, f'{pose_name}.jpg', hw)
                        results[pose_name] = [pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption]
                    except Exception as e:
                        logging.exception(f"Error processing {pose_name}: {str(e)}")
                        results[pose_name] = [f'{pose_name} of cow is not detected', 'Unknown', 'Unknown', 'None', None, 0, 'Unknown', 'Unknown']

                total_weight = sum(result[5] for result in results.values())
                weight_in_mun = total_weight / 40
                
                
                # PRICE BASED ON STRUCTURE#
                

                str_total_weight = f"{int(total_weight)} ± {int(total_weight * 0.10)} kgs  ==  {weight_in_mun:.2f} ± {weight_in_mun * 0.10:.2f} mun"
                price = price_calculation(total_weight)
                
                side_pose_data = results.get('side_pose')
                
                def adjust_price(price, side_pose_data):
    
                    structure_rate = side_pose_data[7][-8]
                    if structure_rate == 'excellent':
                        price = price+(price / 100) * 7
                    elif structure_rate == 'good':
                        pass  
                    elif structure_rate == 'average':
                        price = price+(price / 100) * -5
                    elif structure_rate == 'bad':
                        price = price+(price / 100) * -10

                    beauty = side_pose_data[7][1]
                    if beauty == 'beautiful':
                        price = price+(price / 100) * 7
                    elif beauty == 'good':
                        pass 
                    elif beauty in ['average', 'below']:
                        price = price+(price / 100) * -5

                    return price
                
                adjuste_price = adjust_price(price,side_pose_data)

                str_price = f"PKR {int(adjuste_price * 0.9):,} to PKR {int(adjuste_price * 1.1):,}"
                


                result = {
                        'side_pose_complete_data': results.get('side_pose', None),
                        'front_pose_complete_data': results.get('front_pose', None),
                        'back_pose_complete_data': results.get('back_pose', None),
                        'total_weight': str_total_weight if str_total_weight is not None else None,
                        'price': str_price if str_price is not None else None
                    }
                
                

                save_last_result(result)

                return render_template('result.html', **result)

        except HTTPException as e:
            logging.error("HTTP version not supported: %s", str(e))
            return render_template('error.html', error_message=str(e), code=e.code)

        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
            return render_template('error.html', error_message=str(e), code=500)

    else:
        return render_template('index.html')

@app.route('/result', methods=['GET'])
def show_last_result():
    last_result = load_last_result()
    if last_result:
        return render_template('result.html', **last_result)
    else:
        return "System is unable to detetect any cow in the given images."    
    



if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=80, threaded=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

