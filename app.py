import os
from flask import Flask, request, render_template, g
import cv2
from ultralytics import YOLO
from werkzeug.exceptions import HTTPException
import logging
import threading
import numpy as np
import json

class PoseModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        results = self.model(source=image, save=False, task='pose')
        return results
    
    
    
class ImageProcessor:
    def __init__(self, model_path, input_image_save, output_image_save):
        self.yolo_model = PoseModel(model_path)
        self.input_image_save = input_image_save
        self.output_image_save = output_image_save

    def process_image(self, image):
        try:
            results = self.yolo_model.detect_objects(image)
            
            predicted_data = json.loads(results[0].tojson())  #[{ list of dict }] + all of the output in json format to see what object is detected marked

            # Required Variables
            detected_object = None
            pose = None
            height = None
            object_count = None
            visible_landmarks=None
            
            
            """ Logic: We have 2 classes in our dataset which are 'cow' and 'other'. If there is no object detected or
                other object detected, means there's no cow present in the given image. Hence No keypoints, working
                will be triggered in this situation """
                
                
            count = 0
            
            for item in predicted_data:
                if item.get('name', '') == 'other' or item == []:
                    detected_object = 'other'
                    
                
                elif item.get('name', '') == 'cow':
                    detected_object = 'cow'
                    count = count + 1
                    object_count = f"{count} {detected_object} detected in the given image"
            
                    # predicted keypoints
                    predicted_keypoints=[]

                    kptss = results[0].keypoints.data
                    for kpts in kptss:
                        for kpt in kpts:
                            predicted_keypoints.append(kpt)
                    
                    
                    keypoints_lst_of_dict = [{"x": kpt[0].item(), "y": kpt[1].item(), "z": kpt[2].item()} for kpt in predicted_keypoints]

                    for d in keypoints_lst_of_dict:
                        if d['x'] == 0.0 and d['y'] == 0.0:   # If x,y is 0 which means the object is not present in the photo therefore, z=0
                            d['z'] = 0.0
                            
                    # all landmarks in our dataset
                    landmarks = ['Nose', 
                                'Right Eye', 'Left Eye', 
                                'Side Neck', 
                                'Left Front Hoof', 'Right Front Hoof',
                                'Left Back Hoof', 'Right Back Hoof',
                                'Backbone', 'Tail Root', 'Back Pose Mid', 'Center Point'
                                ]

                    total_landmarks = {}   # will return the results at the end
                    
                    for l, k in zip(landmarks, keypoints_lst_of_dict):
                        total_landmarks[l] = k
                    
                    visible_landmarks=[]
                    
                    for key, value in total_landmarks.items():
                            if value.get('z', 0.0) != 0.0:
                                visible_landmarks.append(key)
                    
                    visible_landmarks = ', '.join(visible_landmarks)  # formatting in the readable form for USER

                    
                        

                    # POSE DETECTION WORKING....
                    # keypoints list with respect to poses

                    if (keypoints_lst_of_dict[1].get('x') == 0 and keypoints_lst_of_dict[2].get('x') != 0) and keypoints_lst_of_dict[10].get('x') == 0:
                        pose ='left side'

                    elif (keypoints_lst_of_dict[1].get('x') != 0 and keypoints_lst_of_dict[2].get('x') == 0) and keypoints_lst_of_dict[10].get('x') == 0:
                        pose ='right side'
                        
                    else:
                        pose=None
                                
                    # HEIGHT ESTIMATION WORKING....
                    if pose == 'left side':
                        first_foot = keypoints_lst_of_dict[4]  # left front foot
                        second_foot = keypoints_lst_of_dict[6] # left back foot
                                            
                        for h in [first_foot, second_foot]:
                            if (h.get('x') and h.get('y')) > 800:
                                height='normal'
                                
                            elif (h.get('x') or h.get('y')) == 0:
                                height = 'Please retake the image as one of the hoof is not clearly visible'
                            
                            else:
                                height='Not Normal'
                        
                    if pose == 'right side':
                        first_foot = keypoints_lst_of_dict[5] # right front foot
                        second_foot = keypoints_lst_of_dict[7]  # right back foot

                        for h in [first_foot, second_foot]:
                            if (h.get('x') and h.get('y')) > 800:
                                height='normal'
                                
                            elif (h.get('x') or h.get('y')) == 0:
                                height ='Please retake the image as one of the hoof is not clearly visible'
                            
                            else:
                                height='Not Normal'    
                                
                       
                    
            predicted_image = results[0].plot() # plotted image
        
            output_image_path = os.path.join(self.output_image_save, 'output_image.jpeg')
            input_image_path = os.path.join(self.input_image_save, 'input_image.jpeg')
            cv2.imwrite(output_image_path, predicted_image)
            cv2.imwrite(input_image_path, image)

            return input_image_path, output_image_path, height, pose, detected_object, object_count, visible_landmarks
            
        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
            return e



class FlaskApp:
    def __init__(self, model_path, input_image_save, output_image_save):
        self.app = Flask(__name__)
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        self.app.config['SUPPRESS_EXCEPTIONS'] = True
        self.image_processor = ImageProcessor(model_path, input_image_save, output_image_save)
        self.thread_lock = threading.Lock()

        self.app.before_request(self.load_model)
        self.app.route('/')(self.home)
        self.app.route('/evaluate', methods=['GET', 'POST'])(self.prediction)

    def load_model(self):
        model = YOLO(os.path.join(self.app.root_path, 'static/model/best_pose_model.pt'))
        g.model = model

    def home(self):
        return render_template('index.html')

    def prediction(self):
        if request.method == 'POST':
            try:
                with self.thread_lock:
                    loaded_image = request.files['image']
                    image_stream = loaded_image.stream
                    image_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                    input_image, output_image, height, pose, detected_object, object_count, visible_landmarks = self.image_processor.process_image(image)

                    return render_template('result.html', input_image=[input_image], output_image=[output_image],
                                           height=height, pose=pose, detected_object=detected_object, object_count=object_count,
                                           visible_landmarks=visible_landmarks)

            except HTTPException as e:
                logging.error("HTTP version not supported: %s", str(e))
                return render_template('error.html', error_message=str(e), code=e.code)

            except Exception as e:
                logging.exception("An error occurred: %s", str(e))
                return render_template('error.html', error_message=str(e), code=500)
        else:
            return render_template('index.html')

    def run(self):
        try:
            self.app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
        except Exception as e:
            print(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    flask_app = FlaskApp(
        # sequence is pose model path, input path, output path.. (may change in future if needed)
        os.path.join(os.path.dirname(__file__), 'static/model/best_pose_model.pt'),
        'static/input/',
        'static/predicted_output/'
    )
    flask_app.run()












# ------------------- OLD CODE ---------------------------------#


# app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# app.config['SUPPRESS_EXCEPTIONS'] = True


# input_image_save = 'static/input/'
# output_image_save = 'static/predicted_output/'

# thread_lock = threading.Lock()


# @app.before_request
# def load_model():
#     # model = YOLO(os.path.join(app.root_path, 'static/model/best_pose_model.pt'))
#     model = YOLO(os.path.join(app.root_path, 'static/model/90 epochs.pt'))

#     g.model = model



# @app.route('/')
# def home():
#     return render_template('index.html')

# def model_implementation(image):
#     try:
#         # image = cv2.imread(image)
#         model = g.model
        
        
#         results = model(source=image, save=False, task='pose')
        
#         predicted_data = json.loads(results[0].tojson())  #[{ list of dict }] + all of the output in json format to see what object is detected marked

#         # Required Variables
#         detected_object = None
#         pose = None
#         height = None
#         object_count = None
        
        
#         """ Logic: We have 2 classes in our dataset which are 'cow' and 'other'. If there is no object detected or
#             other object detected, means there's no cow present in the given image. Hence No keypoints, working
#             will be triggered in this situation """
            
            
#         count = 0
        
#         for item in predicted_data:
#             if item.get('name', '') == 'other' or item == []:
#                 detected_object = 'other'
                
            
#             elif item.get('name', '') == 'cow':
#                 detected_object = 'cow'
#                 count = count + 1
#                 object_count = f"{count} {detected_object} detected in the given image"
        
#                 # predicted keypoints
#                 predicted_keypoints=[]

#                 kptss = results[0].keypoints.data
#                 for kpts in kptss:
#                     for kpt in kpts:
#                         predicted_keypoints.append(kpt)
                
                
#                 list_of_dicts = [{"x": kpt[0].item(), "y": kpt[1].item(), "z": kpt[2].item()} for kpt in predicted_keypoints]

#                 for d in list_of_dicts:
#                     if d['x'] == 0.0 and d['y'] == 0.0:   # If x,y is 0 which means the object is not present in the photo therefore, z=0
#                         d['z'] = 0.0



#                 # POSE DETECTION WORKING....

#                 if (list_of_dicts[1].get('x') == 0 and list_of_dicts[2].get('x') != 0) and list_of_dicts[10].get('x') == 0:
#                     pose ='left'

#                 elif (list_of_dicts[1].get('x') != 0 and list_of_dicts[2].get('x') == 0) and list_of_dicts[10].get('x') == 0:
#                     pose ='right'
#                 else:
#                     pose=None
                            
#                 # HEIGHT ESTIMATION WORKING....
#                 if pose == 'left':
#                     first_foot = list_of_dicts[4]  # left front foot
#                     second_foot = list_of_dicts[5] # left back foot
                                        
#                     for h in [first_foot, second_foot]:
#                         if (h.get('x') and h.get('y')) > 800:
#                             height='normal'
                            
#                         elif (h.get('x') or h.get('y')) == 0:
#                             height = 'Please retake the image as one of the hoof is not clearly visible'
                        
#                         else:
#                             height='Not Normal'
                    
#                 if pose == 'right':
#                     first_foot = list_of_dicts[6] # right front foot
#                     second_foot = list_of_dicts[7]  # right back foot

#                     for h in [first_foot, second_foot]:
#                         if (h.get('x') and h.get('y')) > 800:
#                             height='normal'
                            
#                         elif (h.get('x') or h.get('y')) == 0:
#                             height ='Please retake the image as one of the hoof is not clearly visible'
                        
#                         else:
#                             height='Not Normal'    
                            
                
                
#         predicted_image = results[0].plot() # plotted image
    
#         output_image_path = os.path.join(output_image_save, 'output_image.jpeg')
#         input_image_path = os.path.join(input_image_save, 'input_image.jpeg')
#         cv2.imwrite(output_image_path, predicted_image)
#         cv2.imwrite(input_image_path, image)

#         return input_image_path, output_image_path, height, pose, detected_object, object_count

#     except Exception as e:
#         logging.exception("An error occurred: %s", str(e))
#         return e
    
    

# @app.route('/evaluate', methods=['GET', 'POST'])
# def prediction():
#     if request.method == 'POST':
#         try:
#             with thread_lock:
#                 loaded_image = request.files['image']

#                 image_stream = loaded_image.stream
#                 image_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#                 image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

#                 input_image, output_image, height, pose, detected_object, object_count  = model_implementation(image)
                
#                 return render_template('result.html', input_image=[input_image], output_image=[output_image], 
#                                        height=height, pose=pose, detected_object=detected_object ,object_count=object_count)
        
#         except HTTPException as e: #specific exception.. used because it is common 
#             logging.error("HTTP version not supported: %s", str(e))
#             return render_template('error.html', error_message=str(e), code=e.code)

#         except Exception as e:  # to catch all others
#             logging.exception("An error occurred: %s", str(e)) # to maintain logs
#             return render_template('error.html', error_message=str(e), code=500)   # UI         

#     else:
#         return render_template('index.html')


# if __name__ == "__main__":
#     try:
#         app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
