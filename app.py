import os
from flask import Flask, request, render_template, g
import cv2
from ultralytics import YOLO
from werkzeug.exceptions import HTTPException
import logging
import threading
import pandas as pd
import numpy as np
import json
import joblib
from hijri_converter import convert
from datetime import datetime
from image_caption import caption_generator


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True


input_image_save = 'static/input/'
output_image_save = 'static/predicted_output/'

thread_lock = threading.Lock()


@app.before_request
def load_models():
    pose_model = YOLO(os.path.join(app.root_path, 'static/models/best_pose_model.pt'))
    seg_model = YOLO(os.path.join(app.root_path, 'static/models/best_seg_model.pt'))
    breed_model = YOLO(os.path.join(app.root_path, 'static/models/best_breed_classification_model.pt')) 
    weight_model_path = os.path.join(app.root_path, 'static/models/cow_weight_predictor(ExtraTreesRegressor+Transformer).pkl')
    weight_model = joblib.load(weight_model_path)
    

    g.model1 = pose_model
    g.model2 = seg_model
    g.model3 = breed_model
    g.model4 = weight_model


@app.route('/')
def home():
    return render_template('index.html')

def weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose):  #weight calculation ki logic, plust end mein predicted weight pose wise (total model implementation mein hoga)
    
    weight_model = g.model4
    
    keypoints_coordinates = [
        'nose_x', 'nose_y', 'nose_z',
        'rght_eye_x', 'rght_eye_y', 'rght_eye_z',
        'left_eye_x', 'left_eye_y', 'left_eye_z',
        'side_neck_x', 'side_neck_y', 'side_neck_z',
        'left_front_hoof_x', 'left_front_hoof_y', 'left_front_hoof_z',
        'right_front_hoof_x', 'right_front_hoof_y', 'right_front_hoof_z',
        'left_back_hoof_x', 'left_back_hoof_y', 'left_back_hoof_z',
        'right_back_hoof_x', 'right_back_hoof_y', 'right_back_hoof_z',
        'back_bone_x', 'back_bone_y', 'back_bone_z',
        'tail_root_x', 'tail_root_y', 'tail_root_z',
        'backpose_mid_x', 'backpose_mid_y', 'backpose_mid_z',
        'center_point_x', 'center_point_y', 'center_point_z'
    ]
    
    
    # Create a dictionary to store the processed data
    processed_data = {}
    for i, kpt in enumerate(predicted_keypoints):
        if i * 3 < len(keypoints_coordinates):
            processed_data[keypoints_coordinates[i * 3]] = kpt['x']
            processed_data[keypoints_coordinates[i * 3 + 1]] = kpt['y']
            processed_data[keypoints_coordinates[i * 3 + 2]] = kpt['z']
        else:
            break
    
    
    white_pixels = pixel_details[0]
    black_pixels = pixel_details[1]
    


    
    bbox_width = pose_bbox[2] - pose_bbox[0]
    bbox_height = pose_bbox[3] - pose_bbox[1]

    
    additional_data = {
        'org_image_width': org_image_shapes[0],
        'org_image_height': org_image_shapes[1],
        'pose': pose,
        'pixels_covered': white_pixels,
        'black_pixels': black_pixels,
        'bbox_width' : bbox_width,
        'bbox_height' : bbox_height
    }
    
    # Update processed_data with additional data
    processed_data.update(additional_data)

    # Convert processed data to DataFrame
    new_dataframe = pd.DataFrame([processed_data])
    mapping = {'l_side': 0, 'r_side': 1, 'back': 2, 'front': 3}
    new_dataframe['pose'] = new_dataframe['pose'].map(mapping)
    
    scaler_path='static/models/fitted_scaler.pkl'
    scaler=joblib.load(scaler_path)
    processed_dataframe=scaler.transform(new_dataframe) 
    weight=weight_model.predict(processed_dataframe)
    weight=weight.item()

    return weight



def price_calculation(weight):
    eid_condition=None
    factor_value =None
    
    price_kg=40000/40
    
    current_date = datetime.now().date()
    hijri_date = convert.Gregorian(current_date.year, current_date.month, current_date.day).to_hijri()
    hijri_month = hijri_date.month 
    
    if hijri_month == 12:
        if hijri_date.day >= 10:
            eid_condition = 0.8
        else:
            eid_condition = 1.45
    else:
        eid_condition = 1.0  # Default condition for other months
        
    hijri_months_factor={
            1 : 1.2,
            2 : 1,
            3 : 1.3,
            4 : 1,
            5 : 1,
            6 : 1,
            7 : 1.1,
            8 : 1.15,
            9 : 1.2,
            10 : 1.3,
            11 : 1.35,
            12 : eid_condition 
                    }
    
    for h in hijri_months_factor:
        if h == hijri_month:
            factor_value=hijri_months_factor[h]
    

    price=int((price_kg*factor_value) * weight)
    
    return price



def model_implementation(image, filename):
    try:
        pose_model = g.model1
        seg_model = g.model2
        breed_model = g.model3
        org_image_shapes=(4064,3048)

        
        def pose_image_output(pose):
            if pose in ['l_side', 'r_side']:
                pose='side'
            else:
                pose=pose
                
            pose_predicted_image = pose_results[0].plot() # plotted image pose
            pose_output_image_path = os.path.join(output_image_save, f'{pose}_pose_output_image.jpg')
            try:
                cv2.imwrite(pose_output_image_path, pose_predicted_image)
                return pose_output_image_path
            except Exception as e:
                logging.exception("Error saving Pose Estimation image: %s", str(e))
                return e
                    
            
            
        def seg_image_output(pose):
            if pose in ['l_side', 'r_side']:
                pose='side'
            else:
                pose=pose
                
            seg_predicted_image = seg_results[0].plot() # plotted image seg
            seg_output_image_path = os.path.join(output_image_save, f'{pose}_seg_output_image.jpg')
            try:
                cv2.imwrite(seg_output_image_path, seg_predicted_image)
                return seg_output_image_path
            except Exception as e:
                logging.exception("Error saving segmentation image: %s", str(e))
                return e

        
        pose_results = pose_model(source=image, save=False, task='pose', conf=0.6)
        
        try:
            pose_predicted_data = json.loads(pose_results[0].tojson())  #[{ list of dict }] + all of the output in json format to see what object is detected marked
            pose_bbox_list = pose_results[0].boxes.xyxy[0].tolist()
            pose_bbox = [int(coord) for coord in pose_bbox_list]
            cropped_image = image.copy()
            
            cropped_image[:pose_bbox[1], :, :] = 0  # Set pixels above the bounding box to black
            cropped_image[pose_bbox[3]:, :, :] = 0  # Set pixels below the bounding box to black
            cropped_image[:, :pose_bbox[0], :] = 0  # Set pixels to the left of the bounding box to black
            cropped_image[:, pose_bbox[2]:, :] = 0  # Set pixels to the right of the bounding box to black

            seg_results = seg_model(source=cropped_image, save=False, task='segment')
            

            H, W, _ = image.shape

            for result in seg_results:
                for j, mask in enumerate(result.masks.data):
                    # Convert mask to numpy array and scale to 255
                    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                    
                    # Resize mask to match the original image size
                    resized_mask = cv2.resize(mask, (W, H))
                    
                    # Threshold the mask to get a binary mask
                    _, binary_mask = cv2.threshold(resized_mask, 128, 255, cv2.THRESH_BINARY)
                    
                    # Count white and black pixels in the binary mask
                    white_pixels = np.count_nonzero(binary_mask == 255)
                    black_pixels = np.count_nonzero(binary_mask == 0)
                    
                    pixel_details=[white_pixels,black_pixels]
        except:
            return pose_predicted_data   # empty list will be return 
            

                
        # Required Variables assignments
        detected_object = None
        pose = None
        height = None
        count = 0
        pose_output_image_path = None
        seg_output_image_path = None
        weight_data= 0
        breed_output = None
        image_caption = None
        
        
        """ Logic: We have 2 classes in our dataset which are 'cow' and 'other'. If there is no object detected or
            other object detected, means there's no cow present in the given image. Hence No keypoints, working
            will be triggered in this situation """
        
        
        for item in pose_predicted_data:
            if item.get('name', '') == 'other' or item == []:
                detected_object = 'other'
                
            
            elif item.get('name', '') == 'cow':
                detected_object = 'cow'
                count = count + 1
                object_count = f"{count} {detected_object} detected in the given image"
        
                # predicted keypoints
                predicted_keypoints=[]

                kptss = pose_results[0].keypoints.data
                for kpts in kptss:
                    for kpt in kpts:
                        predicted_keypoints.append(kpt)
                
                
                predicted_keypoints = [{"x": kpt[0].item(), "y": kpt[1].item(), "z": kpt[2].item()} for kpt in predicted_keypoints]

                for d in predicted_keypoints:
                    if d['x'] == 0.0 and d['y'] == 0.0:   # If x,y is 0 which means the object is not present in the photo therefore, z=0
                        d['z'] = 0.0
                        
        """
        ======= "Predicted Keypoints" =========
                    0: Nose
                    1: Right Eye
                    2: Left Eye
                    3: Side Neck
                    4: Left Front Hoof
                    5: Right Front hoof
                    6: Left Back Hoof
                    7: Right Back Hoof
                    8: Backbone
                    9: Tail Root
                    10: Backpose Mid
                    11: Center Point
        """
        
        if filename == 'side_pose.jpg' and detected_object =='cow' and count == 1:
            
            image_caption=caption_generator(cropped_image)
                
            if (predicted_keypoints[1].get('x') == 0 and predicted_keypoints[2].get('x') != 0) and predicted_keypoints[10].get('x') == 0 and predicted_keypoints[11].get('x') != 0:
                pose ='l_side'
                
                if pose == 'l_side':
                    first_foot = predicted_keypoints[4]  # left front foot
                    second_foot = predicted_keypoints[6] # left back foot
                                        
                    for h in [first_foot, second_foot]:
                        if (h.get('x') and h.get('y')) > 800:  # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height = 'Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                            
                    pose_output_image_path = pose_image_output(pose)
                    seg_output_image_path = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
            
            elif (predicted_keypoints[1].get('x') != 0 and predicted_keypoints[2].get('x') == 0) and predicted_keypoints[10].get('x') == 0 and predicted_keypoints[11].get('x') != 0:
                pose ='r_side'
                if pose == 'r_side':
                    first_foot = predicted_keypoints[5] # right front foot
                    second_foot = predicted_keypoints[7]  # right back foot

                    for h in [first_foot, second_foot]:
                        if (h.get('x') and h.get('y')) > 800:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                                
                    pose_output_image_path = pose_image_output(pose)
                    seg_output_image_path = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
                
                
            else:
                pose=None
                        
            
            if pose == 'l_side' or 'r_side':
           
                cropped_image2 = image[int(pose_bbox[1]*0.7):int(pose_bbox[3]*1.07), int(pose_bbox[0]*0.7):int(pose_bbox[2]*1.07)]
 
                breed_results = breed_model(source=cropped_image2, save=False, conf=0.4)
                names_dict=breed_results[0].names
                probs=breed_results[0].probs
                probs_list=probs.data.tolist()
                breed_index=np.argmax(probs_list)
                breed=names_dict.get(breed_index)
                conf=np.max(probs_list)
                
                breed_output= (f'The Predicted breed is "{breed}", with confidence of "{int(conf*100)}" %')
        
        elif filename == 'front_pose.jpg' and detected_object == 'cow' and count == 1:
            if (predicted_keypoints[0].get('x') != 0 and predicted_keypoints[1].get('x') != 0) or (predicted_keypoints[0].get('x') != 0 and predicted_keypoints[2].get('x') != 0):
                pose ='front'
                
                if pose == 'front':
                    first_foot = predicted_keypoints[4] # left front foot
                    second_foot = predicted_keypoints[5]  # right front hoof

                    for h in [first_foot, second_foot]:
                        if (h.get('x') and h.get('y')) > 800:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                    
                    pose_output_image_path = pose_image_output(pose)
                    seg_output_image_path = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
            else:
                pose=None
                
        
        elif filename == 'back_pose.jpg' and detected_object =='cow' and count == 1:
            if (predicted_keypoints[9].get('x') != 0 and predicted_keypoints[10].get('x') != 0):
                pose ='back'
                
                if pose == 'back':
                    first_foot = predicted_keypoints[6] # left back foot
                    second_foot = predicted_keypoints[7]  # right back hoof

                    for h in [first_foot, second_foot]:
                        if (h.get('x') and h.get('y')) > 800:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                            
                    pose_output_image_path = pose_image_output(pose)
                    seg_output_image_path = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
                
            else:
                pose=None
            
        else:
            pose='unidentified'
        

        return pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption

    except Exception as e:
        logging.exception("An error occurred: %s", str(e))
        return e
    

@app.route('/evaluate', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            with thread_lock:
                loaded_images = request.files.getlist('images')
                files = []  # all three images (name) which are accepted
                for files_list in loaded_images:
                    file = files_list.filename
                    files.append(file)

                def image_stream(file):
                    image_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    return image

                side_pose_complete_data = None
                front_pose_complete_data = None
                back_pose_complete_data = None

                for file in loaded_images:
                    if file.filename == 'side_pose.jpg':
                        filename = file.filename
                        image = image_stream(file)
                        try:
                            pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption = model_implementation(image, filename)
                            side_pose_complete_data = [pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption]
                        except:
                            side_pose_complete_data = ['side pose of cow is not detected', 'Unknown', 'Unknown', 'None', None, 0, 'Unknown', 'Unknown']
                            continue

                    elif file.filename == 'front_pose.jpg':
                        filename = file.filename
                        image = image_stream(file)
                        try:
                            pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption = model_implementation(image, filename)
                            front_pose_complete_data = [pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption]
                        except:
                            front_pose_complete_data = ['front pose of cow is not detected', 'Unknown', 'Unknown', 'None', None, 0, 'Unknown', 'Unknown']
                            continue

                    elif file.filename == 'back_pose.jpg':
                        filename = file.filename
                        image = image_stream(file)
                        try:
                            pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption = model_implementation(image, filename)
                            back_pose_complete_data = [pose_output_image_path, height, detected_object, object_count, seg_output_image_path, weight_data, breed_output, image_caption]
                        except:
                            back_pose_complete_data = ['back pose of cow is not detected', 'Unknown', 'Unknown', 'None', None, 0, 'Unknown', 'Unknown']
                            continue

                # Calculate total weight if all three poses are provided
                total_weight = (side_pose_complete_data[5] + front_pose_complete_data[5] + back_pose_complete_data[5])
                weight_in_mun = total_weight / 40

                str_total_weight = f"{int(total_weight)} ± {int(total_weight * 0.10)} kgs  ==  {weight_in_mun:.2f} ± {weight_in_mun * 0.10:.2f} mun "
                price = price_calculation(total_weight)
                str_price = f"PKR {int(price * 0.9)} to PKR {int(price * 1.1)}"

                result = {
                    'side_pose_complete_data': side_pose_complete_data,
                    'front_pose_complete_data': front_pose_complete_data,
                    'back_pose_complete_data': back_pose_complete_data,
                    'total_weight': str_total_weight,
                    'price': str_price
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
    
    
result_file = 'assets/last_result.json'

def save_last_result(result):
    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f)
    except Exception as e:
        logging.exception("An error occurred while saving the result: %s", str(e))



result_file = 'assets/last_result.json'

def load_last_result():
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    else:
        return None



def save_last_result(result):
    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f)
    except Exception as e:
        logging.exception("An error occurred while saving the result: %s", str(e))


    
@app.route('/result', methods=['GET'])
def show_last_result():
    last_result = load_last_result()
    if last_result:
        return render_template('result.html', **last_result)
    else:
        return "No result available."    
    
 

if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=80, threaded=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
