from flask import Flask, g
from hijri_converter import convert
from datetime import datetime
import joblib
import pandas as pd

app = Flask(__name__)


def weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose):  
    weight_model = g.weight_model

    
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
    
    def calculate_weight(org_image_width, org_image_height):
        additional_data = {
            'org_image_width': org_image_width,
            'org_image_height': org_image_height,
            'pose': pose if pose else 'side',
            'pixels_covered': white_pixels if white_pixels is not None else 1,
            'black_pixels': black_pixels if black_pixels is not None else 1,
            'bbox_width': bbox_width if bbox_width is not None else 1,
            'bbox_height': bbox_height if bbox_height is not None else 1
        }
        
        processed_data.update(additional_data)
        
        new_dataframe = pd.DataFrame([processed_data])
        mapping = {'l_side': 0, 'r_side': 1, 'back': 2, 'front': 3}
        new_dataframe['pose'] = new_dataframe['pose'].map(mapping)
        
        # new_dataframe = new_dataframe[['nose_x', 'left_back_hoof_x', 'right_back_hoof_x', 'right_back_hoof_y', 'right_back_hoof_z', 'center_point_x', 'center_point_y', 'center_point_z', 'pose', 'bbox_height']]
        
        to_remove = ['backpose_mid_z',
                    'right_front_hoof_y',
                    'left_eye_z',
                    'backpose_mid_y',
                    'left_eye_y',
                    'nose_z',
                    'left_front_hoof_x',
                    'black_pixels',
                    'back_bone_z',
                    'tail_root_y',
                    'rght_eye_y',
                    'left_front_hoof_z',
                    'nose_y',
                    'bbox_width',
                    'org_image_width',
                    'side_neck_z',
                    'pixels_covered',
                    'back_bone_y',
                    'rght_eye_z',
                    'left_front_hoof_y',
                    'back_bone_x',
                    'side_neck_y',
                    'left_back_hoof_y',
                    'org_image_height',
                    'side_neck_x',
                    'right_front_hoof_x',
                    'right_front_hoof_z',
                    'rght_eye_x',
                    'left_back_hoof_z',
                    'tail_root_z',
                    'left_eye_x',
                    'tail_root_x',
                    'backpose_mid_x']
                            
                            
        new_dataframe = new_dataframe.drop(columns=to_remove)        
                            
        scaler_path = 'static/models/fitted_scaler.pkl'
        scaler = joblib.load(scaler_path)
        
        new_dataframe = new_dataframe.fillna(0)
        processed_dataframe = scaler.transform(new_dataframe)
        weight = weight_model.predict(processed_dataframe)
        return weight.item()
    
    weight = calculate_weight(org_image_shapes[0], org_image_shapes[1])
    
    # weight2 = calculate_weight(org_image_shapes[1], org_image_shapes[0])
    
    # weight = (weight1 1.75
    
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
