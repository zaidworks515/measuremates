import logging
from flask import Flask, g
import json
import os
import cv2
import numpy as np

from weight_price import weight_estimation
from image_caption import caption_generator

app = Flask(__name__)


def model_implementation(image, filename, hw):
    try:
        pose_model = g.pose_model
        seg_model = g.seg_model
        breed_model = g.breed_model
        org_image_shapes=hw
        
        print(f'ORG SHAPE OF IMAGE: {org_image_shapes}')

        
        def pose_image_output(pose):
            if pose in ['l_side', 'r_side']:
                pose='side'
            else:
                pose=pose
                
            pose_predicted_image = pose_results[0].plot() # plotted image pose
            
            return pose_predicted_image
            
                    
        def seg_image_output(pose):
            if pose in ['l_side', 'r_side']:
                pose='side'
            else:
                pose=pose
                
            seg_predicted_image = seg_results[0].plot() # plotted image seg
            
            return seg_predicted_image
            
            
        
        pose_results = pose_model(source=image, save=False, task='pose', conf=0.80)
        
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
                    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                    resized_mask = cv2.resize(mask, (W, H))
                    
                    _, binary_mask = cv2.threshold(resized_mask, 128, 255, cv2.THRESH_BINARY)
                    
                    # Count white and black pixels in the binary mask
                    white_pixels = np.count_nonzero(binary_mask == 255)
                    black_pixels = np.count_nonzero(binary_mask == 0)
                    
                    pixel_details=[white_pixels,black_pixels]
        except:
            return pose_predicted_data    
                
        # Required Variables assignments
        detected_object = None
        object_count = None
        pose = None
        height = None
        count = 0
        pose_output_image = None
        seg_output_image = None
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
            image_caption = image_caption.split(' ')
            
                
            if (predicted_keypoints[1].get('x') == 0 and predicted_keypoints[2].get('x') != 0) and predicted_keypoints[10].get('x') == 0 and predicted_keypoints[11].get('x') != 0:
                pose ='l_side'
                
                if pose == 'l_side':
                    first_foot = predicted_keypoints[4]  # left front foot
                    second_foot = predicted_keypoints[6] # left back foot
                                        
                    for h in [first_foot, second_foot]:
                        if h.get('y') > 380:  # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height = 'Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                            
                    pose_output_image = pose_image_output(pose)
                    seg_output_image = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
            
            elif (predicted_keypoints[1].get('x') != 0 and predicted_keypoints[2].get('x') == 0) and predicted_keypoints[10].get('x') == 0 and predicted_keypoints[11].get('x') != 0:
                pose ='r_side'
                if pose == 'r_side':
                    first_foot = predicted_keypoints[5] # right front foot
                    second_foot = predicted_keypoints[7]  # right back foot

                    for h in [first_foot, second_foot]:
                        if h.get('y') > 380:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                                
                    pose_output_image = pose_image_output(pose)
                    seg_output_image = seg_image_output(pose)
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
                        if h.get('y') > 380:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                    
                    pose_output_image = pose_image_output(pose)
                    seg_output_image = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
            else:
                pose=None
                
        
        elif filename == 'back_pose.jpg' and detected_object =='cow' and count == 1:
            if (predicted_keypoints[9].get('x') != 0 or predicted_keypoints[10].get('x') != 0):
                pose ='back'
                
                if pose == 'back':
                    first_foot = predicted_keypoints[6] # left back foot
                    second_foot = predicted_keypoints[7]  # right back hoof

                    for h in [first_foot, second_foot]:
                        if h.get('y') > 380:   # isko change krna hai height waali condition set krne k liye
                            height='normal'
                            
                        elif (h.get('x') or h.get('y')) == 0:
                            height ='Please retake the image as one of the hoof is not clearly visible'
                        
                        else:
                            height='Not Normal'
                            
                    pose_output_image = pose_image_output(pose)
                    seg_output_image = seg_image_output(pose)
                    weight_data = weight_estimation(predicted_keypoints, pixel_details, pose_bbox, org_image_shapes, pose)
                    
                
            else:
                pose=None
            
        else:
            pose='unidentified'
        

        return pose_output_image, height, detected_object, object_count, seg_output_image, weight_data, breed_output, image_caption

    except Exception as e:
        logging.exception("An error occurred in model_implementation: %s", str(e))
        raise
