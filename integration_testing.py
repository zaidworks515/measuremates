import os
import pytest
import json
import base64
from flask import Flask
from flask_testing import TestCase
from app import app  

class TestFlaskApp(TestCase):

    def create_app(self):
        # Create a new Flask app instance for testing
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        return app

    def setUp(self):
        self.test_image_path = 'tests/side_pose.jpg'

    def tearDown(self):
        # Clean up after tests
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_home_page(self):
        response = self.client.get('/')
        self.assert200(response)
        self.assertIn(b'Welcome to the home page', response.data)

    def test_image_upload(self):
        with open(self.test_image_path, 'rb') as image_file:
            response = self.client.post(
                '/evaluate',
                data={'images': (image_file, 'side_pose.jpg')},
                content_type='multipart/form-data'
            )
        self.assert200(response)
        response_json = json.loads(response.data)
        self.assertIn('pose', response_json)
        self.assertIn('weight', response_json)

    def test_weight_estimation(self):
        predicted_keypoints = [dict(x=1.0, y=2.0, z=3.0) for _ in range(15)]
        pixel_details = [1000, 500]
        pose_bbox = [10, 20, 30, 40]
        org_image_shapes = [640, 480]
        pose = 'side'
        
        weight = self.app.test_client().get('/weight_estimation', query_string=dict(
            predicted_keypoints=json.dumps(predicted_keypoints),
            pixel_details=json.dumps(pixel_details),
            pose_bbox=json.dumps(pose_bbox),
            org_image_shapes=json.dumps(org_image_shapes),
            pose=pose
        ))
        self.assertEqual(weight.status_code, 200)
        weight_json = json.loads(weight.data)
        self.assertIn('weight', weight_json)

    def test_price_calculation(self):
        weight = 250.0
        response = self.client.get('/price_calculation', query_string=dict(weight=weight))
        self.assert200(response)
        response_json = json.loads(response.data)
        self.assertIn('price', response_json)

if __name__ == '__main__':
    pytest.main()
