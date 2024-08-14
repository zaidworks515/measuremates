import unittest
import joblib
from flask import g
from app import app, weight_estimation, price_calculation


class TestWeightEstimation(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.weight_model = joblib.load('static/models/cow_weight_predictor(ExtraTreesRegressor+Transformer).pkl')
        g.model4 = self.weight_model

    def tearDown(self):
        self.app_context.pop()

    def test_weight_estimation(self):
        predicted_keypoints = [
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0}
        ]
        pixel_details = [1000, 500]
        pose_bbox = [50, 50, 200, 200]
        org_image_shapes = (3048, 4064)
        pose = 'side'

        weight = weight_estimation(
            predicted_keypoints=predicted_keypoints,
            pixel_details=pixel_details,
            pose_bbox=pose_bbox,
            org_image_shapes=org_image_shapes,
            pose=pose
        )
        
        self.assertIsInstance(weight, (int, float))
        self.assertGreater(weight, 0)

    def test_price_calculation(self):
        # Test with specific weight values
        test_weights = [10, 50, 100]  
        for weight in test_weights:
            with self.subTest(weight=weight):
                price = price_calculation(weight)
                self.assertIsInstance(price, int)
                self.assertGreater(price, 0)

if __name__ == '__main__':
    unittest.main()

