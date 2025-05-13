import unittest
from Zoidberg.mobilenet_inference import predict

class TestMobileNetInference(unittest.TestCase):
    
    def test_prediction_returns_string(self):
        image_path = "data/test.png"  # Chemin image test existante
        result = predict(image_path)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

if __name__ == "__main__":
    unittest.main()
