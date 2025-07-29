
# CHEST X-RAY BASELINE SOLUTION - DÉPLOYABLE
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ChestXRayClassifier:
    def __init__(self, model_path="chest_xray_final_baseline.pkl"):
        self.model = joblib.load(model_path)
        self.accuracy = "83.5%"
        
    def predict(self, image_path):
        # Préprocessing
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_flat = img_array.reshape(1, -1)
        
        # Prédiction
        pred = self.model.predict(img_flat)[0]
        proba = self.model.predict_proba(img_flat)[0]
        
        return {
            'prediction': 'Pneumonia' if pred == 1 else 'Normal',
            'confidence': proba.max(),
            'probabilities': {'Normal': proba[0], 'Pneumonia': proba[1]},
            'model_accuracy': self.accuracy
        }
        
    def batch_predict(self, image_paths):
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

# Usage example:
# classifier = ChestXRayClassifier()
# result = classifier.predict("chest_xray.jpg")
# print(result)
