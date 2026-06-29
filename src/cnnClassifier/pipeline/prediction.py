import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from cnnClassifier import logger

class PredictionPipeline:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model  # Pass the pre-loaded global model here

    def predict(self):
        logger.info("Executing fast inference from global model cache...")
        
        # Preprocess image
        test_image = load_img(self.filename, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Run inference using the cached model
        predictions = self.model.predict(test_image, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]

        if predicted_class_idx == 0:
            prediction_label = "COVID-19 Positive"
            confidence = float(predictions[0][0])
        else:
            prediction_label = "Normal / Healthy"
            confidence = float(predictions[0][1])

        logger.info(f"Prediction calculated: {prediction_label} ({confidence:.2%})")
        
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.2%}"
        }