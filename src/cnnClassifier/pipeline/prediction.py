import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# UPDATE: Import directly from keras.utils to support modern Keras layout
from tensorflow.keras.utils import load_img, img_to_array
from cnnClassifier import logger

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        logger.info("Initializing prediction pipeline inference execution...")
        
        # 1. Load the production model artifact
        model_path = os.path.join("artifacts", "training", "model.keras")
        model = load_model(model_path)

        # 2. Preprocess the raw input image using updated utility functions
        # Call the direct methods instead of the legacy image object namespace
        test_image = load_img(self.filename, target_size=(224, 224))
        test_image = img_to_array(test_image)
        
        # Scale pixel arrays exactly like the data transformation layer (1./255)
        test_image = test_image / 255.0
        
        # Expand dimensions to add the batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        test_image = np.expand_dims(test_image, axis=0)

        # 3. Compute inference probability values
        predictions = model.predict(test_image, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]

        # Categorical map mapping: 0 -> COVID, 1 -> Normal
        if predicted_class_idx == 0:
            prediction_label = "COVID-19 Positive"
            confidence = float(predictions[0][0])
        else:
            prediction_label = "Normal / Healthy"
            confidence = float(predictions[0][1])

        logger.info(f"Prediction result calculated: {prediction_label} with confidence {confidence:.2%}")
        
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.2%}"
        }