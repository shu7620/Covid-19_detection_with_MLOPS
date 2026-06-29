import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("artifacts", "user_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OPTIMIZATION: Load the model once globally at server startup
MODEL_PATH = os.path.join("artifacts", "training", "model.keras")
logger.info("Caching model architecture into system RAM memory...")
global_model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    filepath = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file payload detected"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No valid filename selected"}), 400

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Pass the globally cached model directly to the pipeline instance
            pipeline = PredictionPipeline(filepath, global_model)
            prediction_result = pipeline.predict()
            
            return jsonify(prediction_result)

    except Exception as e:
        logger.exception(f"Inference failure encountered: {str(e)}")
        return jsonify({"error": "Internal processing exception occurred during prediction."}), 500
        
    finally:
        # SAFE DELETION: Cleans up the image regardless of success or duplicate concurrent runs
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned upload artifact: {filepath}")
            except OSError:
                pass # Fail silently if another thread deleted it first
                
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)