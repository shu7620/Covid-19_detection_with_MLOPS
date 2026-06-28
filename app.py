import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for API connectivity

UPLOAD_FOLDER = os.path.join("artifacts", "user_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    """Renders the central user landing dashboard page UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """Accepts image upload forms and routes data into the prediction layer."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file payload detected inside request forms"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No valid filename selected"}), 400

        if file:
            # Secure image destination path
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Execute inference on user-uploaded file path
            pipeline = PredictionPipeline(filepath)
            prediction_result = pipeline.predict()
            
            # Clean up disk memory space post-inference
            os.remove(filepath)
            
            return jsonify(prediction_result)

    except Exception as e:
        logger.exception(f"Inference failure encountered: {str(e)}")
        return jsonify({"error": "Internal processing exception occurred during prediction."}), 500

if __name__ == "__main__":
    # Standard production-ready host parameters mapped for cloud deployments
    app.run(host="0.0.0.0", port=8080)