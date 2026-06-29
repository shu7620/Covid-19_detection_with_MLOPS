import os
import pytest
from pathlib import Path
from cnnClassifier.config.configuration import ConfigurationManager
from model_architecture.model1_architecture import build_model
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

# 1. Test Data Integrity Gate (Passes because status.txt is whitelisted)
def test_data_validation_status():
    status_file_path = Path("artifacts/data_validation/status.txt")
    assert status_file_path.exists(), "Data validation status file missing."
    with open(status_file_path, "r") as f:
        status = f.read().split(":")[-1].strip()
    assert status == "True", "Data validation failed!"

# 2. Test Model Compilation and Tensor Shapes
def test_model_architecture_compilation():
    config_manager = ConfigurationManager()
    transformation_config = config_manager.get_data_transformation_config()
    input_shape = tuple(transformation_config.params_image_size)
    classes = 2
    model = build_model(input_shape=input_shape, classes=classes, learning_rate=0.001)
    assert model is not None
    assert model.output_shape == (None, classes)

# 3. NEW: Inference Pipeline Sanity Check (Replaces the heavy training test)
def test_prediction_pipeline_sanity():
    """
    Verifies that the inference engine can successfully ingest an image,
    preprocess it, and return a valid structural dictionary prediction.
    """
    # Point to one of your lightweight sample images
    sample_img_path = "tests/sample_data/normal_sample.png"
    
    # Skip the test if you haven't dropped a sample image in the folder yet
    if not os.path.exists(sample_img_path):
        pytest.skip("Sample test image not found in tests/sample_data/")
        
    predictor = PredictionPipeline(filename=sample_img_path)
    result = predictor.predict()
    
    assert isinstance(result, dict), "Prediction output should be a dictionary."
    assert "prediction" in result, "Prediction key missing from output."
    assert "confidence" in result, "Confidence key missing from output."
    logger.info("Pytest Gate: Container inference verification successful.")