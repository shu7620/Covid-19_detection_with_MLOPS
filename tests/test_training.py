import os
import sys
import pytest
import tensorflow as tf
from pathlib import Path

from cnnClassifier.config.configuration import ConfigurationManager
from model_architecture.model1_architecture import build_model
from cnnClassifier import logger

# 1. Test Data Integrity Gate
def test_data_validation_status():
    """
    Ensures that the Data Validation stage successfully ran 
    and left a 'True' status pass indicator for the training pipeline.
    """
    status_file_path = Path("artifacts/data_validation/status.txt")
    
    # Assert that the validation phase was executed
    assert status_file_path.exists(), "Data validation status file does not exist. Run validation stage first."
    
    with open(status_file_path, "r") as f:
        status = f.read().split(":")[-1].strip()
        
    # The training phase should lock up if validation fails
    assert status == "True", "Data validation failed! Cannot proceed with training pipeline tests."
    logger.info("Pytest Gate: Data validation status verified successfully.")


# 2. Test Model Compilation and Tensor Shapes
def test_model_architecture_compilation():
    """
    Verifies that the decoupled CNN model architecture compiles properly
    and matches the expected classification layer output shape.
    """
    config_manager = ConfigurationManager()
    transformation_config = config_manager.get_data_transformation_config()
    
    # Extract structural expectations from params.yaml via entity config
    input_shape = tuple(transformation_config.params_image_size) # e.g., (224, 224, 3)
    classes = 2 # Binary classification (COVID, Normal)
    learning_rate = 0.001
    
    model = build_model(input_shape=input_shape, classes=classes, learning_rate=learning_rate)
    
    # Verify model compile properties
    assert model is not None, "Failed to compile the CNN model architecture."
    assert model.output_shape == (None, classes), f"Expected output tensor shape (None, {classes}), but got {model.output_shape}"
    logger.info("Pytest Gate: Model architecture and output tensor shapes verified.")


# 3. Continuous Training Sanity Execution
def test_pipeline_training_sanity():
    """
    Executes a miniature 1-batch execution run to verify that backpropagation 
    and weight updating loops do not throw gradient execution or shape errors.
    """
    config_manager = ConfigurationManager()
    training_config = config_manager.get_training_config()
    
    # Load the transformed training datasets
    assert os.path.exists(training_config.training_data), "Transformed training dataset directory is missing."
    train_ds = tf.data.Dataset.load(str(training_config.training_data))
    
    # Extract exactly 1 mini-batch to run a fast CPU sanity check
    mini_sample_ds = train_ds.take(1)
    
    model = build_model(
        input_shape=tuple(training_config.params_image_size),
        classes=training_config.params_classes,
        learning_rate=0.001
    )
    
    # Train for exactly 1 epoch over the single mini-batch
    history = model.fit(mini_sample_ds, epochs=1, verbose=0)
    
    # Assert training completed and recorded metrics without crashing or exploding to NaN
    assert "loss" in history.history, "Training execution failed to record loss metrics."
    assert not tf.math.is_nan(history.history["loss"][0]), "Sanity training failed: Loss values exploded to NaN."
    logger.info("Pytest Gate: Pipeline training execution check completed without exceptions.")