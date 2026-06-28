import os
import json
import tensorflow as tf
import numpy as np
import mlflow
import mlflow.keras
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.visualization import generate_training_curves, generate_confusion_matrix_plot
import dagshub

load_dotenv()

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        """Runs validation calculations, generates assets, and tracking records."""
        # 1. Load data and model
        logger.info("Loading validation data and trained model assets...")
        test_ds = tf.data.Dataset.load(str(self.config.test_data_path))
        model = tf.keras.models.load_model(str(self.config.model_path))

        # 2. Extract predictions for medical matrix calculations
        logger.info("Generating predictions across test dataset tensors...")
        y_true = []
        y_pred_probs = []

        for images, labels in test_ds:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # <--  INITIALIZE DAGSHUB HERE -->
        dagshub.init(repo_owner='shu7620', repo_name='End-to-End-ML-project', mlflow=True)

        # 3. Compute Crucial Medical Metrics
        accuracy = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=0) # Assuming 0 is COVID
        recall = recall_score(y_true, y_pred, average='binary', pos_label=0)       # Sensitivity
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=0)
        auc_roc = roc_auc_score(y_true, y_pred_probs[:, 0])

        metrics_dict = {
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall_sensitivity": recall,
            "val_f1_score": f1,
            "val_auc_roc": auc_roc
        }
        
        params_dict = {
            "image_size": str(self.config.params_image_size),  # Convert list to string for MLflow compatibility
            "batch_size": self.config.params_batch_size,
        }

        # 4. Invoke Isolated Plotting Module
        logger.info("Calling internal plotting utilities to isolate diagnostic assets...")
        generate_training_curves(str(self.config.history_path), str(self.config.plots_dir))
        generate_confusion_matrix_plot(y_true, y_pred, classes=["COVID", "Normal"], save_dir=str(self.config.plots_dir))

        # 5. Connect to MLflow remote server via DagsHub
        if self.config.mlflow_uri != "":
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
        # Ensure experiment exists
        mlflow.set_experiment("COVID19_XRay_Classification")

        with mlflow.start_run():
            logger.info("Logging runtime parameters and diagnostic metrics to MLflow...")
            
            # Log Parameters
            mlflow.log_params(params_dict)
            
            # Log Metrics
            mlflow.log_metrics(metrics_dict)
            
            # Log Plots as Artifacts
            mlflow.log_artifacts(str(self.config.plots_dir), artifact_path="evaluation_plots")
            
            # Log Model Architecture
            mlflow.keras.log_model(model, "model", registered_model_name="CNN_Covid_Detection_Model")
            
        logger.info("Evaluation metrics successfully tracked and artifacts archived.")