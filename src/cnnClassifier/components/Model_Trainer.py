import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig
from model_architecture.model2_architecture import build_model
import os
import json



class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        # 1. Load the transformed datasets
        train_ds = tf.data.Dataset.load(str(self.config.training_data))
        val_ds=tf.data.Dataset.load(str(self.config.val_data))
        
        # 2. Build the model from the architecture file
        model = build_model(
            input_shape=tuple(self.config.params_image_size),
            classes=self.config.params_classes,
            learning_rate=0.001
        )
        
        # 3. Train
        # model.fit(
        #     train_ds,
        #     epochs=self.config.params_epochs
        # )
        # Inside your train() method in model_trainer.py:
        history = model.fit(train_ds, validation_data=val_ds, epochs=self.config.params_epochs)

        # Save history dictionary as json
        history_path = os.path.join(self.config.root_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
        
        # 4. Save model
        model.save(self.config.trained_model_path)