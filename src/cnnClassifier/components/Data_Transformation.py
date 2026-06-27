import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save_data(self):
        logger.info(f"Loading and processing data from {self.config.data_path}")
        
        img_shape = tuple(self.config.params_image_size[:-1])

        # 1. Load, Resize, and Encode Labels
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_path,
            validation_split=self.config.params_validation_split,
            subset="training",
            seed=123,
            image_size=img_shape,
            batch_size=self.config.params_batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_path,
            validation_split=self.config.params_validation_split,
            subset="validation",
            seed=123,
            image_size=img_shape,
            batch_size=self.config.params_batch_size,
            label_mode='categorical'
        )

        # 2. Normalization
        logger.info("Applying normalization transformation...")
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # --- NEW LOGGING CODE ---
        # Log the shape specifications of the batches (Features and Labels)
        train_features_spec, train_labels_spec = train_ds.element_spec
        val_features_spec, val_labels_spec = val_ds.element_spec
        
        logger.info(f"Training Batch Image Shape: {train_features_spec.shape}")
        logger.info(f"Training Batch Label Shape: {train_labels_spec.shape}")
        logger.info(f"Validation Batch Image Shape: {val_features_spec.shape}")
        # ------------------------

        # 3. Optimize for Performance
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 4. Save Transformed Datasets
        logger.info(f"Saving transformed datasets...")
        train_ds.save(str(self.config.train_dataset_path))
        val_ds.save(str(self.config.val_dataset_path))
        
        logger.info("Data Preprocessing and Transformation completed successfully.")