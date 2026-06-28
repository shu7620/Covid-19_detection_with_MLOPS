from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    final_dataset_dir: Path
    

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_classes: list
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_dataset_path: Path
    val_dataset_path: Path
    params_image_size: list
    params_batch_size: int
    params_validation_split: float
    
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    val_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_classes: int
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    history_path: Path
    plots_dir: Path
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int