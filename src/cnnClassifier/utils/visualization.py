import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_training_curves(history_json_path: str, save_dir: str):
    """Generates and saves Training vs Validation Accuracy and Loss curves."""
    with open(history_json_path, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history['accuracy']) + 1)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()


def generate_confusion_matrix_plot(y_true, y_pred, classes, save_dir: str):
    """Generates and saves a clean Seaborn heatmap Confusion Matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()