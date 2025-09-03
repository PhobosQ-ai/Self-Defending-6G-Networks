"""
Module for generating plots related to CIC-IoT-2023 dataset analysis.
This includes label distributions, confusion matrices, and other visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_label_distribution(df: pd.DataFrame, title: str, save_path: str = None):
    """
    Plot the distribution of attack labels in the dataset.
    
    Args:
        df: DataFrame containing the 'label' column
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(data=df, y='label', order=df['label'].value_counts().index)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Attack Type', fontsize=14)
    
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.1 * max(df['label'].value_counts()), 
                 p.get_y() + p.get_height()/2., 
                 f'{int(width)}', 
                 ha='center', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title: str = 'Confusion Matrix', save_path: str = None):
    """
    Plot a confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_image(feature_vector, title: str = 'Generated Image from Features', save_path: str = None):
    """
    Plot an image generated from feature vector (as in the notebook's VGG16 example).
    
    Args:
        feature_vector: Array of features
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    normalized_features = ((feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min()) * 255).astype(np.uint8)
    
    height, width = 5, 3
    if len(normalized_features) < height * width:
        normalized_features = np.pad(normalized_features, (0, height * width - len(normalized_features)), 'constant')
    image_array_1d = normalized_features[:height * width].reshape((height, width))
    image_array_rgb = np.stack([image_array_1d] * 3, axis=-1)
    
    plt.imshow(image_array_rgb)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_plots(data_train: pd.DataFrame, data_test: pd.DataFrame, y_true=None, y_pred=None, class_names=None):
    """
    Generate all plots from the notebook analysis.
    
    Args:
        data_train: Processed training DataFrame
        data_test: Processed test DataFrame
        y_true: True labels for confusion matrix (optional)
        y_pred: Predicted labels for confusion matrix (optional)
        class_names: List of class names (optional)
    """
    logger.info("Generating plots...")
    
    plot_label_distribution(data_train, 'Distribution of Attack Labels in Train Set', 'train_label_distribution.png')
    plot_label_distribution(data_test, 'Distribution of Attack Labels in Test Set', 'test_label_distribution.png')
    
    if y_true is not None and y_pred is not None and class_names is not None:
        plot_confusion_matrix(y_true, y_pred, class_names, 'Confusion Matrix (Simulation)', 'confusion_matrix.png')
    
    if not data_train.empty:
        sample_features = data_train.drop('label', axis=1).iloc[0].values
        plot_feature_image(sample_features, 'Generated Image from Sample Features', 'feature_image.png')
    
    logger.info("Plots generated successfully.")

if __name__ == "__main__":
    logger.info("Run this from main.py after data loading.")
