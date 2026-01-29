"""
Classification Metrics Module
Calculates comprehensive classification metrics for sentiment analysis models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: list,
    y_pred: list,
    class_labels: list = None
) -> Dict:
    """
    Calculate comprehensive classification metrics for 3-class sentiment.
    
    Args:
        y_true: Ground truth sentiment classes (list of 0, 1, 2)
        y_pred: Predicted sentiment classes (list of 0, 1, 2)
        class_labels: Optional list of class names (default: ['positive', 'neutral', 'negative'])
        
    Returns:
        Dictionary containing:
        - confusion_matrix: 3x3 confusion matrix
        - accuracy: Overall accuracy
        - precision_per_class: Dict with per-class precision
        - recall_per_class: Dict with per-class recall
        - f1_per_class: Dict with per-class F1-score
        - precision_macro: Macro-averaged precision
        - recall_macro: Macro-averaged recall
        - f1_macro: Macro-averaged F1-score
        - support: Number of samples per class
        - classification_report: Detailed classification report
    """
    if class_labels is None:
        class_labels = ["positive", "neutral", "negative"]
    
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    
    if len(y_true) == 0:
        logger.warning("Empty arrays provided for metrics calculation")
        return {}
    
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        support_per_class = {}
        
        # Calculate metrics for all classes at once
        precision_scores = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        
        for i, class_name in enumerate(class_labels):
            precision_per_class[class_name] = precision_scores[i]
            recall_per_class[class_name] = recall_scores[i]
            f1_per_class[class_name] = f1_scores[i]
            # Support is the count of true instances for each class
            support_per_class[class_name] = int((np.array(y_true) == i).sum())
        
        # Calculate macro averages
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            labels=[0, 1, 2],
            target_names=class_labels,
            output_dict=True,
            zero_division=0
        )
        
        return {
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "support": support_per_class,
            "classification_report": report
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}


def identify_misclassified_samples(
    y_true: list,
    y_pred: list,
    texts: list = None,
    entity_ids: list = None,
    class_labels: list = None
) -> pd.DataFrame:
    """
    Identify and return all misclassified samples.
    
    Args:
        y_true: Ground truth sentiment classes
        y_pred: Predicted sentiment classes
        texts: Optional list of comment texts
        entity_ids: Optional list of entity IDs
        class_labels: Optional list of class names
        
    Returns:
        DataFrame with misclassified samples
    """
    if class_labels is None:
        class_labels = ["positive", "neutral", "negative"]
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    # Find misclassified indices
    misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    
    error_data = {
        "ground_truth_class": [y_true[i] for i in misclassified_indices],
        "ground_truth_label": [class_labels[y_true[i]] for i in misclassified_indices],
        "predicted_class": [y_pred[i] for i in misclassified_indices],
        "predicted_label": [class_labels[y_pred[i]] for i in misclassified_indices]
    }
    
    if texts and len(texts) == len(y_true):
        error_data["comment"] = [texts[i] for i in misclassified_indices]
    
    if entity_ids and len(entity_ids) == len(y_true):
        error_data["entity_id"] = [entity_ids[i] for i in misclassified_indices]
    
    error_df = pd.DataFrame(error_data)
    
    logger.info(f"Identified {len(error_df)} misclassified samples out of {len(y_true)}")
    
    return error_df


def calculate_model_metrics(
    predictions_df: pd.DataFrame,
    model_name: str,
    class_labels: list = None
) -> Dict:
    """
    Calculate metrics for a model's predictions.
    
    Args:
        predictions_df: DataFrame with columns 'ground_truth_sentiment' and 'predicted_sentiment'
        model_name: Name of the model
        class_labels: Optional list of class names
        
    Returns:
        Dictionary with all metrics and metadata
    """
    if class_labels is None:
        class_labels = ["positive", "neutral", "negative"]
    
    y_true = predictions_df['ground_truth_sentiment'].tolist()
    y_pred = predictions_df['predicted_sentiment'].tolist()
    
    # Get texts and entity IDs if available
    texts = predictions_df['comment'].tolist() if 'comment' in predictions_df.columns else None
    entity_ids = predictions_df['entity_id'].tolist() if 'entity_id' in predictions_df.columns else None
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_true, y_pred, class_labels)
    
    # Identify misclassified samples
    error_df = identify_misclassified_samples(y_true, y_pred, texts, entity_ids, class_labels)
    
    # Add metadata
    metrics['model_name'] = model_name
    metrics['total_samples'] = len(predictions_df)
    metrics['correct_predictions'] = (predictions_df['ground_truth_sentiment'] == predictions_df['predicted_sentiment']).sum()
    metrics['incorrect_predictions'] = len(predictions_df) - metrics['correct_predictions']
    metrics['error_analysis'] = error_df
    
    return metrics


def print_metrics_summary(metrics: Dict) -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Metrics dictionary from calculate_classification_metrics()
    """
    print("\n" + "="*80)
    print(f"CLASSIFICATION METRICS")
    print("="*80)
    
    print(f"\nAccuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1-score: {metrics.get('f1_macro', 0):.4f}")
    print(f"Macro Recall: {metrics.get('recall_macro', 0):.4f}")
    print(f"Macro Precision: {metrics.get('precision_macro', 0):.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
    print("-" * 58)
    
    for class_name in ["positive", "neutral", "negative"]:
        precision = metrics.get('precision_per_class', {}).get(class_name, 0)
        recall = metrics.get('recall_per_class', {}).get(class_name, 0)
        f1 = metrics.get('f1_per_class', {}).get(class_name, 0)
        support = metrics.get('support', {}).get(class_name, 0)
        
        print(f"{class_name:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    print("\nConfusion Matrix:")
    cm = metrics.get('confusion_matrix', np.zeros((3, 3)))
    print("                Predicted")
    print("                Positive  Neutral   Negative")
    for i, class_name in enumerate(["Positive", "Neutral", "Negative"]):
        print(f"Actual {class_name:<5} {cm[i, 0]:<10} {cm[i, 1]:<10} {cm[i, 2]:<10}")
    
    print("="*80)
