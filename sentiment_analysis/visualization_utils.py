"""
Visualization Utilities Module
Creates matplotlib visualizations for classification metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import logging
import os

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list = None,
    title: str = "Confusion Matrix",
    output_path: str = None,
    cmap: str = "Blues"
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix (3x3 numpy array)
        class_labels: List of class names (default: ['Positive', 'Neutral', 'Negative'])
        title: Title for the plot
        output_path: Path to save the figure
        cmap: Colormap for heatmap
    """
    if class_labels is None:
        class_labels = ["Positive", "Neutral", "Negative"]
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


def plot_classification_report(
    report_dict: Dict,
    class_labels: list = None,
    title: str = "Classification Report",
    output_path: str = None
) -> None:
    """
    Plot classification report (precision, recall, F1-score per class).
    
    Args:
        report_dict: Classification report dictionary from sklearn
        class_labels: List of class names
        title: Title for the plot
        output_path: Path to save the figure
    """
    if class_labels is None:
        class_labels = ["Positive", "Neutral", "Negative"]
    
    try:
        # Extract metrics for each class
        metrics = ['precision', 'recall', 'f1-score']
        data = {}
        
        for metric in metrics:
            data[metric] = [report_dict.get(label, {}).get(metric, 0) for label in class_labels]
        
        # Create dataframe for easier plotting
        import pandas as pd
        df = pd.DataFrame(data, index=class_labels)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        x = np.arange(len(class_labels))
        width = 0.25
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1)
            ax.bar(x + offset, df[metric], width, label=metric, color=colors[i], alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            offset = width * (i - 1)
            for j, v in enumerate(df[metric]):
                ax.text(j + offset, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification report saved to {output_path}")
        
        plt.close()
    
    except Exception as e:
        logger.error(f"Error plotting classification report: {e}")


def plot_metrics_summary(
    accuracy: float,
    f1_macro: float,
    recall_macro: float,
    precision_macro: float,
    title: str = "Metrics Summary",
    output_path: str = None
) -> None:
    """
    Plot summary metrics (accuracy, macro F1, macro recall, macro precision).
    
    Args:
        accuracy: Accuracy score
        f1_macro: Macro-averaged F1-score
        recall_macro: Macro-averaged recall
        precision_macro: Macro-averaged precision
        title: Title for the plot
        output_path: Path to save the figure
    """
    try:
        metrics_names = ['Accuracy', 'Macro F1', 'Macro Recall', 'Macro Precision']
        metrics_values = [accuracy, f1_macro, recall_macro, precision_macro]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#06A77D', '#1B9A8B', '#2E86AB', '#A23B72']
        bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics summary saved to {output_path}")
        
        plt.close()
    
    except Exception as e:
        logger.error(f"Error plotting metrics summary: {e}")


def plot_support_distribution(
    support: Dict,
    title: str = "Class Support Distribution",
    output_path: str = None
) -> None:
    """
    Plot class distribution (support) as a bar chart.
    
    Args:
        support: Dictionary with class names and sample counts
        title: Title for the plot
        output_path: Path to save the figure
    """
    try:
        classes = list(support.keys())
        counts = list(support.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#06A77D', '#1B9A8B', '#2E86AB']
        bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Support distribution saved to {output_path}")
        
        plt.close()
    
    except Exception as e:
        logger.error(f"Error plotting support distribution: {e}")


def generate_all_visualizations(
    metrics: Dict,
    output_dir: str,
    model_name: str = ""
) -> None:
    """
    Generate all metric visualizations for a model.
    
    Args:
        metrics: Metrics dictionary from metrics.calculate_classification_metrics()
        output_dir: Directory to save visualizations
        model_name: Name of the model (used in plot titles)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate confusion matrix
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        cm_path = os.path.join(output_dir, f"confusion_matrix_{safe_model_name}.png")
        plot_confusion_matrix(
            metrics.get('confusion_matrix', np.zeros((3, 3))),
            class_labels=["Positive", "Neutral", "Negative"],
            title=f"Confusion Matrix - {model_name}",
            output_path=cm_path
        )
        
        # Generate classification report
        report_path = os.path.join(output_dir, f"classification_report_{safe_model_name}.png")
        report_dict = metrics.get('classification_report', {})
        if report_dict:
            plot_classification_report(
                report_dict,
                class_labels=["Positive", "Neutral", "Negative"],
                title=f"Classification Report - {model_name}",
                output_path=report_path
            )
        
        # Generate metrics summary
        summary_path = os.path.join(output_dir, f"metrics_summary_{safe_model_name}.png")
        plot_metrics_summary(
            accuracy=metrics.get('accuracy', 0),
            f1_macro=metrics.get('f1_macro', 0),
            recall_macro=metrics.get('recall_macro', 0),
            precision_macro=metrics.get('precision_macro', 0),
            title=f"Metrics Summary - {model_name}",
            output_path=summary_path
        )
        
        # Generate support distribution
        support_path = os.path.join(output_dir, f"support_distribution_{safe_model_name}.png")
        plot_support_distribution(
            support=metrics.get('support', {}),
            title=f"Class Support Distribution - {model_name}",
            output_path=support_path
        )
        
        logger.info(f"Generated all visualizations for {model_name} in {output_dir}")
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
