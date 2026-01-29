"""
Aggregator Module
Compares predictions across multiple models and generates aggregation metrics with classification performance
"""

import pandas as pd
from typing import Dict
import logging
from collections import Counter
from sentiment_analysis.metrics import calculate_model_metrics

logger = logging.getLogger(__name__)


def calculate_model_agreement(
    results_df: pd.DataFrame,
    group_by_comment: bool = True
) -> Dict:
    """
    Calculate agreement metrics across models.
    
    Args:
        results_df: Results dataframe with all model predictions
        group_by_comment: If True, calculate per-comment agreement
        
    Returns:
        Dictionary with agreement statistics
    """
    if results_df.empty:
        return {}
    
    if group_by_comment:
        # Calculate agreement per comment
        agreement_per_comment = []
        
        for entity_id, group in results_df.groupby('entity_id'):
            comments_group = group[['model_name', 'sentiment_name']].drop_duplicates()
            
            if len(comments_group) > 1:
                sentiment_counts = comments_group['sentiment_name'].value_counts()
                # Agreement = max frequency / total models
                agreement = sentiment_counts.iloc[0] / len(comments_group)
            else:
                agreement = 1.0
            
            agreement_per_comment.append({
                'entity_id': entity_id,
                'num_models': len(comments_group),
                'agreement_score': agreement,
                'sentiment_distribution': comments_group['sentiment_name'].value_counts().to_dict()
            })
        
        return {
            "per_comment": agreement_per_comment,
            "average_agreement": sum(a['agreement_score'] for a in agreement_per_comment) / len(agreement_per_comment)
        }
    
    return {}


def generate_comparison_csv(
    results_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    model_metadata: Dict,
    output_path: str,
    model_metrics_dict: Dict = None
) -> bool:
    """
    Generate side-by-side comparison CSV of all models' predictions with classification metrics.
    Replaces agreement_score with classification metrics (accuracy, precision_macro, f1_macro, recall_macro).
    
    Args:
        results_df: Results dataframe
        comments_df: Original comments dataframe with ground truth labels
        model_metadata: Dictionary with per-model metadata
        output_path: Path for output CSV
        model_metrics_dict: Optional pre-calculated metrics for each model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a lookup for ground truth labels
        gt_lookup = {}
        for _, row in comments_df.iterrows():
            key = (str(row['entity_id']), str(row['comment']))
            gt_lookup[key] = {
                'ground_truth_sentiment': row['sentiment_class'],
                'ground_truth_name': row['sentiment_name']
            }
        
        # Calculate metrics for each model if not provided
        if model_metrics_dict is None:
            model_metrics_dict = {}
            for model_name in results_df['model_name'].unique():
                model_df = results_df[results_df['model_name'] == model_name].copy()
                
                # Add ground truth labels
                model_df['ground_truth_sentiment'] = model_df.apply(
                    lambda row: gt_lookup.get(
                        (str(row['entity_id']), str(row['comment'])),
                        {'ground_truth_sentiment': -1}
                    )['ground_truth_sentiment'],
                    axis=1
                )
                
                model_metrics_dict[model_name] = calculate_model_metrics(model_df, model_name)
        
        # Pivot to get one row per comment with columns for each model
        pivot_data = []
        
        for entity_id, comment_text in results_df[['entity_id', 'comment']].drop_duplicates().iterrows():
            entity_id_val = comment_text['entity_id']
            comment_text_val = comment_text['comment']
            
            comment_results = results_df[results_df['entity_id'] == entity_id_val]
            
            # Get ground truth for this comment
            gt_key = (str(entity_id_val), str(comment_text_val))
            gt_sentiment = gt_lookup.get(gt_key, {}).get('ground_truth_sentiment', -1)
            gt_name = gt_lookup.get(gt_key, {}).get('ground_truth_name', 'unknown')
            
            row = {
                'entity_id': entity_id_val,
                'comment': comment_text_val,
                'ground_truth_sentiment': gt_sentiment,
                'ground_truth_name': gt_name
            }
            
            # Add columns for each model
            for model_name in results_df['model_name'].unique():
                model_result = comment_results[comment_results['model_name'] == model_name]
                
                if not model_result.empty:
                    row[f'{model_name}_predicted_sentiment'] = model_result['sentiment_class'].iloc[0]
                    row[f'{model_name}_predicted_name'] = model_result['sentiment_name'].iloc[0]
                    row[f'{model_name}_confidence'] = model_result['confidence_score'].iloc[0]
                    row[f'{model_name}_inference_time_ms'] = model_result['inference_time_ms'].iloc[0]
                
                # Add classification metrics for this model
                if model_name in model_metrics_dict:
                    metrics = model_metrics_dict[model_name]
                    row[f'{model_name}_accuracy'] = metrics.get('accuracy', 0)
                    row[f'{model_name}_precision_macro'] = metrics.get('precision_macro', 0)
                    row[f'{model_name}_f1_macro'] = metrics.get('f1_macro', 0)
                    row[f'{model_name}_recall_macro'] = metrics.get('recall_macro', 0)
                
                # Add metadata columns for this model
                if model_name in model_metadata:
                    metadata = model_metadata[model_name]
                    row[f'{model_name}_device_used'] = metadata.get('device_used', 'unknown')
                    row[f'{model_name}_throughput_per_sec'] = metadata.get('throughput_per_sec', 0)
            
            pivot_data.append(row)
        
        df_comparison = pd.DataFrame(pivot_data)
        df_comparison.to_csv(output_path, index=False)
        
        logger.info(f"Comparison CSV saved to {output_path}")
        logger.info(f"Total comments compared: {len(df_comparison)}")
        logger.info(f"Total columns: {len(df_comparison.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating comparison CSV: {e}")
        return False


def get_model_consensus(results_df: pd.DataFrame) -> Dict:
    """
    Identify consensus predictions across models.
    
    Args:
        results_df: Results dataframe
        
    Returns:
        Dictionary with consensus information
    """
    consensus_data = []
    
    for entity_id in results_df['entity_id'].unique():
        comment_results = results_df[results_df['entity_id'] == entity_id]
        
        # Get all sentiments and their frequencies
        sentiments = comment_results['sentiment_name'].tolist()
        sentiment_counts = Counter(sentiments)
        most_common_sentiment, count = sentiment_counts.most_common(1)[0]
        
        # Consensus ratio
        consensus_ratio = count / len(sentiments)
        
        # Confidence scores
        avg_confidence = comment_results['confidence_score'].mean()
        
        consensus_data.append({
            'entity_id': entity_id,
            'consensus_sentiment': most_common_sentiment,
            'consensus_ratio': consensus_ratio,
            'agreeing_models': count,
            'total_models': len(sentiments),
            'avg_confidence': avg_confidence,
            'sentiment_distribution': dict(sentiment_counts)
        })
    
    return {
        "consensus": consensus_data,
        "unanimous_agreement_count": sum(1 for d in consensus_data if d['consensus_ratio'] == 1.0),
        "majority_agreement_count": sum(1 for d in consensus_data if d['consensus_ratio'] > 0.5),
        "high_confidence_count": sum(1 for d in consensus_data if d['avg_confidence'] > 0.8)
    }


def print_aggregation_summary(
    model_metrics_dict: Dict,
    model_metadata: Dict
) -> None:
    """
    Print classification metrics summary for all models.
    
    Args:
        model_metrics_dict: Dictionary with model metrics from metrics module
        model_metadata: Dictionary with per-model metadata
    """
    print("\n" + "="*80)
    print("MODEL CLASSIFICATION METRICS SUMMARY")
    print("="*80)
    
    # Create summary table
    metrics_data = []
    for model_name, metrics in model_metrics_dict.items():
        metrics_data.append({
            'Model': model_name.replace('/', '_'),
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision (Macro)': f"{metrics.get('precision_macro', 0):.4f}",
            'Recall (Macro)': f"{metrics.get('recall_macro', 0):.4f}",
            'F1 (Macro)': f"{metrics.get('f1_macro', 0):.4f}",
            'Correct': f"{metrics.get('correct_predictions', 0)}/{metrics.get('total_samples', 0)}",
            'Device': model_metadata.get(model_name, {}).get('device_used', 'unknown'),
            'Throughput': f"{model_metadata.get(model_name, {}).get('throughput_per_sec', 0):.1f} c/s"
        })
    
    # Print as table
    print("\nPer-Model Performance:")
    print("-" * 120)
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Correct', 'Device', 'Throughput']
    print(f"{headers[0]:<35} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<15} {headers[6]:<12} {headers[7]:<12}")
    print("-" * 120)
    
    for data in metrics_data:
        print(f"{data['Model']:<35} {data['Accuracy']:<12} {data['Precision (Macro)']:<12} "
              f"{data['Recall (Macro)']:<12} {data['F1 (Macro)']:<12} {data['Correct']:<15} "
              f"{data['Device']:<12} {data['Throughput']:<12}")
    
    print("="*80)
