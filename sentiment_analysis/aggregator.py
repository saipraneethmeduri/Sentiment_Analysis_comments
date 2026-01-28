"""
Aggregator Module
Compares predictions across multiple models and generates aggregation metrics
"""

import pandas as pd
from typing import Dict
import logging
from collections import Counter

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
    model_metadata: Dict,
    output_path: str
) -> bool:
    """
    Generate side-by-side comparison CSV of all models' predictions.
    
    Args:
        results_df: Results dataframe
        model_metadata: Dictionary with per-model metadata
        output_path: Path for output CSV
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Pivot to get one row per comment with columns for each model
        pivot_data = []
        
        for entity_id, comment_text in results_df[['entity_id', 'comment']].drop_duplicates().iterrows():
            entity_id_val = comment_text['entity_id']
            comment_text_val = comment_text['comment']
            
            comment_results = results_df[results_df['entity_id'] == entity_id_val]
            
            row = {
                'entity_id': entity_id_val,
                'comment': comment_text_val
            }
            
            # Add columns for each model
            for model_name in results_df['model_name'].unique():
                model_result = comment_results[comment_results['model_name'] == model_name]
                
                if not model_result.empty:
                    row[f'{model_name}_sentiment'] = model_result['sentiment_name'].iloc[0]
                    row[f'{model_name}_confidence'] = model_result['confidence_score'].iloc[0]
                    row[f'{model_name}_latency_ms'] = model_result['inference_time_ms'].iloc[0]
                
                # Add metadata columns for this model (same for all rows)
                if model_name in model_metadata:
                    metadata = model_metadata[model_name]
                    row[f'{model_name}_total_comments_processed'] = metadata.get('total_comments_processed', 0)
                    row[f'{model_name}_avg_latency_ms'] = metadata.get('avg_latency_ms', 0)
                    row[f'{model_name}_throughput_per_sec'] = metadata.get('throughput_per_sec', 0)
                    row[f'{model_name}_total_time_ms'] = metadata.get('total_time_ms', 0)
                    row[f'{model_name}_load_time_ms'] = metadata.get('load_time_ms', 0)
                    row[f'{model_name}_device_used'] = metadata.get('device_used', 'unknown')
            
            # Calculate agreement
            sentiments = [row.get(f'{m}_sentiment') for m in results_df['model_name'].unique()]
            sentiments = [s for s in sentiments if s is not None]
            if sentiments:
                sentiment_counts = Counter(sentiments)
                agreement = sentiment_counts.most_common(1)[0][1] / len(sentiments)
                row['model_agreement_score'] = agreement
            
            pivot_data.append(row)
        
        df_comparison = pd.DataFrame(pivot_data)
        df_comparison.to_csv(output_path, index=False)
        
        logger.info(f"Comparison CSV saved to {output_path}")
        logger.info(f"Total comments compared: {len(df_comparison)}")
        
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
    agreement_metrics: Dict,
    consensus_metrics: Dict
) -> None:
    """
    Print aggregation and consensus summary.
    
    Args:
        agreement_metrics: From calculate_model_agreement()
        consensus_metrics: From get_model_consensus()
    """
    print("\n" + "="*80)
    print("MODEL AGREEMENT AND CONSENSUS ANALYSIS")
    print("="*80)
    
    if agreement_metrics and "average_agreement" in agreement_metrics:
        print(f"\nAverage Model Agreement Score: {agreement_metrics['average_agreement']:.2%}")
    
    if consensus_metrics:
        total = len(consensus_metrics['consensus'])
        unanimous = consensus_metrics['unanimous_agreement_count']
        majority = consensus_metrics['majority_agreement_count']
        
        print(f"\nConsensus Statistics:")
        print(f"  Unanimous Agreement (all models agree): {unanimous}/{total} ({unanimous/total*100:.1f}%)")
        print(f"  Majority Agreement (>50% models agree): {majority}/{total} ({majority/total*100:.1f}%)")
        print(f"  High Confidence Predictions (>0.8): {consensus_metrics['high_confidence_count']}/{total} ({consensus_metrics['high_confidence_count']/total*100:.1f}%)")
    
    print("="*80)
