"""
Results Storage Module
Stores sentiment analysis results to CSV with all metrics
"""

import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def prepare_results_dataframe(
    comments_df: pd.DataFrame,
    model_results: Dict,
    normalized_sentiments: Dict
) -> tuple:
    """
    Prepare comprehensive results dataframe with all metrics.
    
    Args:
        comments_df: Original comments dataframe
        model_results: Results from batch_inference.process_all_models()
        normalized_sentiments: Dict with model_name -> {classes, names} mapping
        
    Returns:
        Tuple of (results_df, model_metadata_dict)
    """
    results_rows = []
    model_metadata = {}
    
    # Get comment data
    entity_ids = comments_df['entity_id'].astype(str).tolist()
    comments = comments_df['comment'].astype(str).tolist()
    num_comments = len(comments)
    
    # Process each model's results
    for model_name, model_info in model_results.items():
        inference_results = model_info["inference_results"]
        labels = inference_results["labels"]
        scores = inference_results["scores"]
        latencies = inference_results["latencies"]
        
        # Get normalized sentiments for this model
        if model_name in normalized_sentiments:
            sentiment_classes = normalized_sentiments[model_name]["classes"]
            sentiment_names = normalized_sentiments[model_name]["names"]
        else:
            sentiment_classes = [1] * len(labels)  # Default to neutral
            sentiment_names = ["neutral"] * len(labels)
        
        # Build rows for this model
        for i in range(len(comments)):
            row = {
                "entity_id": entity_ids[i],
                "comment": comments[i],
                "model_name": model_name,
                "sentiment_class": sentiment_classes[i] if i < len(sentiment_classes) else 1,
                "sentiment_name": sentiment_names[i] if i < len(sentiment_names) else "neutral",
                "confidence_score": scores[i] if i < len(scores) else 0.0,
                "inference_time_ms": latencies[i] if i < len(latencies) else 0.0
            }
            results_rows.append(row)
        
        # Store metadata for this model (used later in comparison CSV)
        model_metadata[model_name] = {
            "total_comments_processed": num_comments,
            "avg_latency_ms": model_info.get("avg_latency_ms", 0),
            "throughput_per_sec": model_info.get("throughput_per_sec", 0),
            "total_time_ms": model_info.get("total_time_ms", 0),
            "load_time_ms": model_info.get("load_time_ms", 0),
            "device_used": model_info.get("device", "unknown")
        }
    
    df_results = pd.DataFrame(results_rows)
    logger.info(f"Prepared results dataframe with {len(df_results)} rows")
    
    return df_results, model_metadata


def save_results_csv(
    results_df: pd.DataFrame,
    output_path: str
) -> bool:
    """
    Save results to CSV file.
    
    Args:
        results_df: Results dataframe to save
        output_path: Path for output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Total rows: {len(results_df)}, Columns: {len(results_df.columns)}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        return False


def save_results_per_model(
    results_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Save results to separate CSV file for each model.
    
    Args:
        results_df: Combined results dataframe with all models
        output_dir: Output directory for model-specific CSV files
        
    Returns:
        Dictionary mapping model_name -> output_file_path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    for model_name in results_df['model_name'].unique():
        # Filter results for this model
        model_df = results_df[results_df['model_name'] == model_name].copy()
        
        # Create sanitized filename from model name
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        output_file = os.path.join(output_dir, f"sentiment_results_{safe_model_name}.csv")
        
        try:
            model_df.to_csv(output_file, index=False)
            saved_files[model_name] = output_file
            logger.info(f"Results for {model_name} saved to {output_file}")
            logger.info(f"  - Rows: {len(model_df)}, Columns: {len(model_df.columns)}")
        except Exception as e:
            logger.error(f"Error saving results for {model_name}: {e}")
    
    return saved_files


def get_results_summary(results_df: pd.DataFrame, model_metadata: Dict) -> Dict:
    """
    Generate summary statistics for results.
    
    Args:
        results_df: Results dataframe
        model_metadata: Dictionary with per-model metadata
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_rows": len(results_df),
        "total_comments": results_df['entity_id'].nunique(),
        "total_models": results_df['model_name'].nunique(),
        "sentiment_distribution": results_df['sentiment_name'].value_counts().to_dict(),
        "avg_confidence": results_df['confidence_score'].mean(),
        "processing_time_seconds": max(m.get("total_time_ms", 0) for m in model_metadata.values()) / 1000
    }
    
    # Per-model metrics
    model_metrics = {}
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        metadata = model_metadata.get(model_name, {})
        model_metrics[model_name] = {
            "avg_latency_ms": model_data['inference_time_ms'].mean(),
            "throughput_per_sec": metadata.get("throughput_per_sec", 0),
            "load_time_ms": metadata.get("load_time_ms", 0),
            "avg_confidence": model_data['confidence_score'].mean(),
            "device": metadata.get("device_used", "unknown")
        }
    
    summary["model_metrics"] = model_metrics
    
    return summary


def print_results_summary(summary: Dict) -> None:
    """
    Print results summary to console.
    
    Args:
        summary: Summary dictionary from get_results_summary()
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULTS SUMMARY")
    print("="*80)
    print(f"Total Comments Analyzed: {summary['total_comments']}")
    print(f"Total Models Used: {summary['total_models']}")
    print(f"Average Confidence Score: {summary['avg_confidence']:.4f}")
    print(f"Total Processing Time: {summary['processing_time_seconds']:.2f}s")
    
    print(f"\nSentiment Distribution (across all models):")
    for sentiment, count in summary['sentiment_distribution'].items():
        print(f"  {sentiment}: {count} ({count / summary['total_rows'] * 100:.1f}%)")
    
    print(f"\nPer-Model Metrics:")
    for model_name, metrics in summary['model_metrics'].items():
        print(f"\n  {model_name}")
        print(f"    Device: {metrics['device']}")
        print(f"    Load Time: {metrics['load_time_ms']:.2f}ms")
        print(f"    Avg Latency: {metrics['avg_latency_ms']:.2f}ms/comment")
        print(f"    Throughput: {metrics['throughput_per_sec']:.1f} comments/sec")
        print(f"    Avg Confidence: {metrics['avg_confidence']:.4f}")
    
    print("\n" + "="*80)
