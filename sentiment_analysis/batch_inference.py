"""
Batch Inference Module
Performs batch sentiment analysis with timing metrics
"""

import pandas as pd
import time
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_comments_data(csv_path: str) -> pd.DataFrame:
    """
    Read comments data from CSV with ground truth labels.
    
    Args:
        csv_path: Path to entity_comments_details_20_labeled.csv
        
    Returns:
        DataFrame with entity_id, comment, sentiment_class, and sentiment_name columns
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} comments from {csv_path}")
        logger.info(f"Columns: {list(df.columns)}")
        # Ensure required columns exist
        required_cols = ['entity_id', 'comment', 'sentiment_class', 'sentiment_name']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Expected: {required_cols}, Got: {list(df.columns)}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error reading comments from {csv_path}: {e}")
        return pd.DataFrame()


def batch_inference(
    texts: List[str],
    pipeline,
    model_name: str,
    batch_size: int = 16
) -> Tuple[Dict, float, float]:
    """
    Perform batch inference on texts using a pipeline.
    
    Args:
        texts: List of text inputs
        pipeline: HuggingFace pipeline
        model_name: Name of the model (for logging)
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (results_dict, total_time_ms, avg_latency_ms)
        results_dict has keys: 'labels', 'scores', 'latencies'
    """
    if not texts or pipeline is None:
        return {"labels": [], "scores": [], "latencies": []}, 0, 0
    
    all_labels = []
    all_scores = []
    all_latencies = []
    
    total_start = time.time()
    
    logger.debug(f"Running batch inference for {len(texts)} texts with {model_name}")
    
    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc=f"Inferencing with {model_name}", unit="batch"):
        batch = texts[i:i + batch_size]
        batch_start = time.time()
        
        try:
            # Run inference
            outputs = pipeline(batch)
            batch_time_ms = (time.time() - batch_start) * 1000
            
            # Extract results
            for output, text in zip(outputs, batch):
                # Handle different output formats
                if isinstance(output, list):
                    # Multiple results per input
                    top_result = output[0]
                    label = top_result.get('label', 'NEUTRAL')
                    score = top_result.get('score', 0.0)
                else:
                    # Single result per input
                    label = output.get('label', 'NEUTRAL')
                    score = output.get('score', 0.0)
                
                all_labels.append(label)
                all_scores.append(score)
                all_latencies.append(batch_time_ms / len(batch))
            
            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
        except Exception as e:
            logger.error(f"Error in batch inference at index {i}: {e}")
            # Add placeholder results for failed batch
            for _ in batch:
                all_labels.append("UNKNOWN")
                all_scores.append(0.0)
                all_latencies.append(0.0)
    
    total_time_ms = (time.time() - total_start) * 1000
    avg_latency_ms = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    
    logger.info(
        f"{model_name}: Processed {len(texts)} texts in {total_time_ms:.2f}ms "
        f"(avg {avg_latency_ms:.2f}ms per text)"
    )
    
    return {
        "labels": all_labels,
        "scores": all_scores,
        "latencies": all_latencies
    }, total_time_ms, avg_latency_ms


def process_all_models(
    comments_df: pd.DataFrame,
    loaded_models: Dict,
    batch_size: int = 16
) -> Dict[str, Dict]:
    """
    Run inference on all comments using all loaded models.
    
    Args:
        comments_df: DataFrame with comments
        loaded_models: Dictionary from model_loader.load_all_models()
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with model_name as key and inference results as value
    """
    if comments_df.empty:
        logger.error("Comments dataframe is empty")
        return {}
    
    texts = comments_df['comment'].astype(str).tolist()
    all_results = {}
    
    logger.info(f"Starting inference on {len(texts)} comments with {len(loaded_models)} models")
    
    for model_name, model_info in tqdm(loaded_models.items(), desc="Processing models", unit="model"):
        logger.info(f"Running inference with {model_name}...")
        
        results, total_time, avg_latency = batch_inference(
            texts,
            model_info["pipeline"],
            model_name,
            batch_size=batch_size
        )
        
        # Calculate throughput
        throughput = len(texts) / (total_time / 1000) if total_time > 0 else 0
        
        all_results[model_name] = {
            "inference_results": results,
            "total_time_ms": total_time,
            "avg_latency_ms": avg_latency,
            "throughput_per_sec": throughput,
            "load_time_ms": model_info["load_time_ms"],
            "device": model_info["device"]
        }
    
    logger.info("Inference complete for all models")
    
    return all_results


def get_inference_metrics(model_results: Dict) -> Dict:
    """
    Extract and summarize inference metrics for a model.
    
    Args:
        model_results: Results dictionary from process_all_models()
        
    Returns:
        Dictionary with metrics
    """
    return {
        "total_time_ms": model_results.get("total_time_ms", 0),
        "avg_latency_ms": model_results.get("avg_latency_ms", 0),
        "throughput_per_sec": model_results.get("throughput_per_sec", 0),
        "load_time_ms": model_results.get("load_time_ms", 0),
        "device": model_results.get("device", "unknown")
    }
