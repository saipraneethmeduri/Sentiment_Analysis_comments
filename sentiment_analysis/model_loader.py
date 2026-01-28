"""
Model Loader Module
Loads and initializes HuggingFace transformer models for sentiment analysis
"""

import pandas as pd
import torch
from transformers import pipeline
from typing import Dict, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Detect available device (GPU or CPU).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU for inference")
    
    return device


def load_models_from_csv(csv_path: str) -> Dict[str, str]:
    """
    Load model names from CSV file.
    
    Args:
        csv_path: Path to sentiment_analysis_model_list.csv
        
    Returns:
        Dictionary with index as key and model_name as value
    """
    try:
        df = pd.read_csv(csv_path)
        models = df['model'].tolist()
        return {i: model for i, model in enumerate(models)}
    except Exception as e:
        logger.error(f"Error loading models from {csv_path}: {e}")
        return {}


def initialize_pipeline(
    model_name: str,
    task: str = "text-classification",
    device: str = None,
    cache_dir: Optional[str] = None
) -> tuple:
    """
    Initialize a HuggingFace pipeline for sentiment analysis.
    
    Args:
        model_name: Model identifier from HuggingFace
        task: Task type (default: text-classification)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        cache_dir: Optional directory to cache/download model weights locally.
        
    Returns:
        Tuple of (pipeline, loading_time_ms, device_used)
    """
    if device is None:
        device = get_device()
    
    # Convert device string to device ID for transformers
    device_id = 0 if device == "cuda" else -1
    
    try:
        start_time = time.time()

        # Ensure cache directory exists if provided
        cache_path: Optional[str] = None
        if cache_dir is not None:
            cache_path = str(Path(cache_dir))
            Path(cache_path).mkdir(parents=True, exist_ok=True)

        pipe = pipeline(
            task=task,
            model=model_name,
            device=device_id,
            top_k=None,  # Get all class scores
            cache_dir=cache_path
        )
        
        load_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Loaded model {model_name} in {load_time_ms:.2f}ms on {device}")
        return pipe, load_time_ms, device
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None, 0, device


def load_all_models(
    csv_path: str,
    device: str = None,
    cache_dir: Optional[str] = None
) -> Dict[str, dict]:
    """
    Load all models from CSV and initialize pipelines.
    
    Args:
        csv_path: Path to sentiment_analysis_model_list.csv
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        cache_dir: Optional directory to cache/download model weights locally.
        
    Returns:
        Dictionary with model_name as key and {pipeline, load_time, device} as value
    """
    if device is None:
        device = get_device()
    
    models = load_models_from_csv(csv_path)
    loaded_models = {}
    total_load_time = 0
    
    logger.info(f"Loading {len(models)} models on {device}...")
    
    for idx, model_name in models.items():
        logger.info(f"[{idx + 1}/{len(models)}] Loading {model_name}...")

        pipe, load_time, used_device = initialize_pipeline(
            model_name,
            device=device,
            cache_dir=cache_dir
        )
        
        if pipe is not None:
            loaded_models[model_name] = {
                "pipeline": pipe,
                "load_time_ms": load_time,
                "device": used_device
            }
            total_load_time += load_time
        else:
            logger.warning(f"Skipping failed model: {model_name}")
    
    logger.info(f"Successfully loaded {len(loaded_models)}/{len(models)} models")
    logger.info(f"Total model loading time: {total_load_time:.2f}ms")
    
    return loaded_models


def get_model_pipeline(loaded_models: Dict, model_name: str):
    """
    Get pipeline for a specific model.
    
    Args:
        loaded_models: Dictionary from load_all_models()
        model_name: Name of the model
        
    Returns:
        Pipeline object or None if not found
    """
    if model_name in loaded_models:
        return loaded_models[model_name]["pipeline"]
    else:
        logger.warning(f"Model {model_name} not found in loaded models")
        return None
