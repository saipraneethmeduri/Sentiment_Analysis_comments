"""
Sentiment Normalization Module
Normalizes sentiment outputs from different models to consistent 3-class format
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_sentiment(
    scores_dict: Dict[str, float],
    model_name: str
) -> Tuple[int, str]:
    """
    Normalize sentiment using argmax on score dictionary.
    
    Args:
        scores_dict: Dictionary mapping label names to scores (e.g., {'LABEL_0': 0.1, 'LABEL_1': 0.3, 'LABEL_2': 0.6})
        model_name: Name of the model (for context-specific label mapping)
        
    Returns:
        Tuple of (sentiment_class: int, sentiment_name: str)
        - 0: positive
        - 1: neutral
        - 2: negative
    """
    if not scores_dict:
        logger.warning(f"Empty scores_dict received for model {model_name}")
        return 1, "neutral"
    
    # First, normalize all labels to standard format (positive/neutral/negative)
    normalized_scores = {}
    
    for label, score in scores_dict.items():
        label_lower = str(label).lower().strip()
        
        # Handle LABEL_N format (common in HuggingFace models)
        if label_lower.startswith("label_"):
            label_num = int(label_lower.split("_")[1])
            if model_name == "cardiffnlp/twitter-xlm-roberta-base-sentiment":
                # This model: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
            else:
                # Default: LABEL_0=positive, LABEL_1=neutral, LABEL_2=negative
                label_map = {0: "positive", 1: "neutral", 2: "negative"}
            
            if label_num in label_map:
                normalized_label = label_map[label_num]
                normalized_scores[normalized_label] = score
            continue
        
        # Map text labels to standard format
        if any(kw in label_lower for kw in ["positive", "good", "great", "excellent", "pos"]):
            normalized_scores["positive"] = normalized_scores.get("positive", 0) + score
        elif any(kw in label_lower for kw in ["negative", "bad", "terrible", "neg"]):
            normalized_scores["negative"] = normalized_scores.get("negative", 0) + score
        elif any(kw in label_lower for kw in ["neutral", "neu", "mixed"]):
            normalized_scores["neutral"] = normalized_scores.get("neutral", 0) + score
        else:
            # Unknown label - treat as neutral
            normalized_scores["neutral"] = normalized_scores.get("neutral", 0) + score
    
    # If we don't have any normalized scores, default to neutral
    if not normalized_scores:
        logger.warning(f"Could not normalize any labels from {scores_dict} for model {model_name}")
        return 1, "neutral"
    
    # For 2-class models (only positive and negative), exclude neutral
    if "neutral" not in normalized_scores and len(normalized_scores) == 2:
        logger.debug(f"2-class model detected for {model_name}")
    
    # Apply argmax - select class with highest score
    max_sentiment = max(normalized_scores.items(), key=lambda x: x[1])
    sentiment_name = max_sentiment[0]
    
    # Convert sentiment name to class number
    sentiment_class_map = {
        "positive": 0,
        "neutral": 1,
        "negative": 2
    }
    
    sentiment_class = sentiment_class_map.get(sentiment_name, 1)
    
    return sentiment_class, sentiment_name


def get_sentiment_name(sentiment_class: int) -> str:
    """
    Get sentiment name from class number.
    
    Args:
        sentiment_class: 0=positive, 1=neutral, 2=negative
        
    Returns:
        Sentiment name string
    """
    mapping = {
        0: "positive",
        1: "neutral",
        2: "negative"
    }
    return mapping.get(sentiment_class, "neutral")


def normalize_batch_sentiments(
    scores_dicts: list,
    model_name: str
) -> Tuple[list, list]:
    """
    Normalize a batch of sentiment predictions using argmax.
    
    Args:
        scores_dicts: List of score dictionaries from model outputs
        model_name: Name of the model
        
    Returns:
        Tuple of (sentiment_classes, sentiment_names)
    """
    classes = []
    names = []
    
    for scores_dict in scores_dicts:
        cls, name = normalize_sentiment(scores_dict, model_name)
        classes.append(cls)
        names.append(name)
    
    return classes, names
