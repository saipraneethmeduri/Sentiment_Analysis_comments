"""
Sentiment Normalization Module
Normalizes sentiment outputs from different models to consistent 3-class format
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_sentiment(
    label: str,
    model_name: str,
    confidence: float = None
) -> Tuple[int, str]:
    """
    Normalize sentiment label to 3-class format.
    
    Args:
        label: Original label from model
        model_name: Name of the model (for context-specific handling)
        confidence: Confidence score (optional, for threshold-based mapping)
        
    Returns:
        Tuple of (sentiment_class: int, sentiment_name: str)
        - 0: positive
        - 1: neutral
        - 2: negative
    """
    label = str(label).lower().strip()
    
    # Handle twitter sentiment model outputs (LABEL_0, LABEL_1, LABEL_2)
    if label.startswith("label_"):
        label_num = int(label.split("_")[1])
        if model_name == "cardiffnlp/twitter-xlm-roberta-base-sentiment":
            # This model outputs: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
            mapping = {0: 2, 1: 1, 2: 0}  # Map to our format: 0=positive, 1=neutral, 2=negative
            return mapping[label_num], get_sentiment_name(mapping[label_num])
        return label_num, get_sentiment_name(label_num)
    
    # Positive sentiment variations
    positive_keywords = [
        "positive", "good", "great", "excellent", "amazing", "awesome",
        "love", "wonderful", "fantastic", "perfect", "nice", "best"
    ]
    
    # Negative sentiment variations
    negative_keywords = [
        "negative", "bad", "terrible", "awful", "hate", "horrible",
        "poor", "worst", "disappointing", "useless", "waste", "wrong"
    ]
    
    # Neutral sentiment variations
    neutral_keywords = [
        "neutral", "okay", "ok", "fine", "average", "normal", "none"
    ]
    
    # Check for exact matches or substring matches
    if any(keyword in label for keyword in positive_keywords):
        return 0, "positive"
    elif any(keyword in label for keyword in negative_keywords):
        return 2, "negative"
    elif any(keyword in label for keyword in neutral_keywords):
        return 1, "neutral"
    
    # Default: try to infer from confidence if available
    if confidence is not None:
        if confidence > 0.7:
            return 0, "positive"
        elif confidence < 0.3:
            return 2, "negative"
        else:
            return 1, "neutral"
    
    # Fallback
    logger.warning(f"Could not normalize label: {label} from model {model_name}")
    return 1, "neutral"


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
    labels: list,
    model_name: str,
    confidences: list = None
) -> Tuple[list, list]:
    """
    Normalize a batch of sentiment labels.
    
    Args:
        labels: List of sentiment labels
        model_name: Name of the model
        confidences: Optional list of confidence scores
        
    Returns:
        Tuple of (sentiment_classes, sentiment_names)
    """
    if confidences is None:
        confidences = [None] * len(labels)
    
    classes = []
    names = []
    
    for label, conf in zip(labels, confidences):
        cls, name = normalize_sentiment(label, model_name, conf)
        classes.append(cls)
        names.append(name)
    
    return classes, names


def map_2class_to_3class(
    label: str,
    confidence: float = None
) -> Tuple[int, str]:
    """
    Map 2-class sentiment (positive/negative) to 3-class format.
    For 2-class models without neutral, we assign neutral based on confidence.
    
    Args:
        label: 'positive' or 'negative'
        confidence: Confidence score for the prediction
        
    Returns:
        Tuple of (sentiment_class, sentiment_name)
    """
    label = str(label).lower().strip()
    
    if "positive" in label:
        # If confidence is low, might indicate uncertainty (neutral)
        if confidence is not None and confidence < 0.6:
            return 1, "neutral"
        return 0, "positive"
    elif "negative" in label:
        # If confidence is low, might indicate uncertainty (neutral)
        if confidence is not None and confidence < 0.6:
            return 1, "neutral"
        return 2, "negative"
    
    return 1, "neutral"
