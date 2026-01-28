"""
Main Orchestrator Module
Coordinates all sentiment analysis modules and runs the full pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

from sentiment_analysis.model_loader import load_all_models
from sentiment_analysis.batch_inference import (
    read_comments_data,
    process_all_models,
    get_inference_metrics
)
from sentiment_analysis.sentiment_normalizer import normalize_batch_sentiments
from sentiment_analysis.results_storage import (
    prepare_results_dataframe,
    save_results_csv,
    save_results_per_model,
    get_results_summary,
    print_results_summary
)
from sentiment_analysis.aggregator import (
    calculate_model_agreement,
    generate_comparison_csv,
    get_model_consensus,
    print_aggregation_summary
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths(
    comments_csv: Optional[str] = None,
    models_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    models_cache_dir: Optional[str] = None
) -> dict:
    """
    Setup and validate file paths.
    
    Args:
        comments_csv: Path to comments CSV
        models_csv: Path to models CSV
        output_dir: Output directory
        models_cache_dir: Directory to cache/download HuggingFace models
        
    Returns:
        Dictionary with validated paths
    """
    # Default paths relative to workspace root
    workspace_root = Path(__file__).parent.parent
    
    if comments_csv is None:
        comments_csv = workspace_root / "comments_data" / "entity_comments_details_20.csv"
    
    if models_csv is None:
        models_csv = workspace_root / "sentiment_analysis_model_list.csv"
    
    if output_dir is None:
        output_dir = workspace_root / "sentiment_analysis" / "outputs"

    if models_cache_dir is None:
        models_cache_dir = workspace_root / "models"
    
    # Ensure paths are Path objects
    comments_csv = Path(comments_csv)
    models_csv = Path(models_csv)
    output_dir = Path(output_dir)
    models_cache_dir = Path(models_cache_dir)
    
    # Validate input files exist
    if not comments_csv.exists():
        logger.error(f"Comments CSV not found: {comments_csv}")
        return None
    
    if not models_csv.exists():
        logger.error(f"Models CSV not found: {models_csv}")
        return None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    models_cache_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "comments_csv": str(comments_csv),
        "models_csv": str(models_csv),
        "output_dir": str(output_dir),
        "models_cache_dir": str(models_cache_dir)
    }


def run_sentiment_analysis(
    comments_csv: str,
    models_csv: str,
    output_dir: str,
    batch_size: int = 16,
    device: Optional[str] = None,
    models_cache_dir: Optional[str] = None
) -> bool:
    """
    Run complete sentiment analysis pipeline.
    
    Args:
        comments_csv: Path to comments CSV
        models_csv: Path to models CSV
        output_dir: Output directory
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu')
        models_cache_dir: Directory to cache/download HuggingFace models
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("="*80)
        logger.info("SENTIMENT ANALYSIS PIPELINE STARTED")
        logger.info("="*80)
        
        # Step 1: Load comments data
        logger.info("\n[Step 1/5] Loading comments data...")
        comments_df = read_comments_data(comments_csv)
        if comments_df.empty:
            logger.error("Failed to load comments data")
            return False
        
        # Step 2: Load models
        logger.info("\n[Step 2/5] Loading models...")
        logger.info(f"Model cache directory: {models_cache_dir}")

        loaded_models = load_all_models(
            models_csv,
            device=device,
            cache_dir=models_cache_dir
        )
        if not loaded_models:
            logger.error("Failed to load any models")
            return False
        
        # Step 3: Run inference
        logger.info("\n[Step 3/5] Running batch inference...")
        model_results = process_all_models(comments_df, loaded_models, batch_size=batch_size)
        
        # Step 5: Normalize sentiments
        logger.info("\n[Step 4/5] Normalizing sentiment predictions...")
        normalized_sentiments = {}
        for model_name in model_results.keys():
            inference_results = model_results[model_name]["inference_results"]
            labels = inference_results["labels"]
            scores = inference_results["scores"]
            
            classes, names = normalize_batch_sentiments(labels, model_name, scores)
            normalized_sentiments[model_name] = {
                "classes": classes,
                "names": names
            }
        
        # Step 5: Prepare and save results
        logger.info("\n[Step 5/5] Preparing and saving results...")
        results_df, model_metadata = prepare_results_dataframe(
            comments_df,
            model_results,
            normalized_sentiments
        )
        
        # Save results to separate CSV file for each model
        logger.info("\nSaving individual model results...")
        saved_files = save_results_per_model(results_df, output_dir)
        logger.info(f"Successfully saved {len(saved_files)} model result files")
        
        # Generate and save comparison CSV
        comparison_path = Path(output_dir) / "sentiment_comparison.csv"
        generate_comparison_csv(results_df, model_metadata, str(comparison_path))
        
        # Calculate and display summaries
        results_summary = get_results_summary(results_df, model_metadata)
        print_results_summary(results_summary)
        
        agreement_metrics = calculate_model_agreement(results_df)
        consensus_metrics = get_model_consensus(results_df)
        print_aggregation_summary(agreement_metrics, consensus_metrics)
        
        logger.info("\n" + "="*80)
        logger.info("SENTIMENT ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Per-model result files generated:")
        for model_name, file_path in saved_files.items():
            logger.info(f"  - {model_name}: {file_path}")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return False


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis using HuggingFace Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sentiment_analysis.main
  python -m sentiment_analysis.main --comments comments.csv --models models.csv
  python -m sentiment_analysis.main --device cuda --batch-size 32
        """
    )
    
    parser.add_argument(
        "--comments",
        type=str,
        default=None,
        help="Path to comments CSV file (default: comments_data/entity_comments_details_20.csv)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Path to models CSV file (default: sentiment_analysis_model_list.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: sentiment_analysis/outputs)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for inference (default: auto-detect)"
    )

    parser.add_argument(
        "--models-cache",
        type=str,
        default=None,
        help="Directory to cache/download HuggingFace models (default: ./models)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Setup paths
    paths = setup_paths(args.comments, args.models, args.output, args.models_cache)
    if paths is None:
        logger.error("Failed to setup paths")
        return 1
    
    # Run pipeline
    success = run_sentiment_analysis(
        comments_csv=paths["comments_csv"],
        models_csv=paths["models_csv"],
        output_dir=paths["output_dir"],
        batch_size=args.batch_size,
        device=args.device,
        models_cache_dir=paths["models_cache_dir"]
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
