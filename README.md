# Sentiment Analysis Comments - Project Documentation

## 📋 Project Overview

This project performs multi-model sentiment analysis on user comments using multiple HuggingFace transformer models. It processes comments, runs inference across several pre-trained sentiment analysis models, normalizes predictions to a consistent 3-class format (positive, neutral, negative), and generates comprehensive comparison reports.

### Key Features
- **Multi-model sentiment analysis** - Compare predictions from 4 different transformer models
- **Batch processing** - Efficient batch inference with configurable batch sizes
- **GPU acceleration** - Supports CUDA for faster processing
- **Normalized outputs** - All model outputs normalized to consistent 3-class sentiment format
- **Comprehensive metrics** - Latency, throughput, confidence scores, and model agreement metrics
- **Detailed reporting** - Individual model results and side-by-side comparison CSV files

---

## 🏗️ Project Structure and File Descriptions

### 📂 Root Directory Files

#### **1. pyproject.toml**
- **Purpose**: Project configuration and dependency management
- **Technology**: Python project metadata (compatible with `uv`, `pip`, etc.)
- **Dependencies**:
  - `transformers>=5.0.0` - HuggingFace transformers library
  - `torch>=2.0.0` - PyTorch for model inference
  - `pandas>=3.0.0` - Data manipulation
  - `langdetect>=1.0.9` - Language detection
  - `accelerate>=0.20.0` - Model acceleration
  - `deep-translator>=1.11.4` - Translation utilities
  - `tqdm>=4.65.0` - Progress bars
  - `sentencepiece>=0.2.1` - Tokenization
  - `protobuf>=6.33.4` - Serialization

#### **2. sentiment_analysis_model_list.csv**
- **Purpose**: List of HuggingFace models to use for sentiment analysis
- **Content**: Single column `model` with 4 model identifiers:
  - `ai4bharat/indic-bert` - Indic language BERT model
  - `cardiffnlp/twitter-xlm-roberta-base-sentiment` - Multilingual Twitter sentiment
  - `FacebookAI/xlm-roberta-base` - Cross-lingual RoBERTa base
  - `google/muril-base-cased` - Multilingual Representations for Indian Languages

#### **3. main.py**
- **Purpose**: Simple hello-world entry point (not the main execution file)
- **Note**: The actual execution happens via `sentiment_analysis/main.py`

#### **4. extract_top_entities.py**
- **Purpose**: Data preprocessing script
- **Logic**:
  1. Loads `entity_comment_counts.csv` (full dataset)
  2. Takes top 20 entities by comment count
  3. Saves filtered counts to `entity_comment_counts_20.csv`
  4. Filters corresponding comments from `entity_comments_details.csv`
  5. Saves filtered details to `entity_comments_details_20.csv`
- **Use case**: Create smaller test dataset for faster iteration

#### **5. map_comments.py**
- **Purpose**: Processes raw JSON data to create entity-comment mappings
- **Logic**:
  1. Loads `prod_sunbird__comment_.csv.json` (comment dictionary)
  2. Parses comment data and builds a comment_id → comment_text map
  3. Loads `prod_sunbird_comment_tree.csv.json` (entity-comment relationships)
  4. Extracts entity_id and associated comment_ids from tree structure
  5. Joins entity_ids with actual comment text
  6. Generates two outputs:
     - `entity_comments_details.csv` - All entity-comment pairs
     - `entity_comment_counts.csv` - Count of comments per entity
- **Output Format**:
  - Details CSV: `entity_id, comment`
  - Counts CSV: `entity_id, total_comments`

---

### 📂 sentiment_analysis/ Module

This is the core module containing all sentiment analysis logic.

#### **1. sentiment_analysis/main.py**
**Purpose**: Main orchestrator and CLI entry point

**Logic Flow**:
1. **Argument Parsing** - Parse CLI arguments for file paths, batch size, device, etc.
2. **Path Setup** - Validate input files and create output directories
3. **Data Loading** - Load comments from CSV
4. **Model Loading** - Initialize all HuggingFace model pipelines
5. **Batch Inference** - Run sentiment prediction on all comments
6. **Normalization** - Normalize model outputs to consistent format
7. **Results Storage** - Save individual model results and comparison CSV
8. **Metrics Calculation** - Calculate agreement and consensus metrics
9. **Summary Display** - Print comprehensive summary to console

**CLI Arguments**:
- `--comments` - Path to comments CSV (default: `comments_data/entity_comments_details_20.csv`)
- `--models` - Path to models CSV (default: `sentiment_analysis_model_list.csv`)
- `--output` - Output directory (default: `sentiment_analysis/outputs`)
- `--batch-size` - Batch size for inference (default: 16)
- `--device` - Device to use: `cuda` or `cpu` (default: auto-detect)
- `--models-cache` - Model cache directory (default: `./models`)
- `--log-level` - Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

**Key Functions**:
- `setup_paths()` - Validate and setup file paths
- `run_sentiment_analysis()` - Execute full pipeline
- `main()` - CLI entry point

#### **2. sentiment_analysis/model_loader.py**
**Purpose**: Load and initialize HuggingFace transformer models

**Key Functions**:
- `get_device()` - Auto-detect CUDA availability
- `load_models_from_csv()` - Parse model list from CSV
- `initialize_pipeline()` - Create HuggingFace pipeline for a model
  - Downloads model if not cached
  - Configures device (GPU/CPU)
  - Returns pipeline with metadata (load time, device)
- `load_all_models()` - Load all models from CSV
  - Iterates through model list
  - Initializes each pipeline
  - Returns dictionary: `model_name → {pipeline, load_time_ms, device}`

**Logic**:
- Downloads models to `models/` directory (HuggingFace cache)
- Supports both CUDA and CPU inference
- Tracks loading time for each model
- Uses `transformers.pipeline()` with `text-classification` task

#### **3. sentiment_analysis/batch_inference.py**
**Purpose**: Perform batch inference with timing metrics

**Key Functions**:
- `read_comments_data()` - Load comments CSV into pandas DataFrame
- `batch_inference()` - Run inference on a list of texts
  - Splits texts into batches
  - Processes each batch through pipeline
  - Tracks per-batch and per-text latency
  - Returns: labels, scores, latencies
- `process_all_models()` - Run inference for all models
  - Iterates through loaded models
  - Runs batch_inference for each
  - Calculates throughput (comments/sec)
  - Returns comprehensive results dictionary

**Metrics Tracked**:
- Total inference time (ms)
- Average latency per text (ms)
- Throughput (texts/second)
- Per-text latencies
- Confidence scores

**Logic**:
- Uses `tqdm` for progress bars
- Handles different model output formats (single result vs. list)
- Error handling for failed batches

#### **4. sentiment_analysis/sentiment_normalizer.py**
**Purpose**: Normalize different model outputs to consistent 3-class format

**Sentiment Classes**:
- **0 = Positive**
- **1 = Neutral**
- **2 = Negative**

**Key Functions**:
- `normalize_sentiment()` - Normalize single label
  - Handles LABEL_0/LABEL_1/LABEL_2 format
  - Special handling for twitter-xlm-roberta (LABEL_0=negative, LABEL_2=positive)
  - Keyword-based matching for sentiment words
  - Confidence-based fallback for ambiguous labels
- `get_sentiment_name()` - Convert class number to name
- `normalize_batch_sentiments()` - Normalize batch of labels
- `map_2class_to_3class()` - Convert 2-class to 3-class (uses confidence threshold)

**Logic**:
Different models output different label formats:
- Some use "positive", "negative", "neutral"
- Twitter model uses LABEL_0, LABEL_1, LABEL_2
- Some only output 2 classes (positive/negative)

This module standardizes all to: 0=positive, 1=neutral, 2=negative

#### **5. sentiment_analysis/results_storage.py**
**Purpose**: Store and organize results into CSV files

**Key Functions**:
- `prepare_results_dataframe()` - Create comprehensive results DataFrame
  - Combines comments, predictions, confidences, latencies
  - One row per (comment, model) combination
- `save_results_csv()` - Save DataFrame to CSV
- `save_results_per_model()` - Save separate CSV for each model
  - Creates files like `sentiment_results_ai4bharat_indic_bert.csv`
- `get_results_summary()` - Calculate summary statistics
- `print_results_summary()` - Display formatted summary

**Output Files**:
Each model gets its own CSV with columns:
- `entity_id` - Entity identifier
- `comment` - Original comment text
- `model_name` - Name of the model
- `sentiment_class` - Normalized class (0/1/2)
- `sentiment_name` - Sentiment name (positive/neutral/negative)
- `confidence_score` - Model confidence
- `inference_time_ms` - Time to process this comment

**Summary Metrics**:
- Total comments analyzed
- Sentiment distribution (positive/neutral/negative counts)
- Average confidence score
- Per-model latency and throughput
- Processing time

#### **6. sentiment_analysis/aggregator.py**
**Purpose**: Compare predictions across models and calculate agreement

**Key Functions**:
- `calculate_model_agreement()` - Calculate per-comment agreement
  - Agreement score = (most common prediction count) / (total models)
  - Returns average agreement across all comments
- `generate_comparison_csv()` - Create side-by-side comparison
  - One row per comment
  - Columns for each model's sentiment, confidence, latency
  - Includes model metadata (load time, throughput, device)
  - Calculates model_agreement_score
- `get_model_consensus()` - Identify consensus predictions
  - Most common sentiment across models
  - Consensus ratio (how many models agreed)
  - Counts unanimous and majority agreements
- `print_aggregation_summary()` - Display aggregation metrics

**Output: sentiment_comparison.csv**
Columns:
- `entity_id`, `comment`
- For each model:
  - `{model}_sentiment` - Predicted sentiment
  - `{model}_confidence` - Confidence score
  - `{model}_latency_ms` - Inference latency
  - `{model}_total_comments_processed` - Total processed
  - `{model}_avg_latency_ms` - Average latency
  - `{model}_throughput_per_sec` - Throughput
  - `{model}_total_time_ms` - Total processing time
  - `{model}_load_time_ms` - Model load time
  - `{model}_device_used` - Device (cuda/cpu)
- `model_agreement_score` - Agreement across models

---

## 🔄 Execution Flow

```
1. User runs: python -m sentiment_analysis.main [--options]
                    ↓
2. Parse CLI arguments and setup paths
                    ↓
3. Load comments from CSV (comments_data/entity_comments_details_20.csv)
                    ↓
4. Load all models from sentiment_analysis_model_list.csv
   - Download models if not cached
   - Initialize pipelines on GPU/CPU
                    ↓
5. Run batch inference for each model
   - Process comments in batches
   - Track latency and confidence
                    ↓
6. Normalize sentiment predictions to 3-class format
                    ↓
7. Save results:
   - Individual model CSVs (outputs/sentiment_results_{model}.csv)
   - Comparison CSV (outputs/sentiment_comparison.csv)
                    ↓
8. Calculate and display metrics:
   - Model agreement scores
   - Consensus statistics
   - Performance metrics (latency, throughput)
                    ↓
9. Display summary and exit
```

---

## 📊 Output Files

### Individual Model Results
**Location**: `sentiment_analysis/outputs/sentiment_results_{model_name}.csv`

**Columns**:
- `entity_id` - Entity identifier
- `comment` - Comment text
- `model_name` - Model used
- `sentiment_class` - 0/1/2 (positive/neutral/negative)
- `sentiment_name` - positive/neutral/negative
- `confidence_score` - Model confidence (0-1)
- `inference_time_ms` - Latency per comment

### Comparison CSV
**Location**: `sentiment_analysis/outputs/sentiment_comparison.csv`

**Contains**: Side-by-side predictions from all models with:
- All sentiment predictions and confidences
- Per-model performance metrics
- Model agreement scores
- Full metadata for each model

---

## 🚀 Implementation Steps for Another System

### Prerequisites
- **Operating System**: Linux
- **Python Version**: Python 3.12 or higher
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing

### Step 1: Install Python 3.12+
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip

# macOS (using Homebrew)
brew install python@3.12

# Verify installation
python3.12 --version
```

### Step 2: Clone or Create Project Directory
```bash
# Create project directory
mkdir sentiment_analysis_comments
cd sentiment_analysis_comments

# Copy all project files to this directory
# Or clone from repository if available
```

### Step 3: Install uv Package Manager (Recommended)
```bash
# Install uv (faster alternative to pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### Step 4: Create Virtual Environment
```bash
# Using uv (recommended)
uv venv

# Or using standard venv
python3.12 -m venv .venv
```

### Step 5: Activate Virtual Environment
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 6: Install Dependencies
```bash
# Using uv (automatically reads pyproject.toml)
uv sync

# Or using pip
pip install -e .

# Or install individually
pip install transformers>=5.0.0 torch>=2.0.0 pandas>=3.0.0 \
    langdetect>=1.0.9 accelerate>=0.20.0 deep-translator>=1.11.4 \
    tqdm>=4.65.0 sentencepiece>=0.2.1 protobuf>=6.33.4 ipykernel>=7.1.0
```

### Step 7: Prepare Data Files

**Required Directory Structure**:
```
sentiment_analysis_comments/
├── comments_data/
│   └── entity_comments_details_20.csv  # Your comments data
├── sentiment_analysis_model_list.csv    # Model list
├── sentiment_analysis/                  # Python module
└── models/                              # Will be created automatically
```

**Create comments CSV** with columns:
- `entity_id` - Unique identifier for entity
- `comment` - Comment text to analyze

### Step 8: Run the Analysis

**Basic execution** (CPU, default settings):
```bash
python -m sentiment_analysis.main
```

**With GPU acceleration**:
```bash
python -m sentiment_analysis.main --device cuda
```

**Custom batch size** (larger for more memory):
```bash
python -m sentiment_analysis.main --batch-size 32 --device cuda
```

**Custom data files**:
```bash
python -m sentiment_analysis.main \
    --comments /path/to/comments.csv \
    --models /path/to/models.csv \
    --output /path/to/output
```

**Debug mode**:
```bash
python -m sentiment_analysis.main --log-level DEBUG
```

### Step 9: View Results

Results will be saved to `sentiment_analysis/outputs/`:
- Individual model files: `sentiment_results_{model_name}.csv`
- Comparison file: `sentiment_comparison.csv`

```
```
---

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 10 GB free space (for models cache)
- **Python**: 3.12+

### Recommended Requirements (for GPU)
- **GPU**: NVIDIA GPU with ~4 GB VRAM
- **CUDA**: 11.8 or 12.1
- **RAM**: 16-32 GB
- **Disk**: SSD with 20+ GB free space

### Performance Estimates
**CPU (16 GB RAM)**:
- ~100 comments: 2-3 minutes
- ~1000 comments: 20-30 minutes

**GPU**:
- ~100 comments: 20-40 seconds
- ~1000 comments: 2-4 minutes

---

## 📝 Notes

### Model Cache
Models are downloaded once to `models/` directory (~5-10 GB total). Subsequent runs reuse cached models.

### Memory Usage
Each model requires ~2-4 GB RAM/VRAM when loaded. With 4 models:
- CPU: ~12-16 GB RAM recommended
- GPU: ~4 GB VRAM recommended

### Troubleshooting

**CUDA Out of Memory**:
```bash
# Reduce batch size
python -m sentiment_analysis.main --batch-size 8 --device cuda
```

**Slow CPU Processing**:
```bash
# Use fewer models (edit sentiment_analysis_model_list.csv)
# Or increase batch size for better CPU utilization
python -m sentiment_analysis.main --batch-size 32
```

**Import Errors**:
```bash
# Reinstall dependencies
pip install --upgrade -e .
```

---

## 📦 Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | ≥5.0.0 | HuggingFace model inference |
| torch | ≥2.0.0 | PyTorch deep learning framework |
| pandas | ≥3.0.0 | Data manipulation and CSV handling |
| langdetect | ≥1.0.9 | Language detection |
| accelerate | ≥0.20.0 | Model optimization and GPU support |
| deep-translator | ≥1.11.4 | Translation utilities |
| tqdm | ≥4.65.0 | Progress bars |
| sentencepiece | ≥0.2.1 | Tokenization for some models |
| protobuf | ≥6.33.4 | Protocol buffers for model serialization |
| ipykernel | ≥7.1.0 | Jupyter notebook support |

---

## 🎯 Use Cases

1. **Comparative Model Analysis** - Evaluate which sentiment model works best for your data
2. **Consensus-based Prediction** - Use majority voting across models for more reliable results
3. **Multilingual Sentiment** - Models support English and Indic languages
4. **Performance Benchmarking** - Compare latency and throughput across models
5. **Research & Development** - Experiment with different model combinations

---

## 📧 Support

For issues or questions:
1. Check logs with `--log-level DEBUG`
2. Verify data format matches expected CSV structure
3. Ensure all dependencies are correctly installed
4. Check GPU availability if using CUDA

---

**Last Updated**: January 2026  
**Python Version**: 3.12+  
**License**: Not specified in project files
# Sentiment_Analysis_comments
# Sentiment_Analysis_comments
