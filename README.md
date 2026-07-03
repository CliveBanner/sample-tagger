# Sample Tagger

A tool for automatically discovering, labeling, and clustering large collections of audio samples. The tagger extracts acoustic features, runs clustering, and provides a rich web interface for manual review, tagging, UMAP visualization, and natural language semantic search.

## Features
- **Discovery**: Fast concurrent scanning of large sample libraries (stat checks, metadata extraction).
- **Labeling Pipeline**: Extracts neural embeddings using PANNs (CNN14) and applies fast folder heuristics.
- **Machine Learning**: Trains a custom logistic regression classifier head on human judgments and applies it across the whole database.
- **Semantic Search**: Uses CLAP text-to-audio models for natural language queries like "warm analog bass" or "vinyl breakbeat."
- **Web UI**: Includes an interactive UMAP projection map, built-in media player, search results with inline rating, and task management.

## Installation

You can install the package in editable mode:

```bash
pip install -e .
```

## Usage

### Command Line Pipeline
Run the main pipeline using the command line:

```bash
# Discover and register new samples
sample-tagger discover <path_to_samples>

# Extract features and tag (runs PANNs embedding extraction and path heuristics)
sample-tagger label

# Find similar samples by path
sample-tagger sim "path/to/sample.wav"
```

### Machine Learning & CLAP CLI
There is a dedicated CLI for ML tasks:
```bash
# Full ML pipeline: exports embeddings, trains LogisticRegression, writes predictions
sample-tagger-ml pipeline samples.db

# Extract CLAP embeddings for semantic search
sample-tagger-ml clap-embed samples.db --full
```

### Web Dashboard
To view the map, search semantically, manually review classifications, and control the pipeline:

```bash
# Start the web UI server on port 8765
sample-tagger-web
```
