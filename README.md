# Sample Tagger

A tool for automatically discovering, labeling, and clustering large collections of audio samples. The tagger extracts acoustic features, runs clustering, and provides a web interface for manual review and tagging.

## Installation

You can install the package in editable mode:

```bash
pip install -e .
```

## Usage

Run the main pipeline using the command line:

```bash
# Discover and extract features
sample-tagger <path_to_samples> --stage discover

# Label extracted samples
sample-tagger <path_to_samples> --stage label

# Optional: Run PANNS relabeling separately
sample-tagger <path_to_samples> --stage relabel-panns
```

To view and manually review the classifications, start the web UI:

```bash
sample-tagger-web
```
