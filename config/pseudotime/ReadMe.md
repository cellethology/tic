# Pseudo-Time Analysis Pipeline Configuration Guide

This document provides guidance on how to customize and modify the configuration files for the pseudo-time analysis pipeline using Hydra.

## Configuration Overview

The pipeline configuration is managed by Hydra, a powerful configuration framework. The configuration files are stored in YAML format and are located in the `config/pseudotime/` directory.

### Main Configuration Components

1. **General Settings (`general`)**
   - Defines the root directories, device settings, and global parameters for the pipeline.

2. **Dataset Configuration (`dataset`)**
   - Contains dataset-specific parameters like cell type mappings, biomarkers, and processing settings.

3. **Sampler Configuration (`sampler`)**
   - Specifies the settings for subgraph sampling.

4. **Pipeline Modules (`pipeline`)**
   - Contains configurations for embedding preparation, pseudo-time analysis, and neighborhood analysis.

5. **Constants (`constants`)**
   - Stores static mappings and lists like cell type mappings, biomarker lists, and cell type frequencies.

---

## How to Modify Configurations

### 1. General Settings

The `general` section defines paths and global parameters for the pipeline.

Example:
```yaml
general:
  data_root: "/path/to/your/data"  # Root directory of your dataset
  output_dir: "/path/to/your/output"  # Directory to save pipeline outputs
  model_path: "/path/to/model/weights.pt"  # Path to the pretrained model weights
  device: "mps"  # Set to "cuda", "mps", or "cpu" depending on your environment
  random_seed: 42  # Random seed for reproducibility
  n_clusters: 2  # Number of clusters for pseudo-time analysis

```
