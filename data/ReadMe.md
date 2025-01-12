# Data Preparation and Processing Guide

This document explains how to structure your raw data and use the provided scripts to process it into graphs and figures for further analysis.

---

## Directory Structure

Ensure your dataset follows the structure outlined below:
```yaml
{dataset_root}/ ├── fig/ # Directory for processed figures 
                ├── graph/ # Directory for processed graphs 
                ├── tg_graph/ # TensorGraph processed data (optional for advanced analysis) 
                ├── model/ # Models used for downstream analysis (optional) 
                ├── metadata/ │── {region_level_metadata}.csv # Metadata for region-level information 
                ├── voronoi/├── {region_id}.cell_data.csv # Cell coordinates for each region 
                            ├── {region_id}.cell_features.csv # Additional cell-level features 
                            ├── {region_id}.cell_types.csv # Cell types for each region 
                            └── {region_id}.expression.csv # Biomarker expression data
```

## Raw Data Details

### Metadata (`metadata/`)
- Region-level CSV file containing metadata:
    - **Columns:**
        - `REGION_ID`: Unique identifier for each region.
        - `survival_status`, `survival_day`: Clinical outcomes.
        - `status`, `hpvstatus`, `recurred`, `isrecurrence`: Additional annotations.

### Cell Coordinates (`voronoi/{region_id}.cell_data.csv`)
- Coordinates for each cell in the region:
    | CELL_ID | X  | Y  |
    |---------|-----|-----|
    | 0       | 3   | 10  |
    | 1       | 25  | 12  |
    | 2       | 16  | 30  |
    | 3       | 32  | 26  |

### Cell Features (`voronoi/{region_id}.cell_features.csv`)
- Additional features for cells (e.g., size):
    | CELL_ID | SIZE |
    |---------|------|
    | 0       | 0.2  |
    | 1       | 0.5  |
    | 2       | 0.8  |
    | 3       | 0.1  |

### Cell Types (`voronoi/{region_id}.cell_types.csv`)
- Types assigned to each cell:
    | CELL_ID | CELL_TYPE     |
    |---------|---------------|
    | 0       | CD4 T cell    |
    | 1       | CD8 T cell    |
    | 2       | Tumor cell    |
    | 3       | B cell        |

### Biomarker Expression (`voronoi/{region_id}.expression.csv`)
- Biomarker expression levels for each cell:
    | CELL_ID | REGION_ID | BIOMARKER1 | BIOMARKER2 |
    |---------|-----------|------------|------------|
    | 0       | region1   | 0.5        | 1.1        |
    | 1       | region1   | 0.8        | 2.3        |
    | 2       | region1   | 1.5        | 0.9        |
    | 3       | region1   | 5.5        | 0.1        |

---

## How to Process Raw Data

### Pre-Requisites
1. Install [space-gm](https://github.com/space-gm/space-gm) following the instructions provided in the repository.
2. Ensure the raw data is organized according to the structure described above.

### Script Usage

The processing script is located in the `scripts/` directory. Use the following command to process your data:

```bash
python scripts/process_regions.py --data-root {dataset_root} --num-workers 4    

```
Parameters
* --data-root: Root directory containing the raw data.
* --num-workers: Number of parallel workers (default: 4).

Outputs
* Processed figures will be saved in {dataset_root}/fig.
* Processed graphs will be saved in {dataset_root}/graph.
* Logs for errors (if any) will be saved in {dataset_root}/error_regions.log.