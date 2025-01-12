import os
import spacegm
from multiprocessing import Pool
from tqdm import tqdm
import argparse

"""
# Space-GM Dataset Processing Script

This script processes region-level data for a single dataset, constructing graphs and generating visualization figures.

## Features:
1. Handles region data with cell coordinates, features, types, and biomarker expressions.
2. Outputs processed graphs in `.gpkl` format and corresponding visualization figures.
3. Logs errors encountered during processing.
4. Supports parallel processing with customizable worker count.

## Default Dataset Structure:
- Raw Data:
  - `voronoi/{region_id}.cell_data.csv`
  - `voronoi/{region_id}.cell_features.csv`
  - `voronoi/{region_id}.cell_types.csv`
  - `voronoi/{region_id}.expression.csv`
  - `voronoi/{region_id}.json`
- Processed Output:
  - `graph/{region_id}.gpkl`
  - `fig/{region_id}_voronoi.png`
  - `fig/{region_id}_graph.png`

## Usage:
```bash
python process_spacegm_data.py --data_root <path_to_data_root> --num_workers <number_of_workers>
```

Default paths:
- Processed graphs: `graph` folder
- Processed figures: `fig` folder

"""

def process_region(args):
    """ Process a single region's data """
    raw_data_path, graph_output_path, fig_output_path, region_id, error_log_file = args

    # Define file paths
    cell_coords_file = os.path.join(raw_data_path, f"{region_id}.cell_data.csv")
    cell_features_file = os.path.join(raw_data_path, f"{region_id}.cell_features.csv")
    cell_types_file = os.path.join(raw_data_path, f"{region_id}.cell_types.csv")
    expression_file = os.path.join(raw_data_path, f"{region_id}.expression.csv")
    voronoi_file = os.path.join(raw_data_path, f"{region_id}.json")

    graph_output_file = os.path.join(graph_output_path, f"{region_id}.gpkl")
    voronoi_img_output = os.path.join(fig_output_path, f"{region_id}_voronoi.png")
    graph_img_output = os.path.join(fig_output_path, f"{region_id}_graph.png")

    try:
        # Construct graph and save
        spacegm.construct_graph_for_region(
            region_id=region_id,
            cell_coords_file=cell_coords_file,
            cell_features_file=cell_features_file,
            cell_types_file=cell_types_file,
            cell_biomarker_expression_file=expression_file,
            voronoi_file=voronoi_file,
            graph_output=graph_output_file,
            voronoi_polygon_img_output=voronoi_img_output,
            graph_img_output=graph_img_output,
            figsize=10
        )
    except Exception as e:
        with open(error_log_file, "a") as log:
            log.write(f"Error processing region: {region_id}, Error: {str(e)}\n")
        print(f"Error processing region: {region_id}, skipped. See log for details.")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process Space-GM dataset regions into graphs and figures.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4).")

    args = parser.parse_args()

    # Set up paths
    data_root = args.data_root
    num_workers = args.num_workers

    raw_data_path = os.path.join(data_root, "voronoi")
    graph_output_path = os.path.join(data_root, "graph")
    fig_output_path = os.path.join(data_root, "fig")
    error_log_file = os.path.join(data_root, "error_regions.log")

    os.makedirs(graph_output_path, exist_ok=True)
    os.makedirs(fig_output_path, exist_ok=True)

    # Generate task list
    region_ids = [
        filename.split(".")[0] for filename in os.listdir(raw_data_path)
        if filename.endswith(".cell_data.csv")
    ]

    task_list = [
        (raw_data_path, graph_output_path, fig_output_path, region_id, error_log_file)
        for region_id in region_ids
    ]

    # Process the dataset
    print(f"Using {num_workers} CPU cores for parallel processing...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_region, task_list), total=len(task_list), desc="Processing Regions"))

    print("All regions processed successfully!")
    print(f"Check {error_log_file} for errors.")
