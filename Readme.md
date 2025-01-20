# TIC: Temporal Inference of Cells
Temporal Inference of Cells (TIC) is a computational framework for analyzing cellular micro-environments using graph-based pseudo-time analysis. TIC integrates tools for graph construction, embedding preparation, pseudo-time trajectory computation, and biomarker trend visualization.

## Installation

To use TIC, you need to install its dependencies:

### Step 1: Clone the Repository
```
git clone https://github.com/JiahaoZhang-Public/tic.git
cd tic
git submodule update --init
```
### Step 2: Install Submodules
TIC relies on the following submodules:

* space-gm: For constructing graphs of cellular micro-environments.
* slingshot: Provides the core algorithm for pseudo-time trajectory computation.

Install these submodules:
```
# Install space-gm
git submodule update --init tools/space-gm
cd tools/space-gm
pip install -e .

# Install slingshot
cd ../slingshot
pip install -e .

# Return to the main directory
cd ../..
pip install -e .
```
## Project Framework
The directory structure for the TIC project is as follows:
```
TIC/
├── adapters/           # Adapter scripts for data and model integration
├── config/             # Configuration files (Hydra-based)
│   ├── pseudotime/     # Configuration files for pseudo-time analysis
├── core/               # Core modules for data processing and analysis
├── data/               # Directory for input datasets and intermediate files
├── notebook/           # Jupyter notebooks for example workflows
├── outputs/            # Output directory for pseudo-time analysis results
├── scripts/            # Scripts for running experiments and pipelines
├── tools/              # External tools and dependencies
├── utils/              # Utility scripts and helper functions
├── Readme.md           # Project documentation
├── requirements.txt    # Python dependencies
└── setup.py            # Installation script

```

## Working Pipeline:
### Step 1: Prepare Space-gm Dataset
To begin using TIC, you first need to prepare the Space-gm dataset. Space-gm constructs graphs based on cellular micro-environments that will later be used for pseudo-time analysis.

Run the following command to preprocess the data:
```bash
python scripts/process_spacegm_data.py --data_root <path_to_data_root> --num_workers <number_of_workers>
```
--data_root: Path to the root directory containing your raw data.
--num_workers: Number of parallel workers to use for data processing (adjust depending on your system's resources).
This script will prepare the necessary files, including the graph representations of the cellular micro-environments, to be used in the next steps.
### Step 2: Perform Pseudo-time Analysis
After processing the Space-gm data, you'll need to set up the configuration for the pseudo-time analysis. This will involve specifying parameters such as the graph structure, the starting points for pseudo-time estimation, and other relevant configurations.

1. Set up configuration:

* Navigate to the configuration directory: {project_root}/config/pseudotime/.
* Open and adjust the new.yaml file based on your dataset and analysis needs. This configuration file contains parameters that guide the pseudo-time analysis process.
* For more details, refer to the config/pseudotime/ReadMe.md file for documentation on the parameters you can adjust.

2. Run pseudo-time analysis: 
After configuring the new.yaml file, execute the following script to run the pseudo-time analysis:
```bash
python scripts/pseudotime_analysis.py
```
This script will compute the pseudo-time trajectory using the graph constructed in Step 1. The output will include the computed pseudo-time values for each cell, which can be further analyzed and visualized.

### Step 3: Visualize Pseudo-time
Once the pseudo-time analysis is completed, you can visualize the trajectory of the cellular micro-environment over pseudo-time.

To visualize the results, open the Jupyter notebook: 
```bash
notebook/visualize.ipynb
```
This notebook will guide you through the process of plotting pseudo-time trajectories, showing how different cellular features change over time.

### Step 4: Perform Causal Inference
After obtaining the pseudo-time trajectories, you may be interested in investigating causal relationships within your data. TIC integrates tools for causal inference to explore the drivers of cellular state transitions.

1. Set up configuration: Similar to the pseudo-time setup, navigate to the configuration directory {project_root}/config/pseudotime/, and configure the causal inference parameters in the new.yaml file.

2. Run causal inference: Once the configuration is set up, execute the following script to perform causal inference:
```bash
python scripts/causal_inference.py
```
This step will analyze the dependencies between cellular markers and compute causal relationships, potentially helping identify key regulatory factors in cellular state transitions.
