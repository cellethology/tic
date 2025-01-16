#scripts/causal_inference.py
"""
Created on Thur Jan 16 12:31 2025

@author: Jiahao Zhang
@Description: 
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from spacegm.data import CellularGraphDataset
from core.causal_inference import load_and_prepare_data, run_causal_inference_analysis
from utils.visualization import visualize_granger_results

def load_dataset(cfg):
    dataset = CellularGraphDataset(cfg.general.data_root, **cfg.dataset)
    return dataset

def perform_causal_analysis(cfg, dataset):
    raw_dir = cfg.general.data_root + '/voronoi'
    causal_input =  load_and_prepare_data(dataset,cfg.causal_inference.pseudotime_file,raw_dir)
    run_causal_inference_analysis(causal_input,"PropensityScoreMatching")

def visualize_results(p_values_df, significance_df):
    visualize_granger_results(p_values_df, significance_df)

@hydra.main(config_path="../config/test", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize and load data
    dataset = load_dataset(cfg)

    # Perform causal analysis
    perform_causal_analysis(cfg, dataset)

if __name__ == "__main__":
    main()
