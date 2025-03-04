import os
import pandas as pd

from core.data.cell import Biomarkers, Cell
from core.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from core.data.plot import plot_microe_graph, plot_tissue_graph
from core.data.tissue import Tissue


def process_region_to_tissue(raw_dir, region_id):
    # 读取CSV文件
    cell_coords_df = pd.read_csv(f"{raw_dir}/{region_id}.cell_data.csv")
    cell_features_df = pd.read_csv(f"{raw_dir}/{region_id}.cell_features.csv")
    cell_types_df = pd.read_csv(f"{raw_dir}/{region_id}.cell_types.csv")
    biomarkers_df = pd.read_csv(f"{raw_dir}/{region_id}.expression.csv")
    
    # 去除ACQUISITION_ID列
    biomarkers_df = biomarkers_df.drop(columns=["ACQUISITION_ID"])

    # 将每个cell的信息合并成一个字典
    cell_info = {}
    for _, row in cell_coords_df.iterrows():
        cell_info[row['CELL_ID']] = {'pos': (row['X'], row['Y'])}

    for _, row in cell_features_df.iterrows():
        cell_info[row['CELL_ID']]['size'] = row['SIZE']

    for _, row in cell_types_df.iterrows():
        cell_info[row['CELL_ID']]['cell_type'] = row['CELL_TYPE']

    # 处理biomarkers数据
    for _, row in biomarkers_df.iterrows():
        biomarkers = {col: row[col] for col in biomarkers_df.columns if col != 'CELL_ID'}
        cell_info[row['CELL_ID']]['biomarkers'] = Biomarkers(**biomarkers)

    # 创建Cell对象，排除缺少信息的细胞
    cells = []
    for cell_id, info in cell_info.items():
        # 检查是否缺少任何必要信息
        if 'pos' not in info or 'size' not in info or 'cell_type' not in info or 'biomarkers' not in info:
            # print(f"Skipping cell {cell_id} due to missing information.")
            continue  # 如果缺少信息，跳过该细胞

        # 创建Cell对象
        cell = Cell(cell_id, info['pos'], info['size'], info['cell_type'], info['biomarkers'])
        cells.append(cell)

    # 创建Tissue对象
    tissue = Tissue(region_id, cells)
    
    return tissue

if __name__ == "__main__":
    raw_dir = "/Users/zhangjiahao/Project/tic/data/example/voronoi"
    tissue = process_region_to_tissue(raw_dir,region_id="UPMC_c001_v001_r001_reg001")

    start_time = os.times()
    print(f"Start time: {start_time}")
    print(tissue.tissue_id)
    tissue.to_graph(node_feature_fn=node_feature_fn, edge_index_fn=edge_index_fn, edge_attr_fn=edge_attr_fn)
    end_time = os.times()
    print(f"End time: {end_time}")
    print(f"Total time: {end_time[4] - start_time[4]}")
    plot_tissue_graph(tissue)

    microE = tissue.get_microenvironment(tissue.cells[0].cell_id, k=3)
    neiborhood_biomarker_matrix = microE.get_neighborhood_biomarker_matrix()
    print(neiborhood_biomarker_matrix.shape)
    plot_microe_graph(microE)
