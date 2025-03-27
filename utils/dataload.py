import os
import anndata
import pandas as pd

from tic.constant import FILE_MAPPING
from tic.data.cell import Biomarkers, Cell
from tic.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from tic.data.tissue import Tissue

def process_region_to_tissue_generic(
    raw_dir: str, 
    region_id: str,
    csv_file_mapping: dict = FILE_MAPPING,
    required_columns: dict = None,
) -> Tissue:
    """
    A generic function to read raw single-cell data from multiple CSV files,
    merge them into Cell objects, and return a Tissue instance.

    This function is meant to be easily adapted to different single-cell datasets
    by customizing how CSV columns map to cell attributes (pos, size, cell_type, biomarkers, etc.).

    :param raw_dir: The directory containing raw CSV files (e.g. "Raw/").
    :param region_id: The region/tissue identifier (used in CSV file names).
    :param csv_file_mapping: A dictionary describing the files needed, for example:
        {
          'coords': f"{region_id}.cell_data.csv",
          'features': f"{region_id}.cell_features.csv",
          'types': f"{region_id}.cell_types.csv",
          'expression': f"{region_id}.expression.csv"
        }
      so you can unify reading logic from different dataset structures.
    :param required_columns: A dictionary specifying required columns for each file type, for example:
        {
          'coords': ['CELL_ID', 'X', 'Y'],
          'features': ['CELL_ID', 'SIZE'],
          'types': ['CELL_ID', 'CELL_TYPE'],
          'expression': ['CELL_ID', 'CD3e', 'CD4', ...],
        }
      If any required column is missing, the function can raise a warning or skip cells.
      If None, defaults are used or no checks are performed.

    :return: A Tissue object containing all the constructed Cell objects for this region.
    """
    if required_columns is None:
        required_columns = {
            'coords': ['CELL_ID', 'X', 'Y'],
            'features': ['CELL_ID', 'SIZE'],
            'types': ['CELL_ID', 'CELL_TYPE'],
            'expression': ['CELL_ID']  # The rest are assumed to be biomarker columns
        }

    # 1) Read each CSV
    coords_path = os.path.join(raw_dir, csv_file_mapping['coords'].format(region_id=region_id))
    features_path = os.path.join(raw_dir, csv_file_mapping['features'].format(region_id=region_id))
    types_path = os.path.join(raw_dir, csv_file_mapping['types'].format(region_id=region_id))
    expression_path = os.path.join(raw_dir, csv_file_mapping['expression'].format(region_id=region_id))

    coords_df = pd.read_csv(coords_path)
    features_df = pd.read_csv(features_path)
    types_df = pd.read_csv(types_path)
    expression_df = pd.read_csv(expression_path)

    # 2) Optionally drop columns if not needed (like ACQUISITION_ID).
    # For demonstration, we skip if not found.
    drop_cols = ['ACQUISITION_ID']
    for col in drop_cols:
        if col in expression_df.columns:
            expression_df = expression_df.drop(columns=[col])

    # 3) Build a dictionary to hold cell attributes.
    cell_info = {}

    # 3a) Process coords_df
    for _, row in coords_df.iterrows():
        cid = row['CELL_ID']
        x, y = row['X'], row['Y']
        cell_info[cid] = {
            'pos': (x, y)
        }

    # 3b) Process features_df
    for _, row in features_df.iterrows():
        cid = row['CELL_ID']
        if cid not in cell_info:
            cell_info[cid] = {}
        cell_info[cid]['size'] = row['SIZE']

    # 3c) Process types_df
    for _, row in types_df.iterrows():
        cid = row['CELL_ID']
        if cid not in cell_info:
            cell_info[cid] = {}
        cell_info[cid]['cell_type'] = row['CELL_TYPE']

    # 3d) Process expression_df -> biomarkers
    #    We treat all columns except 'CELL_ID' as biomarker columns.
    for _, row in expression_df.iterrows():
        cid = row['CELL_ID']
        if cid not in cell_info:
            cell_info[cid] = {}
        biomarker_dict = {}
        for col in expression_df.columns:
            if col != 'CELL_ID':
                biomarker_dict[col] = row[col]
        cell_info[cid]['biomarkers'] = Biomarkers(**biomarker_dict)

    # 4) Construct Cell objects, skip those missing essential info
    cells = []
    for cell_id, info in cell_info.items():
        if 'pos' not in info or 'size' not in info or 'cell_type' not in info or 'biomarkers' not in info:
            # If any required info is missing, skip.
            continue

        # Create Cell object
        c = Cell(
            tissue_id=region_id,
            cell_id=cell_id,
            pos=info['pos'],
            size=info['size'],
            cell_type=info['cell_type'],
            biomarkers=info['biomarkers']
        )
        cells.append(c)

    # 5) Create Tissue
    tissue = Tissue(tissue_id=region_id, cells=cells)
    return tissue


def process_region_to_anndata(
    raw_dir: str, 
    region_id: str,
    csv_file_mapping: dict,
    required_columns: dict = None
) -> anndata.AnnData:
    """
    读取单个 image 对应的多个 CSV 文件，将零散的单细胞数据整合成一个 AnnData 对象。
    
    :param raw_dir: 存放原始 CSV 文件的目录。
    :param region_id: 表示组织或区域的 ID，用于文件名匹配。
    :param csv_file_mapping: 文件映射字典，例如：
         {
           'coords': "{region_id}.cell_data.csv",
           'features': "{region_id}.cell_features.csv",
           'types': "{region_id}.cell_types.csv",
           'expression': "{region_id}.expression.csv"
         }
    :param required_columns: 针对不同文件需要的必选列定义，如：
         {
           'coords': ['CELL_ID', 'X', 'Y'],
           'features': ['CELL_ID', 'SIZE'],
           'types': ['CELL_ID', 'CELL_TYPE'],
           'expression': ['CELL_ID']  # 后续列均为 biomarker 列
         }
         如果为 None，则使用默认设置。
    :return: 构建好的 AnnData 对象，其中：
         - X：存储 biomarker 表达矩阵
         - obs：存储细胞的元信息（cell id, cell type, size 等）
         - obsm["spatial"]：存储空间坐标
         - var：存储 biomarker 的注释信息（可选）
    """
    # 默认必选列设置
    if required_columns is None:
        required_columns = {
            'coords': ['CELL_ID', 'X', 'Y'],
            'features': ['CELL_ID', 'SIZE'],
            'types': ['CELL_ID', 'CELL_TYPE'],
            'expression': ['CELL_ID']  # 剩余列均视为 biomarker
        }
    
    # 构造各个文件的路径
    coords_path = os.path.join(raw_dir, csv_file_mapping['coords'].format(region_id=region_id))
    features_path = os.path.join(raw_dir, csv_file_mapping['features'].format(region_id=region_id))
    types_path = os.path.join(raw_dir, csv_file_mapping['types'].format(region_id=region_id))
    expression_path = os.path.join(raw_dir, csv_file_mapping['expression'].format(region_id=region_id))
    
    # 读取 CSV 文件
    coords_df = pd.read_csv(coords_path)
    features_df = pd.read_csv(features_path)
    types_df = pd.read_csv(types_path)
    expression_df = pd.read_csv(expression_path)
    
    # 可选：删除不需要的列，例如 'ACQUISITION_ID'
    drop_cols = ['ACQUISITION_ID']
    for col in drop_cols:
        if col in expression_df.columns:
            expression_df = expression_df.drop(columns=[col])
    
    # 通过 CELL_ID 将各个 DataFrame 进行合并
    df = (
        coords_df.merge(features_df, on="CELL_ID")
                 .merge(types_df, on="CELL_ID")
                 .merge(expression_df, on="CELL_ID")
    )
    
    # 构建 AnnData 对象
    # 1. 从 expression 数据中提取 biomarker 表达矩阵 X
    #    假设 expression_df 除了 CELL_ID 外，其余列均为 biomarker
    biomarker_columns = [col for col in expression_df.columns if col != 'CELL_ID']
    X = df[biomarker_columns].to_numpy()
    
    # 2. 构建 obs：保存细胞元数据，如 cell type、size 等（CELL_ID 作为 index）
    obs = df[['CELL_ID', 'CELL_TYPE', 'SIZE']].copy()
    obs = obs.set_index('CELL_ID')
    
    # 3. 构建 obsm：保存空间坐标（X, Y）
    spatial_coords = df[['X', 'Y']].to_numpy()
    obsm = {"spatial": spatial_coords}
    
    # 4. 构建 var： biomarker 注释信息（这里仅使用 biomarker 名称作为 index）
    var = pd.DataFrame(index=biomarker_columns)
    
    # 5. 创建 AnnData 对象
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    
    # 可选：将其它全局信息存入 uns 中
    adata.uns["region_id"] = region_id
    
    return adata

if __name__ == "__main__":
    # Example usage
    raw_dir = "/Users/zhangjiahao/Project/tic/data/example/Raw"
    region_id = "UPMC_c001_v001_r001_reg001"
    region_1 = process_region_to_anndata(raw_dir=raw_dir,region_id = region_id, csv_file_mapping=FILE_MAPPING)
    print(region_1)
    tissue = Tissue.from_anndata(region_1)
    print(tissue)
    tissue_ann = tissue.to_anndata()
    print(tissue_ann)
    # tissue = process_region_to_tissue_generic(raw_dir, region_id)
    # tissue.to_graph(node_feature_fn=node_feature_fn, edge_index_fn=edge_index_fn, edge_attr_fn=edge_attr_fn)
    # print(tissue)

    # example_microe = tissue.get_microenvironment(tissue.cells[0].cell_id, k=3, microe_neighbor_cutoff=200.0)
    # center_cell = example_microe.export_center_cell_with_representations()
    # print(center_cell)

    # X, Y, X_labels, Y_labels = example_microe.prepare_for_causal_inference(y_biomarkers="PanCK")
    # print(X.shape, Y.shape, X_labels, Y_labels)    

