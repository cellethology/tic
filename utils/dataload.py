import os
import pandas as pd

from core.constant import FILE_MAPPING
from core.data.cell import Biomarkers, Cell
from core.data.tissue import Tissue

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

