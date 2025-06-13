#!/usr/bin/env python
"""
Convert BGI GEF files to AnnData H5AD format with validation checks.
Make Sure you have installed the stereopy package following https://github.com/STOmics/Stereopy.
(You need a seperate conda environment to install stereopy, python version==3.8, then pip install stereopy,
if you meet any error, please refer to the github page for more details.)

You can set up the environment by:
--------------------------------
conda create -n st python=3.8
conda activate st
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple 
pip install stereopy 
--------------------------------

Example:
    python -m tic.data.bgi.transform_gef2h5ad \\
        --input /path/to/input.gef \\
        --output /path/to/output.h5ad \\
        --flavor scanpy \\
        --no-quality-check \\
        --log-level DEBUG
"""

import argparse
import sys
from stereo.io.reader import read_gef, stereo_to_anndata # type: ignore
import scanpy as sc
import logging
from pathlib import Path

def process_bgi_gef(
    input_gef_path: str,
    output_h5ad_path: str = None,
    flavor: str = 'scanpy',
    check_quality: bool = True,
    log_level: str = 'INFO'
) -> sc.AnnData:
    """
    Process BGI GEF file to AnnData H5AD format with validation checks.
    
    Parameters:
    -----------
    input_gef_path : str
        Path to input .gef file
    output_h5ad_path : str, optional
        Output path for .h5ad file. If None, will use same directory as input.
    flavor : str
        Conversion flavor ('scanpy' or 'seurat')
    check_quality : bool
        Whether to perform basic quality checks
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', etc.)
    
    Returns:
    --------
    AnnData object

    Example:
    --------
    >>> from tic.data.bgi.transform_gef2h5ad import process_bgi_gef
    >>> adata = process_bgi_gef(
    >>>     input_gef_path="Your GEF file path",
    >>>     output_h5ad_path="Your H5AD file path",
    >>>     flavor='scanpy',
    >>>     check_quality=True,
    >>>     log_level='INFO'
    >>> )
    """
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input path
        input_path = Path(input_gef_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_gef_path}")
        
        logger.info(f"Reading GEF file: {input_gef_path}")
        Exper_data = read_gef(input_gef_path)
        logger.debug(f"Raw Stereo data structure:\n{Exper_data}")
        
        # Conversion to AnnData
        logger.info("Converting to AnnData format")
        adata = stereo_to_anndata(data=Exper_data, flavor=flavor)
        
        # Basic checks
        if check_quality:
            logger.info("Performing quality checks")
            if adata.n_obs == 0:
                raise ValueError("No cells found in the data!")
            if adata.n_vars == 0:
                raise ValueError("No genes found in the data!")
            
            logger.info(f"Data contains {adata.n_obs} cells and {adata.n_vars} genes")
            logger.debug(f"Gene names sample: {adata.var_names[:5]}")
        
        # Set default output path if not provided
        if output_h5ad_path is None:
            output_path = input_path.with_suffix('.h5ad')
        else:
            output_path = Path(output_h5ad_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output
        logger.info(f"Writing output to {output_path}")
        adata.write_h5ad(output_path)
        logger.info("Processing completed successfully")
        
        return adata
        
    except Exception as e:
        logger.error(f"Error processing {input_gef_path}: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert BGI GEF files to AnnData H5AD format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="Path to input .gef file"
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help="Path to output .h5ad file (default: same dir as input with .h5ad extension)"
    )
    
    parser.add_argument(
        '-f', '--flavor',
        choices=['scanpy', 'seurat'],
        default='scanpy',
        help="Conversion flavor"
    )
    
    parser.add_argument(
        '--no-quality-check',
        action='store_false',
        dest='check_quality',
        help="Disable quality checks"
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set logging level"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        adata = process_bgi_gef(
            input_gef_path=args.input,
            output_h5ad_path=args.output,
            flavor=args.flavor,
            check_quality=args.check_quality,
            log_level=args.log_level
        )
        return 0
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())