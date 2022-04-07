import pandas as pd
import scanpy as sc
import numpy as np
import os

def read_HER2ST(exprs_path, coords_path, process_sc = True, subseting = True):
    """
    read into the expression data and coordinates information for ST or 10x visium data.
    :param exprs_path: file path to the expression matrix (HER2ST).
    :param coords_path: file path to the coordinates information (HER2ST).
    :param process_sc: boolean for preprocessing the data.
    :param subseting: selecting subset data based on hvgs.
    :return:
    """

    slice = sc.read(exprs_path)

    # preprocessing
    if process_sc:
        min_cells = round(slice.shape[0] * .04)
        min_genes = round(slice.shape[1] / 20)
        sc.pp.filter_genes(slice, min_cells=min_cells)
        sc.pp.filter_cells(slice, min_genes=min_genes)

    # read and find common cells by the coords combinations.
    coords = pd.read_csv(coords_path, delimiter = "\t")
    slice_cells = slice.obs.index.tolist()
    coords["x"] = coords["x"].astype(str)
    coords["y"] = coords["y"].astype(str)
    coords_cells = coords['x'].str.cat(coords['y'], sep = "x")
    coords_cells = coords_cells.tolist()
    common_cells = set(slice_cells).intersection(coords_cells)
    ix1 = [slice_cells.index(i) for i in common_cells]
    ix2 = [coords_cells.index(j) for j in common_cells]

    slice = slice[ix1, :]
    coords = coords.iloc[ix2, :]
    grid_coords = np.array(coords)[:, 2:4]
    slice.obsm['spatial'] = grid_coords

    pixel_coords = np.array(coords)[:, 4:6]
    slice.obsm['pixel'] = pixel_coords

    # conduct analysis by using only hvgs.
    if subseting:
        # normalization
        Y = slice.X
        S = Y.sum(axis=1)
        sc.pp.normalize_total(slice, target_sum=np.median(S))
        sc.pp.log1p(slice)
        sc.pp.highly_variable_genes(slice, min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg = slice.var["highly_variable"].tolist()
        hvg_ixs = [i for i, v in enumerate(hvg) if v]
        slice = slice[:, hvg_ixs]

    return slice


def read_LIBD(h5ad_path, process_sc = True, subseting = True):
    """
    read into the expression data and coordinates information for ST or 10x visium data.
    :param h5ad_path: file path to the h5ad object (10x spatial LIBD).
    :param process_sc: boolean for preprocessing the data.
    :param subseting: selecting subset data based on hvgs.
    :return:
    """
    h5ad_exisit = os.path.isfile(h5ad_path)
    if h5ad_exisit==False:
        import sys
        sys.exit("no h5ad file was found")
    slice = sc.read_h5ad(h5ad_path)
    # preprocessing
    if process_sc:
        min_cells = round(slice.shape[0] * .04)
        min_genes = round(slice.shape[1] / 20)
        sc.pp.filter_genes(slice, min_cells=min_cells)
        sc.pp.filter_cells(slice, min_genes=min_genes)

    # read and find common cells by the coords combinations.
    coords = slice.obs[['row', 'col']]
    slice.obsm['spatial'] = np.array(coords)
    pixel_coords = slice.obs[['pixelrow', 'pixelcol']]
    slice.obsm['pixel'] = np.array(pixel_coords)

    # conduct analysis by using only hvgs.
    if subseting:
        # normalization
        Y = slice.X.A
        S = Y.sum(axis=1)
        sc.pp.normalize_total(slice, target_sum=np.median(S))
        sc.pp.log1p(slice)
        sc.pp.highly_variable_genes(slice, min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg = slice.var["highly_variable"].tolist()
        hvg_ixs = [i for i, v in enumerate(hvg) if v]
        slice = slice[:, hvg_ixs]

    return slice