import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import os

parser = argparse.ArgumentParser(description='preprocessing')

# input & ouput
parser.add_argument('--input_h5ad_path', type=str, default=None,
                    help='path to input h5ad file')
parser.add_argument('--input_h5_path', type=str, default=None,
                    help='path to input 10X file')
parser.add_argument('--count_csv_path', type=str, default=None,
                    help='path to counts csv file')
parser.add_argument('--label_csv_path', type=str, default=None,
                    help='path to labels csv file')
parser.add_argument('--save_h5ad_dir', type=str, default="./",
                    help='dir to savings')

# preprocessing
parser.add_argument('--filter', action="store_true",
                    help='Whether do filtering')

parser.add_argument('--norm', action="store_true",
                    help='Whether do normalization')

parser.add_argument("--log", action="store_true",
                    help='Whether do log operation')

parser.add_argument("--scale", action="store_true",
                    help='Whether do scale operation')

parser.add_argument("--select_hvg", action="store_true",
                    help="Whether select highly variable gene")


def preprocess_to_h5ad(
        input_h5ad_path=None, input_h5_path=None, count_csv_path=None, label_csv_path=None, save_h5ad_dir="./",
        do_filter=False, do_log=False, do_select_hvg=False, do_norm=False, do_scale=False):
    # 1. read data from h5ad, h5 or csv files.
    if input_h5ad_path != None and input_h5_path == None and count_csv_path == None:
        adata = sc.read_h5ad(input_h5ad_path)
        print("Read data from h5ad file: {}".format(input_h5ad_path))

        _, h5ad_file_name = os.path.split(input_h5ad_path)
        save_file_name = h5ad_file_name

    elif input_h5_path != None and input_h5ad_path == None and count_csv_path == None:
        with h5py.File(input_h5_path, 'r') as f:
            X = f['X'][:]
            Y = f['Y'][:].astype(str)
        adata = sc.AnnData(X=X, obs={'x': Y})
        print("Read data from h5 file: {}".format(input_h5_path))

        _, input_h5_file_name = os.path.split(input_h5_path)
        save_file_name = input_h5_file_name.replace(".h5", ".h5ad")

    elif count_csv_path != None and input_h5ad_path == None and input_h5_path == None:
        # read the count matrix from the path
        count_frame = pd.read_csv(count_csv_path, index_col=0)
        print("counts shape:{}".format(count_frame.shape))

        if label_csv_path != None:
            label_frame = pd.read_csv(label_csv_path, index_col=0, header=0)
            print("labels shape:{}".format(label_frame.shape))
            if count_frame.shape[0] != label_frame.shape[0]:
                raise Exception("The shapes of counts and labels do not match!")
            label_frame.index = count_frame.index = count_frame.index.astype(str)
            adata = sc.AnnData(X=count_frame, obs=label_frame)
            print("Read data from csv file: {}".format(count_csv_path))
            print("Read laebl from csv file: {}".format(label_csv_path))
        else:
            adata = sc.AnnData(X=count_frame)
            print("Read data from csv file: {}".format(count_csv_path))

        _, counts_file_name = os.path.split(count_csv_path)
        save_file_name = counts_file_name.replace(".csv", ".h5ad").replace("_counts", "")

    # 2. preprocess anndata
    preprocessed_flag = do_filter | do_log | do_select_hvg | do_norm | do_scale
    # filter operation
    if do_filter == True:
        # filter the genes and cells
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        # filter the mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        low_MT_MASK = (adata.obs.pct_counts_mt < 5)
        adata = adata[low_MT_MASK]
        # filter the ERCC spike-in RNAs
        adata.var['ERCC'] = adata.var_names.str.startswith('ERCC-')  # annotate the group of ERCC spike-in as 'ERCC'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['ERCC'], percent_top=None, log1p=False, inplace=True)
        low_ERCC_mask = (adata.obs.pct_counts_ERCC < 10)
        adata = adata[low_ERCC_mask]

    # log operation
    if do_log and np.max(adata.X > 100):
        sc.pp.log1p(adata)
    # select highly variable gene
    if do_select_hvg:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
    # normalize
    if do_norm:
        sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True)
        adata.raw = adata
    # scale
    if do_scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)

    # 3. save preprocessed h5ad
    if save_h5ad_dir is not None:
        if os.path.exists(save_h5ad_dir) != True:
            os.makedirs(save_h5ad_dir)

        if preprocessed_flag == True:
            save_file_name = save_file_name.replace(".h5ad", "_preprocessed.h5ad")

        save_path = os.path.join(save_h5ad_dir, save_file_name)
        adata.write(save_path)
        print("Successfully generate preprocessed file: {}".format(save_file_name))
    return adata


if __name__ == "__main__":
    args = parser.parse_args()
    processed_adata = preprocess_to_h5ad(
        args.input_h5ad_path, args.input_h5_path, args.count_csv_path, args.label_csv_path, args.save_h5ad_dir,
        do_filter=args.filter, do_log=args.log, do_norm=args.norm, do_select_hvg=args.select_hvg, do_scale=args.scale)
