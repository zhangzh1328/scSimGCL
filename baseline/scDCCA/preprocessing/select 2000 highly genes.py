import numpy as np
import pandas as pd
import h5py
#import gc
import scanpy as sc



file_path="example data\\muraro_preprocessed_log_genefilt.csv"
#data = pd.read_csv(file_path, index_col=None, header=None,sep=" ")####将第一行和第一列作为数据,会给数据添加上行名和列名
data = pd.read_csv(file_path, index_col=0, header=0,sep=" ")####将第一行和第一列作为行名和列名，适用于有行名和列名的数据
print(data)
data = pd.DataFrame(data)
adata = sc.AnnData(data)
print(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)  # 高变基因
adata = adata[:, adata.var['highly_variable']]
#sc.pp.scale(adata, max_value=3)
print(adata.X.shape)
print(adata.X)
np.savetxt("example data\\muraro_2000.txt", adata.X, delimiter=',')
print("完成")