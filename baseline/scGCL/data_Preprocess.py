import csv
import numpy as np
import pandas as pd
import cmath as cm
import h5py
from scipy import sparse
import scipy
from sklearn.utils.class_weight import compute_class_weight
import sklearn
import cluster
import scanpy as sc
from matplotlib import rcParams
from sklearn.metrics import normalized_mutual_info_score, pairwise, adjusted_rand_score,silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
def read_cell_to_image(data_path,label_path,class_num):
    data = pd.read_csv(data_path, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    label = np.array(label)[:, 1]
    data_array = []
    data_for_count = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('double'))
        gene_list_all = np.sum(gene_list_for_count)
        gene_list_median = np.median(gene_list_for_count)
        gene_list_for_count = gene_list_for_count*(gene_list_median/gene_list_all)
        data_for_count.append(gene_list_all/gene_list_median)
        gene_list_for_count = np.log2(gene_list_for_count+1)
        gene_list = gene_list_for_count.tolist()
        gene_len =  len(gene_list)
        figure_size = int(gene_len**0.5)
        if figure_size*figure_size == gene_len:
            data = np.array(gene_list).reshape(figure_size,figure_size,1).astype('double')
        else:
            for j in range((figure_size+1)**2-gene_len):
                gene_list.append(0)
            data =  np.array(gene_list).reshape(figure_size+1, figure_size+1, 1).astype('double')
        data_array.append(data)
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[int(label[i][5]) - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array,data_label,gene_name,cell_name,data_for_count

def read_cell(data_path,label_path,class_num):
    data = pd.read_csv(data_path, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    label = np.array(label)[:, 1]
    data_array = []
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(arr1[1:, i + 1].astype('float64'))
        gene_list_all = np.sum(gene_list_for_count)
        gene_list_median = np.median(gene_list_for_count)
        gene_list_for_count = gene_list_for_count * (gene_list_median / gene_list_all)
        gene_list_for_count = np.log2(gene_list_for_count + 1)
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        data_array.append(np.array(gene_list))

    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[int(label[i][5]) - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    print(data_array)
    return data_array, data_label,gene_name,cell_name

def nomalize_for_AF(filename,gene_num,raw, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        X = np.ceil(X).astype(np.int)
        adata = sc.AnnData(X)
        adata.obs['Group'] = cell_label
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        count = adata.X
        # if raw == False:
        #     a = pd.DataFrame(count).T
        #     a.to_csv("./results/raw.csv")
        return count,adata.obs['Group']
def nomalize_for_COVID(file_name,label_path,gene_num):
    data = pd.read_csv(file_name, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=";")
    # data_label = []
    label = np.array(label)[:, 4]
    cell_type, cell_label = np.unique(label, return_inverse=True)
    data_label = []
    for i in range(len(cell_label)):
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    print(data_label)
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    X = arr1[1:, 1:].T.astype(np.int)
    print(X)
    X = np.ceil(X).astype(np.int)
    adata = sc.AnnData(X)
    # print(cell_type.shape)
    adata.obs['Group'] = data_label
    adata.obs['cell_name'] = cell_name
    adata.var['Gene'] = gene_name
    adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=False,
                      logtrans_input=True)
    count = adata.X
    a = pd.DataFrame(count).T
    a.to_csv("./results/raw.csv")
    return count, cell_label,adata.obs['size_factors']

def nomalize_for_AD(file_name,label_path,gene_num):
    data = pd.read_csv(file_name, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    # data_label = []
    label = np.array(label)[:, 2]
    cell_type, cell_label = np.unique(label, return_inverse=True)
    data_label = []
    for i in range(len(cell_label)):
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    print(data_label)
    arr1 = np.array(data)
    gene_name = np.array(arr1[1:, 0])
    cell_name = np.array(arr1[0, 1:])
    X = arr1[1:, 1:].T
    adata = sc.AnnData(X)
    # print(cell_type.shape)
    adata.obs['Group'] = data_label
    adata.obs['cell_name'] = cell_name
    adata.var['Gene'] = gene_name
    adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=False,
                      logtrans_input=True)
    count = adata.X
    # a = pd.DataFrame(count).T
    # a.to_csv("./results/raw.csv")
    return count, cell_label,adata.obs['size_factors'],adata.var['Gene']


def nomalize_for_Zeisel(file_name,label_path,gene_num,rate):
    data = pd.read_csv(file_name, header=None, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    # data_label = []
    label = np.array(label)[:, 1]
    cell_type, cell_label = np.unique(label, return_inverse=True)
    data_label = []
    for i in range(len(cell_label)):
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    print(data_label.shape)
    arr1 = np.array(data)
    cell_name = np.array(arr1[1:, 0])
    gene_name= np.array(arr1[0, 1:])
    X = arr1[1:, 1:]
    print(X.shape)
    adata = sc.AnnData(X)
    # print(cell_type.shape)
    adata.obs['Group'] = data_label
    adata.obs['cell_name'] = cell_name
    adata.var['Gene'] = gene_name
    adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=False,
                      logtrans_input=True)
    #create dataset
    count = adata.X
    X_zero, i, j, ix = impute_dropout(count, 1, 0.1)
    a = pd.DataFrame(X_zero).T
    a.to_csv("./results/raw-"+str(0.1)+".csv")
    X_zero, i, j, ix = impute_dropout(count, 1, 0.3)
    a = pd.DataFrame(X_zero).T
    a.to_csv("./results/raw-" + str(0.3) + ".csv")
    X_zero, i, j, ix = impute_dropout(count, 1, 0.5)
    a = pd.DataFrame(X_zero).T
    a.to_csv("./results/raw-" + str(0.5) + ".csv")
    return X_zero, cell_label,adata.obs['size_factors'],adata.var['Gene']

def create_adata_AD(file_name,file_label,original_file,funtion_name,file_res,gene_list):
    data = pd.read_csv(file_name, header=None, sep=",")
    label_ = pd.read_csv(file_label, header=0, sep=",")
    label = np.array(label_)[:, 2]
    batch_label = np.array(label_)[:, 1]

    cell_type, cell_label = np.unique(label, return_inverse=True)
    class_num = np.array(cell_label).max() + 1
    data_label = []
    for i in range(len(cell_label)):
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    X = np.array(data)[1:, 1:].T
    count_for = torch.tensor(X)
    count_for = F.normalize(count_for, dim=-1, p=2).detach().cpu().numpy()
    adata = sc.AnnData(count_for)
    adata.obs[funtion_name] = data_label
    _, _, _, gene =  nomalize_for_AD(original_file,file_label,2048)
    a = pd.DataFrame(gene)
    a.to_csv(gene_list)
    data_o = pd.read_csv(gene_list, header=0, sep=",")
    adata.var_names = np.array(data_o)[:,1]
    print(adata.var_names)
    adata.obs[funtion_name] = label
    adata.obs["batches"] = batch_label
    # NMI_list = []
    # ARI_list = []
    # acc_list = []
    # sc_list = []
    # estimator = KMeans(n_clusters=class_num)
    # for i in range(10):
    #     estimator.fit(count_for)
    #     y_pred = estimator.predict(count_for)
    #     nmi = normalized_mutual_info_score(cell_label, y_pred, average_method='arithmetic')
    #     ari = adjusted_rand_score(cell_label, y_pred)
    #     acc_ = acc(cell_label, y_pred)
    #     sca = silhouette_score(count_for, y_pred, metric='euclidean')
    #     # zzz = np.concatenate((cell_label.reshape(3660,1),y_pred.reshape(3660,1)),axis = 1)
    #     # a = pd.DataFrame(zzz)
    #     # a.to_csv("./compare_funtion/AFGRL/AFGRL-master/results/zzz.csv")
    #     NMI_list.append(nmi)
    #     ARI_list.append(ari)
    #     acc_list.append(acc_)
    #     sc_list.append(sca)
    #
    # nmi = sum(NMI_list) / len(NMI_list)
    # ari = sum(ARI_list) / len(ARI_list)
    # acc_ = sum(acc_list) / len(acc_list)
    # sca = sum(sc_list) / len(sc_list)
    # data_result = {"nmi": nmi, "ari": ari, "acc": acc_, "sc": sca}
    # with open(file_res, "a", encoding='utf-8')as f:
    #     f.writelines("--------------"+funtion_name+"----------\n")
    #     f.writelines("--------------ari:" + str(ari) + "----------\n")
    #     f.writelines("--------------nmi:" + str(nmi) + "----------\n")
    #     f.writelines("--------------acc:" + str(acc_) + "----------\n")
    #     f.writelines("--------------sc:" + str(sca) + "----------\n")
    #     f.close()
    # print("---------ari:" + str(ari))
    # print("---------nmi:" + str(nmi))
    # print("---------acc:" + str(acc_))
    # print("---------sc:" + str(sca))
    return adata, cell_type

def create_adata_Klein(file_name,file_label,funtion_name):
    data_set = "Klein"
    data = pd.read_csv(file_name, header=None, sep=",")
    label_ = pd.read_csv(file_label, header=0, sep=",")
    label = np.array(label_)[:, 1]
    cell_type, cell_label = np.unique(label, return_inverse=True)
    class_num = np.array(cell_label).max() + 1
    data_label = []
    for i in range(len(cell_label)):
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    X = np.array(data)[1:, 1:].T
    count_for = torch.tensor(X)
    count_for = F.normalize(count_for, dim=-1, p=2).detach().cpu().numpy()
    a = pd.DataFrame(count_for).T
    a.to_csv("./compare_funtion/AFGRL/AFGRL-master/results/"+funtion_name+"-"+data_set+"-imputed.csv")
    adata = sc.AnnData(X)
    adata.obs[funtion_name] = data_label.astype("str")
    # adata.obs[funtion_name].cat.categories = ['0','1','2','3'].astype('category')
    # NMI_list = []
    # ARI_list = []
    # acc_list = []
    # sc_list = []
    # estimator = KMeans(n_clusters=class_num)
    # for i in range(10):
    #     estimator.fit(count_for)
    #     y_pred = estimator.predict(count_for)
    #     nmi = normalized_mutual_info_score(cell_label, y_pred, average_method='arithmetic')
    #     ari = adjusted_rand_score(cell_label, y_pred)
    #     acc_ = acc(cell_label, y_pred)
    #     sca = silhouette_score(count_for, y_pred, metric='euclidean')
    #     # zzz = np.concatenate((cell_label.reshape(3660,1),y_pred.reshape(3660,1)),axis = 1)
    #     # a = pd.DataFrame(zzz)
    #     # a.to_csv("./compare_funtion/AFGRL/AFGRL-master/results/zzz.csv")
    #     NMI_list.append(nmi)
    #     ARI_list.append(ari)
    #     acc_list.append(acc_)
    #     sc_list.append(sca)
    #
    # nmi = sum(NMI_list) / len(NMI_list)
    # ari = sum(ARI_list) / len(ARI_list)
    # acc_ = sum(acc_list) / len(acc_list)
    # sca = sum(sc_list) / len(sc_list)
    # data_result = {"nmi": nmi, "ari": ari, "acc": acc_, "sc": sca}
    # with open(file_res, "a", encoding='utf-8')as f:
    #     f.writelines("--------------"+funtion_name+"----------\n")
    #     f.writelines("--------------ari:" + str(ari) + "----------\n")
    #     f.writelines("--------------nmi:" + str(nmi) + "----------\n")
    #     f.writelines("--------------acc:" + str(acc_) + "----------\n")
    #     f.writelines("--------------sc:" + str(sca) + "----------\n")
    #     f.close()
    # print("---------ari:" + str(ari))
    # print("---------nmi:" + str(nmi))
    # print("---------acc:" + str(acc_))
    # print("---------sc:" + str(sca))
    return adata, cell_type





def create_adata(filename,file_original,function_name,file_res):
    data = pd.read_csv(filename, header=None, sep=",")

    with h5py.File(file_original, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        print(np.array(f["obs"]))
        batch_name = np.array(f["obs"]["organism"])
        batch_type, batch_label = np.unique(batch_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
    print(class_num)
    X = np.array(data.T)[1:,1:]
    count_for = torch.tensor(X)
    count_for = F.normalize(count_for, dim = -1, p = 2).detach().cpu().numpy()
    #X = np.ceil(X).astype(np.int)
    # reducer = umap.UMAP(random_state=20150101)
    # embedding = reducer.fit_transform(X)
    adata = sc.AnnData(count_for)
    cell_name = cell_name.astype(str)
    batch_name = batch_name.astype(str)
    adata.obs["batches"] = batch_name
    adata.obs[function_name] = cell_name
    NMI_list = []
    ARI_list = []
    acc_list = []
    sc_list = []
    estimator = KMeans(n_clusters=class_num)
    for i in range(10):
        estimator.fit(count_for)
        y_pred = estimator.predict(count_for)
        nmi = normalized_mutual_info_score(cell_label, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(cell_label, y_pred)
        acc_ = acc(cell_label, y_pred)
        sca = silhouette_score(count_for,y_pred,metric='euclidean')
        # zzz = np.concatenate((cell_label.reshape(3660,1),y_pred.reshape(3660,1)),axis = 1)
        # a = pd.DataFrame(zzz)
        # a.to_csv("./compare_funtion/AFGRL/AFGRL-master/results/zzz.csv")
        NMI_list.append(nmi)
        ARI_list.append(ari)
        acc_list.append(acc_)
        sc_list.append(sca)


    nmi = sum(NMI_list) / len(NMI_list)
    ari = sum(ARI_list) / len(ARI_list)
    acc_ = sum(acc_list) / len(acc_list)
    sca = sum(sc_list) / len(sc_list)
    data_result = {"nmi":nmi,"ari":ari,"acc":acc_,"sc":sca}
    with open(file_res, "a", encoding='utf-8')as f:
        f.writelines("--------------"+function_name+"----------\n")
        f.writelines("--------------ari:" + str(ari) + "----------\n")
        f.writelines("--------------nmi:" + str(nmi) + "----------\n")
        f.writelines("--------------acc:" + str(acc_) + "----------\n")
        f.writelines("--------------sc:" + str(sca) + "----------\n")
        f.close()
    print("---------ari:"+ str(ari))
    print("---------nmi:" + str(nmi))
    print("---------acc:" + str(acc_))
    print("---------sc:" + str(sca))
    return adata,cell_type,data_result
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10
def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def read_cell_for_h5(filename,gene_num, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max()+1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        print(X.shape)
        indicator = np.where(X > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()

        data = X[:, sum_gene > 10]

        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-gene_num:]
        data = data[:, index]
        # print(data.shape)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        data = scaler.fit_transform(data)
        print(X)
    return data,data_label,cell_label,cell_type,cell_type.shape[0],obs,var

def read_cell_for_h5_to_csv(filename,rate,gene_num, output_file,sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_count_array=[]
        data_array = []
        semi_label_index = []
        np.random.seed(1)
        semi_label = np.random.permutation(cell_name.shape[0])
        semi_label_index = int((1 - rate) * cell_name.shape[0])
        semi_label_train = semi_label[:semi_label_index]
        semi_label_real = cell_label[semi_label_train]
        weight_label_train = semi_label[semi_label_index+1:]
        cell_label[semi_label_train] = class_num
        class_weight = 'balanced'
        weight = compute_class_weight(class_weight, np.array(range(class_num)), cell_label[weight_label_train])
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num+1)
            x[cell_label[i]] = 1
            data_label.append(x)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        print(X.shape)
        indicator = np.where(X > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()

        data = X[:, sum_gene>10]


        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-gene_num:]
        data = data[:, index]
        # print(data.shape)
        # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        # scaler = scaler.fit(data)
        # data = scaler.fit_transform(data)
        for i in range(cell_label.shape[0]):
            gene_list_for_count = np.array(data[i,0:].astype('double'))
            # 把单细胞表达数据转化为图片
            data_array.append(gene_list_for_count)

        a = pd.DataFrame(data_array).T
        a.to_csv(output_file)

def read_cell_for_h5_to_image(filename,rate,gene_num, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_count_array=[]
        data_array = []
        semi_label_index = []
        np.random.seed(0)
        semi_label = np.random.permutation(cell_name.shape[0])
        # rate表示取多少标签进行训练
        semi_label_index = int((1 - rate) * cell_name.shape[0])
        semi_label_train = semi_label[:semi_label_index]
        #为了测试准确率用
        semi_label_real = cell_label[semi_label_train]
        weight_label_train = semi_label[semi_label_index+1:]
        cell_label[semi_label_train] = class_num
        class_weight = 'balanced'
        weight = compute_class_weight(class_weight, np.array(range(class_num)), cell_label[weight_label_train])
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num+1)
            x[cell_label[i]] = 1
            data_label.append(x)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        print(X.shape)
        indicator = np.where(X > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()

        data = X[:, sum_gene>10]


        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-gene_num:]
        data = data[:, index]
        # print(data.shape)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        data = scaler.fit_transform(data)
        for i in range(cell_label.shape[0]):
            gene_list_for_count = np.array(data[i,0:].astype('double'))
            # 把单细胞表达数据转化为图片
            gene_list_max = np.max(gene_list_for_count)
            data_count_array.append(gene_list_max)

            gene_list = gene_list_for_count.tolist()
            gene_len = len(gene_list)
            figure_size = int(gene_len ** 0.5)
            if figure_size * figure_size == gene_len:
                data_train = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
                figure_size_real = figure_size
            else:
                for j in range((figure_size + 1) ** 2 - gene_len):
                    gene_list.append(0)
                data_train = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
                figure_size_real = figure_size + 1
            data_array.append(data_train)

        test_data = np.array(data_array)[semi_label_train]
    return data_array,data_label,cell_type,cell_type.shape[0],obs,var,figure_size_real,data_count_array,weight,scaler,semi_label_real,test_data

def read_cell_for_h5_pre_clustering(filename,rate,gene_num,cluster_num, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        # cell_name = np.array(f["obs"]["cell_type1"])
        # cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        # label_1 = cluster.k_means(embedding, data[i].cell_type_num)
        data_count_array = []
        data_array = []
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        print(X.shape)
        indicator = np.where(X > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()

        data = X[:, sum_gene > 10]

        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-gene_num:]
        data = data[:, index]
        # print(data.shape)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        data = scaler.fit_transform(data)
        for i in range(data.shape[0]):
            gene_list_for_count = np.array(data[i, 0:].astype('double'))
            gene_list_max = np.max(gene_list_for_count)
            data_count_array.append(gene_list_max)

            gene_list = gene_list_for_count.tolist()
            gene_len = len(gene_list)
            figure_size = int(gene_len ** 0.5)
            if figure_size * figure_size == gene_len:
                data_train = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
                figure_size_real = figure_size
            else:
                for j in range((figure_size + 1) ** 2 - gene_len):
                    gene_list.append(0)
                data_train = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
                figure_size_real = figure_size + 1
            data_array.append(data_train)



        class_num = cluster_num
        data_label = []
        label_1 = cluster.k_means(data, cluster_num)
        semi_label_index = []
        np.random.seed(6)
        semi_label = np.random.permutation(label_1.shape[0])
        semi_label_index = int((1 - rate) * label_1.shape[0])
        semi_label_train = semi_label[:semi_label_index]
        # 为了测试准确率用
        semi_label_real = label_1[semi_label_train]
        weight_label_train = semi_label[semi_label_index + 1:]
        label_1[semi_label_train] = class_num
        class_weight = 'balanced'
        weight = compute_class_weight(class_weight, np.array(range(class_num)), label_1[weight_label_train])
        # semi_label_index = int((1 - rate) * cell_name.shape[0])
        # semi_label_train = semi_label[:semi_label_index]
        # semi_label_real = cell_label[semi_label_train]
        # weight_label_train = semi_label[semi_label_index + 1:]
        # cell_label[semi_label_train] = class_num
        # class_weight = 'balanced'
        # weight = compute_class_weight(class_weight, np.array(range(class_num)), cell_label[weight_label_train])
        for i in range(label_1.shape[0]):
            x = np.zeros(class_num + 1)
            x[label_1[i]] = 1
            data_label.append(x)
        cell_type = np.array(range(class_num))
        test_data = np.array(data_array)[semi_label_train]
    return data_array, data_label, cell_type, cell_type.shape[0], obs, var, figure_size_real, data_count_array, weight, scaler, semi_label_real, test_data





def read_cell_for_h5_imputed(array_file,filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        print(np.array(f["obs"]))
        batch_name = np.array(f["obs"]["organ"])
        batch_type, batch_label = np.unique(batch_name, return_inverse=True)
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max()+1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        data = pd.read_csv(array_file, header=None, sep=",")
        arr1 = np.array(data)
        data_array = []
        cell_num = np.shape(cell_name)[0]
        print(cell_num)

        for i in range(cell_num):
            gene_list_for_count = np.array(arr1[1:, i + 1].astype('float64'))
            # gene_list_all = np.sum(gene_list_for_count)
            # gene_list_median = np.median(gene_list_for_count)
            # gene_list_for_count = gene_list_for_count * (gene_list_median / gene_list_all)
            # gene_list_for_count = np.log2(gene_list_for_count + 1)
            gene_list = gene_list_for_count.tolist()
            gene_len = len(gene_list)
            data_array.append(np.array(gene_list))
        data_array = np.array(data_array)
    return data_array,data_label,cell_type,cell_type.shape[0],obs,var,batch_type,batch_label

def read_cell_for_interpretable_imputed(data_path,label_path,class_num,data_set,gene_num,type, sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=0, sep=",")
    if data_set == "AD" or data_set == "HP" or data_set == "lung_cancer" or data_set == "AD_cluster":
        label = pd.read_csv(label_path, header=0, sep=",")
    elif data_set == "COVID-19" or data_set=="COVID-19_cluster":
        label = pd.read_csv(label_path, header=0, sep=";")
    elif data_set == "breast_cancer":
        label = pd.read_csv(label_path, header=0, sep=",")
    arr1 = np.array(data)
    if data_set == "Klein_old" or data_set=="Klein_new":
        label = np.array(label).astype("str")
    X = arr1[:, 1:].T
    indicator = np.where(X > 0, 1, 0)
    sum_gene = np.sum(indicator, axis=0).flatten()
    #find gene!!
    data = X[:, sum_gene > 10]

    var_gene = np.var(data, axis=0)
    index = np.argsort(var_gene)[-gene_num:]
    data = data[:, index]
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.fit_transform(data)
    gene_name = np.array(arr1[index, 0])
    cell_name = np.array(arr1[0, 1:])
    if data_set == "AD" or data_set == "HP"or data_set == "lung_cancer":
        label = np.array(label)[:, 1]
    elif data_set == "COVID-19":
        label = np.array(label)[:, 2]
    elif data_set == "COVID-19_cluster":
        label = np.array(label)[:, 4]
    elif data_set == "breast_cancer":
        label = np.array(label)[:,1]
    if data_set == "AD":
        label2number = {"AD": 1, "ct": 2}
    elif data_set == "AD_cluster":
        label = np.array(label)[:, 2]
        label2number = {'OPC': 1, 'astro': 2, 'doublet': 3, 'endo': 4, 'mg': 5, 'neuron': 6, 'oligo': 7, 'unID': 8}
    elif data_set == "HP":
        label2number = {"""disease: Type 2 Diabetic""": 1, """disease: Non-Diabetic""": 2}
    elif data_set == "COVID-19":
        label2number = {"C-1": 2, "C-2": 2, "C-3": 2, "C-4": 2, "COV-1": 1, "COV-2": 1, "COV-3": 1, "COV-4": 1,
                        "COV-5": 1, "COV-6": 1, "COV-7": 1, "COV-8": 1}
    elif data_set == "COVID-19_cluster":
        label2number = {"Non-classical Monocytes": 1, "CD8 EM": 2, "Classical/intermediate Monocytes": 3,
                        "Naive CD4/CD8": 4, "Memory B Cells": 5,
                        "NK Cells": 6, "CD4 CM": 7, "CD4 EM": 8, "Naive B Cells": 9, "Plasma Cells": 10,
                        "T/NK_Cells": 11, "CD8 EMRA like": 12, "Platelets": 13}
    elif data_set == "breast_cancer":
        label2number = {"cancer": 1, "contrl": 2}
    elif data_set == "lung_cancer":
        label2number = {"AIS":1,"MIA":2,"IAC":3}
    elif data_set == "Klein_new":
        label2number = {"0.0":1,"0.5":2,"1.0":3,"1.5":4,"2.0":5,"2.5":6,"3.0":7}
    data_array = []

    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        if type=="interpretable":
            gene_list_for_count = np.array(data[i, 0:].astype('double'))
        elif type == "normal":
            gene_list_for_count = np.array(arr1[0:, i + 1].astype('double'))
        # 把单细胞表达数据转化为图片
        gene_list = gene_list_for_count.tolist()
        data_array.append(np.array(gene_list))
    data_array = np.array(data_array)
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    return data_array, data_label, gene_name, cell_name,label,scaler

def read_interpretable_for_train(data_path,label_path,class_num,data_set,rate,gene_num,sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=0, sep=",")
    if data_set == "COVID-19" or data_set == "COVID-19_cluster":
        label = pd.read_csv(label_path, header=0, sep=";")
    else:
        label = pd.read_csv(label_path, header=0, sep=",")
    data_label = []
    label2number={}
    if data_set == "AD":
        label = np.array(label)[:, 1]
        label2number = {"AD": 1, "ct": 2}
    elif data_set == "AD_cluster":
        label = np.array(label)[:, 2]
        label2number = {'OPC':1, 'astro':2, 'doublet':3, 'endo':4, 'mg':5, 'neuron':6, 'oligo':7, 'unID':8}
    elif data_set == "breast_cancer":
        label = np.array(label)[:, 1]
        label2number = {"cancer": 1, "contrl": 2}
    elif data_set == "HP":
        label = np.array(label)[:, 1]
        label2number = {"""disease: Type 2 Diabetic""": 1, """disease: Non-Diabetic""": 2}
    elif data_set == "COVID-19":
        label = np.array(label)[:, 2]
        label2number = {"C-1": 2, "C-2": 2, "C-3": 2, "C-4": 2, "COV-1": 1, "COV-2": 1, "COV-3": 1, "COV-4": 1,
                        "COV-5": 1, "COV-6": 1, "COV-7": 1, "COV-8": 1}
    elif data_set == "COVID-19_cluster":
        label = np.array(label)[:, 4]
        label2number = {"Non-classical Monocytes":1,"CD8 EM":2,"Classical/intermediate Monocytes":3,"Naive CD4/CD8":4,"Memory B Cells":5,
        "NK Cells":6,"CD4 CM":7,"CD4 EM":8,"Naive B Cells":9,"Plasma Cells":10,"T/NK_Cells":11,"CD8 EMRA like":12,"Platelets":13}
    elif data_set == "lung_cancer":
        label = np.array(label)[:, 1]
        label2number = {"AIS":1,"MIA":2,"IAC":3}
    label_u = np.unique(label,return_inverse=True)
    print(label_u)
    for i in range(len(label)):
        x = np.zeros(class_num+1)
        x[label2number[label[i]] - 1] = 1
        data_label.append(x)
    data_label = np.array(data_label)

    data_label_index = np.argmax(data_label,1)
    arr1 = np.array(data)

    X = arr1[:,1:].T
    indicator = np.where(X > 0, 1, 0)
    sum_gene = np.sum(indicator, axis=0).flatten()

    data = X[:, sum_gene > 10]

    var_gene = np.var(data, axis=0)
    index = np.argsort(var_gene)[-gene_num:]
    data = data[:, index]
    # print(data.shape)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.fit_transform(data)
    gene_name = np.array(arr1[index, 0])
    cell_name = np.array(arr1[0, 1:])
    print(data.shape)
    data_array = []
    data_for_count = []
    np.random.seed(2)
    semi_label = np.random.permutation(cell_name.shape[0])
    semi_label_index = int((1 - rate) * cell_name.shape[0])
    semi_label_train = semi_label[:semi_label_index]
    semi_label_real = data_label_index[semi_label_train]
    weight_label_train = semi_label[semi_label_index + 1:]
    if  data_set == "lung_cancer":
        data_label[semi_label_train]=np.array([0,0,0,1])
    elif data_set == "COVID-19_cluster":
        data_label[semi_label_train] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    elif data_set == "AD_cluster":
        data_label[semi_label_train] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    else:
        data_label[semi_label_train] = np.array([0,0,1])
    class_weight = 'balanced'
    weight = compute_class_weight(class_weight, np.array(range(class_num)), data_label_index[weight_label_train])
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(data[i, 0:].astype('double'))
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        figure_size = int(gene_len ** 0.5)
        if figure_size * figure_size == gene_len:
            data_x = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
            figure_size_real = figure_size
        else:
            for j in range((figure_size + 1) ** 2 - gene_len):
                gene_list.append(0)
            data_x = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
            figure_size_real = figure_size + 1
        data_array.append(data_x)
    data_array = np.array(data_array)

    test_data = np.array(data_array)[semi_label_train]
    return data_array, data_label, gene_name, cell_name, figure_size_real,weight,scaler,semi_label_real,test_data

# Revised freom Original version in scVI
# Ref:
# https://github.com/romain-lopez/scVI-reproducibility/blob/master/demo_code/benchmarking.py


def impute_dropout(X, seed=1, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        X_zero = np.copy(X)
        # select non-zero subset
        i, j = np.nonzero(X_zero)
    # If the input is a sparse matrix
    else:
        X_zero = scipy.sparse.lil_matrix.copy(X)
        # select non-zero subset
        i, j = X_zero.nonzero()

    np.random.seed(seed)
    # changes here:
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(
        np.floor(rate * len(i))), replace=False)
    # X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
    X_zero[i[ix], j[ix]] = 0.0

    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

def read_klein_for_train(data_path,label_path,class_num,rate,gene_num,sparsify = False, skip_exprs = False):
    data = pd.read_csv(data_path, header=0, sep=",")
    label = pd.read_csv(label_path, header=0, sep=",")
    label = np.array(label)[:, 1]
    data_label = []
    for i in range(len(label)):
        x = np.zeros(class_num+1)
        x[label[i]] = 1
        data_label.append(x)
    data_label = np.array(data_label)
    data_label_index = np.argmax(data_label,1)
    arr1 = np.array(data)
    X = arr1[:,1:]
    indicator = np.where(X > 0, 1, 0)
    sum_gene = np.sum(indicator, axis=0).flatten()

    data = X[:, sum_gene > 10]

    var_gene = np.var(data, axis=0)
    index = np.argsort(var_gene)[-gene_num:]
    data = data[:, index]
    # print(data.shape)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.fit_transform(data)

    gene_name = np.array(arr1[0, index])
    cell_name = np.array(arr1[:, 0])

    data_array = []
    data_for_count = []
    np.random.seed(1)
    semi_label = np.random.permutation(cell_name.shape[0])
    semi_label_index = int((1 - rate) * cell_name.shape[0])
    semi_label_train = semi_label[:semi_label_index]
    semi_label_real = data_label_index[semi_label_train]
    weight_label_train = semi_label[semi_label_index + 1:]
    data_label[semi_label_train]=np.array([0,0,0,0,1])
    class_weight = 'balanced'
    weight = compute_class_weight(class_weight, np.array(range(class_num)), data_label_index[weight_label_train])
    cell_num = np.shape(cell_name)[0]
    for i in range(cell_num):
        gene_list_for_count = np.array(data[i, 0:].astype('double'))
        gene_list = gene_list_for_count.tolist()
        gene_len = len(gene_list)
        figure_size = int(gene_len ** 0.5)
        if figure_size * figure_size == gene_len:
            data_x = np.array(gene_list).reshape(figure_size, figure_size, 1).astype('double')
            figure_size_real = figure_size
        else:
            for j in range((figure_size + 1) ** 2 - gene_len):
                gene_list.append(0)
            data_x = np.array(gene_list).reshape(figure_size + 1, figure_size + 1, 1).astype('double')
            figure_size_real = figure_size + 1
        data_array.append(data_x)
    data_array = np.array(data_array)
    test_data = np.array(data_array)[semi_label_train]
    return data_array, data_label, gene_name, cell_name, figure_size_real,weight,scaler,semi_label_real,test_data

# IMPUTATION METRICS
# Revised freom Original version in scVI
# Ref:
# https://github.com/romain-lopez/scVI-reproducibility/blob/master/demo_code/benchmarking.py


def imputation_error(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        result = np.abs(x - y)
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(x - yuse)
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)

def imputation_cosine(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    cosine similarity between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        print(x)
        print(y)
        result = cosine_similarity(x, y)
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        x = x.reshape(1, -1)
        yuse = yuse.reshape(1, -1)
        result = cosine_similarity(x, yuse)
    # return np.median(np.abs(x - yuse))
    return result[0][0]
def imputation_cosine_log(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    cosine similarity between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        result = cosine_similarity(x, np.log(y+1))
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        x = x.reshape(1, -1)
        yuse = yuse.reshape(1, -1)
        result = cosine_similarity(x, np.log(yuse+1))
    # return np.median(np.abs(x - yuse))
    return result[0][0]
def imputation_error_log(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        result = np.abs(x - np.log(y+1))
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(x - np.log(yuse+1))
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)
def imputation_impute_matrix(imputed_file,original_file,rate):
    data_imputed = pd.read_csv(imputed_file, header=None, sep=",")
    data_original = pd.read_csv(original_file, header=None, sep=",")
    X_imputed = np.array(data_imputed)[1:, 1:].T
    X_original = np.array(data_original)[1:, 1:].T
    # count_for_imputed = torch.tensor(X_imputed)
    # count_for_imputed_ = F.normalize(count_for_imputed, dim=-1, p=2).detach().cpu().numpy()
    # count_for_original = torch.tensor(X_original)
    # count_for_original_ = F.normalize(count_for_original, dim=-1, p=2).detach().cpu().numpy()
    X_zero, i, j, ix = impute_dropout(X_original,1,rate)
    mean,median,min,max = imputation_error(X_imputed,X_original,X_zero,i,j,ix)
    cosine_sim = imputation_cosine(X_imputed,X_original,X_zero,i,j,ix)
    print(median)
    print(cosine_sim)
# def read_interpretable_for_train_T2D_imputed(data_path,class_num,type,sparsify = False, skip_exprs = False):
#     data = pd.read_table(data_path, header=None, sep="\t")
#     arr1 = np.array(data)
#     if(type == 1):
#         gene_name = np.array(arr1[7:, 0])
#         cell_name = np.array(arr1[6, 1:])
#         label = np.array(arr1[1, 1:])
#     else:
#         gene_name = np.array(arr1[1:, 0])
#         cell_name = np.array(arr1[0, 1:])
#     label2number = {"Control":1,"T2D":2}
#     data_array = []
#     data_for_count = []
#     cell_num = np.shape(cell_name)[0]
#     for i in range(cell_num):
#         gene_list_for_count = np.array(arr1[7:, i + 1].astype('double'))
#         gene_list_max = np.max(gene_list_for_count)
#         data_for_count.append(gene_list_max)
#         gene_list_for_count = gene_list_for_count / gene_list_max
#         gene_list = gene_list_for_count.tolist()
#         gene_len = len(gene_list)
#         data_array.append(np.array(gene_list))
#     data_array = np.array(data_array)
#     data_label = []
#     for i in range(len(label)):
#         x = np.zeros(class_num)
#         x[label2number[label[i]] - 1] = 1
#         data_label.append(x)
#     data_label = np.array(data_label)
#     return data_array, data_label, gene_name, cell_name, data_for_count

if __name__=="__main__":
    #read_cell_to_image("./test_csv/splatter_exprSet_test.csv","./test_csv/splatter_exprSet_test_label.csv",4)
    #read_cell_for_h5_to_csv("./compare_funtion/sagan/test_csv/Romanov/data.h5",0.5,2500,"./compare_funtion/sagan/test_csv/Romanov/Romanov.csv")
    # print(semi_label_real.shape)
    # print(test_data.shape)
    # a = pd.read_csv("./compare_funtion/AFGRL/AFGRL-master/test_csv/Alzheimer/GSE138852_counts.csv")
    # print(a.shape)
    data_set = "Quake_10x_Spleen"
    function_list = ["raw","AutoClass","GraphSCI","MAGIC","scTAG","scGCL"]
    res_list = []
    gene_list = "./compare_funtion/AFGRL/AFGRL-master/results/"+data_set+"/"+data_set+"-gene_list.csv"

    for i in range(len(function_list)):
        result_file = "./compare_funtion/AFGRL/AFGRL-master/results/"+data_set+"/"+data_set+"-res.txt"
        adata,cell_type,_= create_adata("./compare_funtion/AFGRL/AFGRL-master/results/"+data_set + "/"+function_list[i]+"-"+data_set+"-imputed.csv","./compare_funtion/AFGRL/AFGRL-master/test_csv/Quake_10x_Spleen/data.h5",function_list[i],result_file)
        rcParams['figure.figsize'] = 12,8

        sc.pp.neighbors(adata)
        dp_object = sc.tl.umap(adata)
        axes_dict = dp_object.get_axes()
        # figure out which axis you want by printing the axes_dict
        # print(axes_dict)
        ax = axes_dict[...]
        ax.xaxis.label.set_fontsize(15)
        for label in ax.get_xticklabels():
            label.set_fontsize('large')
        # 如果设置了 adata 的 .raw 属性时，下图显示了“raw”（标准化、对数化但未校正）基因表达矩阵。
        sc.pl.umap(adata,ncols = 2, color=[function_list[i]],save = "./"+data_set+"/"+function_list[i]+".png")

        # sc.tl.louvain(adata)
        # sc.tl.paga(adata,groups=function_list[i])
        # sc.pl.paga(adata,threshold=0.03,save = "/Klein/"+function_list[i]+"_paga.png")
    #______________imputation_matrix_______________
    # data_set = "Klein"
    # imputed_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/scAFGRL-Klein-imputed.csv"
    # original_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/raw-Klein-imputed.csv"
    # imputation_impute_matrix(imputed_file, original_file, 0.5)
    # data_set = "Klein"
    # imputed_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/imputation_eval/dropout_0.5/MAGIC-Klein-imputed-0.5.csv"
    # original_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/raw-Klein-imputed.csv"
    # imputation_impute_matrix(imputed_file, original_file, 0.5)
    # data_set = "Zeisel"
    # imputed_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/imputation_eval/dropout_0.1/scAFGRL-Zeisel-imputed-0.1.csv"
    # original_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/raw-Zeisel-imputed.csv"
    # imputation_impute_matrix(imputed_file, original_file, 0.1)
    # data_set = "Zeisel"
    # imputed_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/imputation_eval/dropout_0.3/scAFGRL-Zeisel-imputed-0.3.csv"
    # original_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/raw-Zeisel-imputed.csv"
    # imputation_impute_matrix(imputed_file, original_file, 0.3)
    # data_set = "Zeisel"
    # imputed_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/imputation_eval/dropout_0.5/scAFGRL-Zeisel-imputed-0.5.csv"
    # original_file = "./compare_funtion/AFGRL/AFGRL-master/results/" + data_set + "/raw-Zeisel-imputed.csv"
    # imputation_impute_matrix(imputed_file, original_file, 0.5)
        #sc.tl.rank_genes_groups(adata, function_list[i], method='wilcoxon')
        # marker_genes = sc.pl.rank_genes_groups(adata, sharey = False)
        # adata.write(result_file)
        # sc.pl.matrixplot(adata,["ACAP3","CPSF3L","NADK","CLSTN1","SLC2A5","PADI2","CAMK2N1","ECE1","C1QA","ID3","RPL11","SRRM1"],function_list[i],dendrogram=True,cmap='Blues',standard_scale='var',colorbar_title='column scaled\nexpression',save="/"+data_set+"/"+function_list[i]+"_gene_expression.png")
        # marker_genes = ["SCN2B","C1QTNF3-AMACR","CH25H","GPR85","WRNIP1","SLC39A7","CBX6","LRFN5","ADAM10","AC004448.5","GAP43","TRIM52","RTRF","GPNMB","PLCB4","AMOTL2","SPESP1","CALN1","KCNH7","SCAI","ASTE1","DAB2IP","FCGBP","EEF2","ZDHHC5","UAP1L1","MAST1","FAM131B","KIRREL3","CNR1","WASH1","VWA1","LRRK1","IKZF2","SPHK2","ZNF358","KCNH3","RG55","BCAR1","C4orf48"
        #                 "DPYSL4","CAPN3",""]
        # marker_genes = ["CPSF3L","ACAP3","RPL11","TXLNA","C1orf123","DHCR24","NADK","LDLRAP1","SH3BGRL3","ZDHHC18","LINC01358","CHD5"
        #                 ,"PLOD1","ADGRB2","MAP7D1","TMEM125","FUBP1","SLC22A15","PADI2","C1QA"]


        # print(marker_genes)
        # f = open('./result_txt/result_' + str(data_set) + '_' + str(function_list[i]) + '_genes.txt', mode='w')
        # f.write("id" + "\n")
        # for j in range(160):
        #     f.write(str(marker_genes[j]) + "\n")
        # f.close()
        # ax = sc.pl.rank_genes_groups_tracksplot(adata, groupby=function_list[i],save="/"+data_set+"/"+function_list[i]+"_gene_expression.png");
    #data_array, data_label, gene_name, cell_name, figure_size_real,weight,scaler,semi_label_real,test_data= read_interpretable_for_train("./compare_funtion/sagan/test_csv/lung_cancer/lung_cancer.csv","./compare_funtion/sagan/test_csv/lung_cancer/lung_label.csv",2,"lung_cancer",0.5,4900)
    #data_array, data_label, gene_name, cell_name, figure_size_real, weight, scaler, semi_label_real, test_data =read_klein_for_train("./compare_funtion/sagan/test_csv/Klein/data.csv","./compare_funtion/sagan/test_csv/Klein/label.csv",4,1,4900)
    # data_org = data_array.reshape(data_array.shape[0], 4900)
    # a = pd.DataFrame(data_org).T
    # a.to_csv("./compare_funtion/sagan/results/Klein_raw.csv")
    #print(data_array.shape)
    #data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = read_interpretable_for_train(
       # "./compare_funtion/sagan/test_csv/HP_interpretable/HP.csv",
       # "./compare_funtion/sagan/test_csv/HP_interpretable/HP_label.csv", 2, "HP")
   #  data_array, data_label, gene_name, cell_name, data_for_count = read_cell_for_interpretable_imputed(
   #      "./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_counts.csv",
   #      "./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_covariates.csv", 2, "AD")
   #  data_array, data_label, gene_name, cell_name, data_for_count, figure_size_real = read_interpretable_for_train_COVID19(
   #      "./compare_funtion/sagan/test_csv/COVID-19_interpretable/GSE164948_covid_control_RNA_counts.csv",
   #      "./compare_funtion/sagan/test_csv/COVID-19_interpretable/GSE164948_covid_control_count_metadata.csv", 2)

    # data_array, data_label, gene_name, cell_name, data_for_count,figure_size = read_interpretable_for_train_T2D("./compare_funtion/sagan/test_csv/t2d_interpretable/T2D_raw.tsv",2)
    # print(cell_name.shape)
    # print(data_array.shape)
    # print(gene_name.shape)