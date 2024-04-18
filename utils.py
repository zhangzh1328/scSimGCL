import os
import torch
import random
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import *
from scipy.optimize import linear_sum_assignment as linear_assignment
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def loader_construction(data_path):
    data = sc.read_h5ad(data_path)
    X_all = data.X
    y_all = data.obs.values[:,0]
    input_dim = X_all.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    train_set = CellDataset(X_train, y_train)
    test_set = CellDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=0)
    return train_loader, test_loader, input_dim


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp

