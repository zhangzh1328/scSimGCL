import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz
from torch_geometric.data import Data
import data_Preprocess
import scipy.sparse as sp
import pandas as pd
from torch_geometric.utils import remove_self_loops, to_undirected
from graph_funtion import *

class Singlecell(InMemoryDataset):

    def __init__(self, root, name, filepath, transform=None, pre_transform=None):
        self.name = name.lower()
        self.filepath = filepath
        self.labelpath = "./test_csv/Alzheimer/GSE138852_covariates.csv"
        # self.labelpath = "./test_csv/Zeisel/label.csv"
        super(Singlecell, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self):
        return 'amazon_electronics_{}.npz'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        raw = False
        # data, data_label,size_factor = data_Preprocess.nomalize_for_COVID(self.filepath,self.labelpath,2048);
        # data, data_label, size_factor,gene = data_Preprocess.nomalize_for_AD(self.filepath, self.labelpath, 2048);
        data, data_label = data_Preprocess.nomalize_for_AF(self.filepath, 2048,raw);

        x = torch.tensor(np.array(data),dtype=torch.float32)
        y = torch.tensor(data_label, dtype=torch.long)
        #edge_index =  np.corrcoef(data)
        #print(edge_index.shape)
        #edge_index = np.where(edge_index>0.6,1,0)
        adj, adj_n = get_adj(data)
        adj = sp.coo_matrix(adj_n)
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, x.size(0))  # Internal coalesce.
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        print(self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())