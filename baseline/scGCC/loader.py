from anndata._core.anndata import AnnData
from torch.utils.data import Dataset
import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import warnings


class Matrix(Dataset):
    def __init__(self,
                 adata: AnnData = None,
                 global_graph: Data = None,
                 obs_label_colname: str = "x",
                 augmentation: bool = False,
                 args_augmentation: dict = {}
                 ):

        super().__init__()

        X_all = adata.X
        X_all[np.isnan(X_all)] = 0
        adata.X = X_all
        self.adata = adata
        
        # data
        # scipy.sparse.csr.csr_matrix or numpy.ndarray
        if isinstance(self.adata.X, np.ndarray):
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()
        # label (if exist, build the label encoder)
        #if self.adata.obs.get(obs_label_colname) is not None:
        self.label = self.adata.obs.values[:,0]
        self.unique_label = list(set(self.label))
        # 标签编码 str:number
        self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
        # 标签解码 number:str
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        #else:
        #    self.label = None

        # do the augmentation
        self.augmentation = augmentation
        self.num_cells, self.num_genes = self.adata.shape
        self.args_augmentation = args_augmentation
        # 深拷贝数据
        self.data_for_augmentation = deepcopy(self.data)
        self.global_graph = global_graph

    def RandomAugmentation(self, sample, index):
        # find neighbor for augmentation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #print(self.global_graph.dtype)
            #print(.dtype)
            #print(torch.Tensor([index]).dtype)
            for neighbor in NeighborLoader(self.global_graph, num_neighbors=[self.data.shape[0]],
                                           input_nodes=torch.Tensor([index]).long()):
                neighbor_idx = neighbor.n_id.numpy()  # tensor transform to numpy

        tr = transformation(self.data_for_augmentation, sample, neighbor_idx)
        # Mask
        tr.random_mask(self.args_augmentation['mask_percentage'], self.args_augmentation['mask_prob'])

        # Gaussian noise
        tr.random_gaussian_noise(self.args_augmentation['noise_percentage'], self.args_augmentation['sigma'],
                                 self.args_augmentation['noise_prob'])
        # inner exchange
        tr.random_exchange(self.args_augmentation['swap_percentage'], self.args_augmentation['swap_prob'])

        # exchange with one neighbor
        tr.one_neighbor_crossover(self.args_augmentation['exchange_percentage'],
                                  self.args_augmentation['exchange_prob'])

        # exchange with many neighbors
        tr.some_neighbors_crossover(self.args_augmentation['cross_percentage'],
                                    self.args_augmentation['cross_prob'])
        tr.ToTensor()
        return tr.cell_profile

    # a cell
    def __getitem__(self, index):

        sample = self.data[index]

        if self.label is not None:
            label = self.label_encoder[self.label[index]]
        else:
            label = -1
        # do augmentation for a cell
        if self.augmentation:
            sample_1 = self.RandomAugmentation(sample, index)
            sample = [sample, sample_1]
        return sample, index, label

    def __len__(self):
        return self.adata.X.shape[0]


class transformation():
    def __init__(self,
                 dataset,  # row*column
                 cell_profile,  # 1*column
                 neighbor_idx):

        self.dataset = dataset
        self.cell_profile = deepcopy(cell_profile)
        self.gene_num = len(self.cell_profile)
        self.cell_num = len(self.dataset)
        self.neighbor_idx = neighbor_idx

    def build_mask(self, masked_percentage: float):
        # 根据基因的数量（对列）构建mask
        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool),
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        np.random.shuffle(mask)
        return mask

    # add dropout events
    def random_mask(self,
                    mask_percentage: float = 0.2,
                    apply_mask_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_mask_prob:
            # create the mask
            mask = self.build_mask(mask_percentage)
            # do the mutation
            self.cell_profile[mask] = 0

    # add gaussian noise
    def random_gaussian_noise(self,
                              noise_percentage: float = 0.7,
                              sigma: float = 0.2,
                              apply_noise_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_noise_prob:
            # create the mask
            mask = self.build_mask(noise_percentage)
            # create the noise
            noise = np.random.normal(0, sigma, int(self.gene_num * noise_percentage))
            # do the mutation
            self.cell_profile[mask] += noise

    # Randomly exchange gene expression values within single cell
    def random_exchange(self,
                    swap_percentage: float = 0.2,
                    apply_swap_prob: float = 0.5):
        s = np.random.uniform(0, 1)
        if s < apply_swap_prob:
            # create the number of pairs for swapping 
            swap_instances = int(self.gene_num * swap_percentage / 2)
            swap_pair = np.random.randint(self.gene_num, size=(swap_instances, 2))
            # do the mutation
            self.cell_profile[swap_pair[:, 0]], self.cell_profile[swap_pair[:, 1]] = \
                self.cell_profile[swap_pair[:, 1]], self.cell_profile[swap_pair[:, 0]]

    # Randomly exchange gene expression values with a randomly selected neighboring cell
    def one_neighbor_crossover(self,
                               cross_percentage: float = 0.3,
                               apply_cross_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_cross_prob:
            # choose one neighbor for crossover
            list_idx = np.random.randint(1, len(self.neighbor_idx))
            cross_idx = self.neighbor_idx[list_idx]
            cross_instance = self.dataset[cross_idx]
            # build the mask
            mask = self.build_mask(cross_percentage)
            # do the mutation
            tmp = cross_instance[mask].copy()
            cross_instance[mask], self.cell_profile[mask] = self.cell_profile[mask], tmp

    # Randomly exchange gene expression values with some neighboring cells
    def some_neighbors_crossover(self,
                                 change_percentage: float = 0.3,
                                 apply_mutation_prob: float = 0.5):

        s = np.random.uniform(0, 1)
        if s < apply_mutation_prob:
            mask = self.build_mask(change_percentage)
            chosen = self.dataset[self.neighbor_idx[1:], :][:, mask]
            mutations = np.apply_along_axis(random_substitution, axis=0, arr=chosen)
            self.cell_profile[mask] = mutations[0]

    def ToTensor(self):
        self.cell_profile = torch.from_numpy(self.cell_profile)


def random_substitution(x):
    random_cell = np.random.randint(x.shape)
    return x[random_cell]
