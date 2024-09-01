# -*- coding: utf-8 -*-
# @Author  : sw t
import argparse
import math
import os
import random
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import loader
import layer

from sklearn.cluster import KMeans

import scanpy as sc
import pandas as pd
import umap
import hdbscan

from metrics import compute_metrics

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch scRNA-seq scGCC Training')

# 1.input h5ad data
parser.add_argument('--input_h5ad_path', type=str, default="",
                    help='path to input h5ad file')

parser.add_argument('--label_col_name', type=str, default="x",
                    help='the label name')
# 2.hyper-parameters
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of loading workers')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs')

parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N',
                    help='minibatch size')

parser.add_argument('--lr', '--learning_rate', default=1, type=float, metavar='LR')

parser.add_argument('--low_dim', default=256, type=int,
                    help='feature dimension (default: 256)')

parser.add_argument('--queue_size', default=1024, type=int,
                    help='queue size')

parser.add_argument('--moco_m', default=0.99, type=float,
                    help='moco momentum of updating key encoder')

parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')

# augmentation prob
parser.add_argument("--aug_prob", type=float, default=0.5,
                    help="The prob of doing augmentation")

# cluster
parser.add_argument('--cluster_name', default='hdbscan', type=str,
                    help='name of clustering method')

parser.add_argument('--num_cluster', default=-1, type=int,
                    help='number of clusters')

# random
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

# gpu
parser.add_argument('--gpu', default=0, type=int)  # None

# logs and savings
parser.add_argument('--eval_freq', default=10, type=int,
                    metavar='N', help='Save frequency')

parser.add_argument('--log_freq', default=10, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--save_dir', default='./result', type=str,
                    help='result saving directory')
# graph
parser.add_argument('--head', default=20, type=int, help='Number of heads')
# similarity measures
parser.add_argument('--graph_type', choices=['KNN', 'PKNN'], default='KNN')
# KNN
parser.add_argument('--k', type=int, help='Number of neighbors for graph', default=10)
# set threshold value
parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0,
                    help='Number of standard deviations to add to the mean of distances/correlation values.')
# graph path
parser.add_argument('--graph_path', help='load graph from the path')
# save graph
parser.add_argument('--save_graph', action='store_true', default=True, help='save graph to the path of save_dir')
# mlp
parser.add_argument('--mlp', action="store_true", default=False)


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    main_part(args)


def main_part(args):
    print(args)

    # 1. Build Dataloader
    # Load h5ad data
    input_h5ad_path = args.input_h5ad_path
    processed_adata = sc.read_h5ad(input_h5ad_path)
    processed_adata.X[np.isnan(processed_adata.X)] = 0
    
    label_col_name = args.label_col_name
    print('----------------------------data attr------------------------------')
    print(processed_adata)
    print('-------------------------------------------------------------------')
    # find dataset name
    pre_path, filename = os.path.split(input_h5ad_path)
    dataset_name, ext = os.path.splitext(filename)
    # save path
    save_path = os.path.join(args.save_dir, "scGCC")
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    # create global graph
    if not args.graph_path:
        edgelist = prepare_graphs(processed_adata, dataset_name, save_path, args)
    else:
        edgelist = load_graph(args.graph_path)
    # number of cells
    num_nodes = processed_adata.shape[0]
    print(f'Number of nodes in graph: {num_nodes}.')
    print(f'The graph has {len(edgelist)} edges.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    print(f'The undirected graph has {edge_index.shape[1]}.')
    global_graph = Data(x=processed_adata.X, edge_index=edge_index)
    # numbering the node
    global_graph.n_id = torch.arange(global_graph.num_nodes)

    # Define data augmentations
    args_augmentation = {

        # mask
        'mask_percentage': 0.2,
        'mask_prob': args.aug_prob,

        # gaussian noise
        'noise_percentage': 0.7,
        'sigma': 0.2,
        'noise_prob': args.aug_prob,

        # inner exchange
        'swap_percentage': 0.2,
        'swap_prob': args.aug_prob,

        # exchange with one neighbor
        'exchange_percentage': 0.30,
        'exchange_prob': args.aug_prob,

        # exchange with many neighbors
        'cross_percentage': 0.30,
        'cross_prob': args.aug_prob
    }
    # create dataset loader
    train_dataset = loader.Matrix(
        adata=processed_adata,
        global_graph=global_graph,
        obs_label_colname=label_col_name,
        augmentation=True,
        args_augmentation=args_augmentation
    )
    eval_dataset = loader.Matrix(
        adata=processed_adata,
        obs_label_colname=label_col_name,
        augmentation=False
    )
    # 细胞数量小于批次大小
    if train_dataset.num_cells < args.batch_size:
        args.batch_size = train_dataset.num_cells
        args.queue_size = train_dataset.num_cells

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=num_nodes, shuffle=False,
        sampler=None, num_workers=args.workers, pin_memory=True)

    # 2. Create Model
    print("creating model 'GAT'")
    model = layer.MoCo(
        layer.GATEncoder,
        int(train_dataset.num_genes),
        args.low_dim, args.queue_size, args.moco_m, args.temperature, args.head, args.mlp)
    print(model)

    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9, weight_decay=1e-4)

    # 2. Train Encoder
    best_ari, best_eval_supervised_metrics, best_pd_labels, best_embeddings = -1, None, None, None
    # train the model
    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_unsupervised_metrics = train(train_loader, global_graph, model, criterion, optimizer, epoch, args)
        # training log & unsupervised metrics
        if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
            if epoch == 0:
                with open(os.path.join(save_path, 'log_scGCC_{}.txt'.format(dataset_name)), "w") as f:
                    f.writelines(f"epoch\t" + '\t'.join((str(key) for key in train_unsupervised_metrics.keys())) + "\n")
                    f.writelines(f"{epoch}\t" + '\t'.join(
                        (str(train_unsupervised_metrics[key]) for key in train_unsupervised_metrics.keys())) + "\n")
            else:
                with open(os.path.join(save_path, 'log_scGCC_{}.txt'.format(dataset_name)), "a") as f:
                    f.writelines(f"{epoch}\t" + '\t'.join(
                        (str(train_unsupervised_metrics[key]) for key in train_unsupervised_metrics.keys())) + "\n")

        # inference log & supervised metrics
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            # eval和train的数据一致，这里返回编码后的embedding和groundtruth
            embeddings, gt_labels = inference(eval_loader, model, global_graph)

            # if gt_label exists and metric can be computed
            if train_dataset.label is not None:
                num_cluster = len(train_dataset.unique_label) if args.num_cluster == -1 else args.num_cluster

                # multiple random experiments
                for random in range(1):
                    eval_supervised_metrics, pd_labels = cluster(embeddings, gt_labels, num_cluster, args)
                    # compute metrics
                    if eval_supervised_metrics["ARI"] > best_ari:
                        best_ari = eval_supervised_metrics["ARI"]
                        best_eval_supervised_metrics = eval_supervised_metrics
                        best_pd_labels = pd_labels
                        best_embeddings = embeddings
                print("Epoch: {}\t {}\n".format(epoch, eval_supervised_metrics))

                with open(os.path.join(save_path, 'log_scGCC_{}.txt'.format(dataset_name)), "a") as f:
                    f.writelines("{}\teval\t{}\n".format(epoch, eval_supervised_metrics))
            else:
                if args.num_cluster > 0:
                    num_cluster = args.num_cluster
                    print("cluster num is set to {}".format(num_cluster))
                    best_pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embeddings).labels_
                else:
                    best_pd_labels = None

    # 3. Final Savings
    # save feature & labels
    np.savetxt(os.path.join(save_path, "feature_scGCC_{}.csv".format(dataset_name)), best_embeddings, delimiter=',')

    if best_pd_labels is not None:
        pd_labels_df = pd.DataFrame(best_pd_labels, columns=['pd_labels'])
        pd_labels_df.to_csv(os.path.join(save_path, "pd_label_scGCC_{}.csv".format(dataset_name)))

    if train_dataset.label is not None:
        label_decoded = [train_dataset.label_decoder[i] for i in gt_labels]
        save_labels_df = pd.DataFrame(label_decoded, columns=['gt_labels'])
        save_labels_df.to_csv(os.path.join(save_path, "gt_label_scGCC_{}.csv".format(dataset_name)))

        if best_pd_labels is not None:
            # write metrics into txt
            best_metrics = best_eval_supervised_metrics
            txt_path = os.path.join(save_path, "metric_scGCC.txt")
            f = open(txt_path, "a")
            record_string = dataset_name
            for key in best_metrics.keys():
                record_string += " {}".format(best_metrics[key])
            record_string += "\n"
            f.write(record_string)
            f.close()


# train(train_loader, model, criterion, optimizer, epoch, args)
def train(train_loader, global_graph, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # build part edge of global graph
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subgraph = global_graph.subgraph(torch.LongTensor(index))
            subgraph.num_nodes = len(index)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        # find the edge index of subgraph
        subgraph_edge_index = subgraph.edge_index.to(torch.long).cuda(args.gpu, non_blocking=True)
        # compute output
        # return logits, labels
        output, target = model(im_q=images[0], im_k=images[1], edge_index=subgraph_edge_index)

        # InfoNCE loss 这里的target都是0 output是logit：N*(1+r)
        loss = criterion(output, target)

        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0]
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i + 1)

    unsupervised_metrics = {"accuracy": acc_inst.avg.item(), "loss": losses.avg}

    return unsupervised_metrics


def inference(eval_loader, model, global_graph):
    print('Inference...')
    model.eval()
    features = []
    labels = []

    for i, (images, index, label) in enumerate(eval_loader):
        images = images.cuda()
        with torch.no_grad():
            feat = model(images, edge_index=global_graph.edge_index.cuda(), is_eval=True)
        feat_pred = feat.data.cpu().numpy()
        label_true = label
        features.append(feat_pred)
        labels.append(label_true)
    # concatenate
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


#  create graphs
def prepare_graphs(adata_khvg, dataset_name, save_path, args):
    if args.graph_type == 'KNN':
        print('Computing KNN graph by scanpy...')
        # use package scanpy to compute knn graph
        distances_csr_matrix = \
            sc.pp.neighbors(adata_khvg, n_neighbors=args.k + 1, knn=True, copy=True).obsp[
                'distances']
        # ndarray
        distances = distances_csr_matrix.A
        # resize
        neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], args.k))

    elif args.graph_type == 'PKNN':
        print('Computing PKNN graph...')
        if isinstance(adata_khvg.X, np.ndarray):
            X_khvg = adata_khvg.X
        else:
            X_khvg = adata_khvg.X.toarray()
        distances, neighbors = correlation(data_numpy=X_khvg, k=args.k + 1)

    if args.graph_distance_cutoff_num_stds:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(args.graph_distance_cutoff_num_stds) * np.std(
            np.nonzero(distances), axis=None)
    # shape: 2 * (the number of edge)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if args.graph_distance_cutoff_num_stds:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    # save
    if args.save_graph:
        num_hvg = adata_khvg.shape[1]
        k_file = args.k
        if args.graph_type == 'KNN':
            graph_name = 'Scanpy'
        elif args.graph_type == 'PKNN':
            graph_name = 'Pearson'

        filename = f'{dataset_name}_{graph_name}_KNN_K{k_file}_gHVG_{num_hvg}.txt'

        final_path = os.path.join(save_path, filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


# PKNN
def correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors


# load graph
def load_graph(edge_path):
    edgelist = []
    with open(edge_path, 'r') as edge_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edge_file.readlines()]
    return edgelist


# cluster
def cluster(embedding, gt_labels, num_cluster, args):
    # kmeans
    if args.cluster_name == 'kmeans':
        print("cluster num is set to {}".format(num_cluster))
        pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embedding).labels_
        # compute metrics
        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
        return eval_supervised_metrics, pd_labels

    # hdbscan
    if args.cluster_name == 'hdbscan':
        umap_reducer = umap.UMAP()
        u = umap_reducer.fit_transform(embedding)
        cl_sizes = [10, 25, 50, 100]
        min_samples = [5, 10, 25, 50]
        hdbscan_dict = {}
        ari_dict = {}
        for cl_size in cl_sizes:
            for min_sample in min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                clusterer.fit(u)
                ari_dict[(cl_size, min_sample)] = compute_metrics(gt_labels, clusterer.labels_)
                hdbscan_dict[(cl_size, min_sample)] = clusterer.labels_
        max_tuple = max(ari_dict, key=lambda x: ari_dict[x]['ARI'])
        return ari_dict[max_tuple], hdbscan_dict[max_tuple]


if __name__ == '__main__':
    args = parser.parse_args()
    main()
