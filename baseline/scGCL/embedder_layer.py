import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from models.logreg import LogisticRegression
from utils import printConfig
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import utils
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
import math
random.seed(0)
import data_Preprocess
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, pairwise
# scGCL Model
# Revised freom Original version in AFGRL
# Ref:
# https://github.com/Namkyeong/AFGRL/tree/master/embedder.py

class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self, epoch):
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None

        for bc, batch_data in enumerate(self._loader):
            # augmentation = utils.Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
            #                                   float(self._args.aug_params[2]), float(self._args.aug_params[3]))

            batch_data.to(self._device)
            # view1, view2 = augmentation._feature_masking(batch_data, self._device)

            emb, loss = self._model(x = batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                                                           neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                                                           edge_weight=batch_data.edge_attr, epoch=epoch)
            # emb, loss = self._model(x=view1.x, x2=view2.x, y=batch_data.y, edge_index=view1.edge_index,
            #                         edge_index_2=view2.edge_index,
            #                         neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
            #                         edge_weight=view1.edge_attr, edge_weight_2=view2.edge_attr, epoch=epoch)
            emb = emb.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])


    def evaluate(self, task, epoch, sillog):
        if task == "node":
            self.evaluate_node(epoch)
        elif task == "clustering":
            self.evaluate_clustering(epoch,sillog)
        elif task == "similarity":
            self.run_similarity_search(epoch)
        

    def evaluate_node(self, epoch):

        # print()
        # print("Evaluating ...")
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):

            self._train_mask = self._dataset[0].train_mask[i]
            self._dev_mask = self._dataset[0].val_mask[i]
            if self._args.dataset == "wikics":
                self._test_mask = self._dataset[0].test_mask
            else:
                self._test_mask = self._dataset[0].test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch, dev_acc, dev_std, test_acc, test_std))

        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_test_acc = test_acc
            self.best_dev_std = dev_std
            self.best_test_std = test_std
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
            self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)
        print(self.st_best)


    def evaluate_clustering(self, epoch,sillog):
        
        embeddings = F.normalize(self._embeddings, dim = -1, p = 2).detach().cpu().numpy()

        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()

        estimator = KMeans(n_clusters = nb_class)

        NMI_list = []

        for i in range(10):
            estimator.fit(embeddings)
            y_pred = estimator.predict(embeddings)
            s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
            NMI_list.append(s1)
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)
        silhid = metrics.silhouette_score(self._embeddings.detach().cpu().numpy(), y_pred, metric='euclidean')
        s1 = sum(NMI_list) / len(NMI_list)
        sillog.append(silhid)
        arr_sil = np.array(sillog)
        ##recover ability
        # data_original = pd.read_csv("./results/Klein/raw-Klein-imputed.csv", header=None, sep=",")
        # X_original = np.array(data_original)[1:, 1:].T
        # count_for_original = torch.tensor(X_original)
        # count_for_original_ = F.normalize(count_for_original, dim=-1, p=2).detach().cpu().numpy()
        # X_zero, i, j, ix = data_Preprocess.impute_dropout(X_original, 1, 0.5)
        # mean, median, min, max = data_Preprocess.imputation_error(embeddings, count_for_original_, X_zero, i, j, ix)
        # cosine_sim = data_Preprocess.imputation_cosine(self._embeddings.detach().cpu().numpy(), X_original, X_zero, i, j, ix)
        #print('** [{}] [Current Epoch {}] recover gene expression: {:.4f} **'.format(self.args.embedder, epoch, cosine_sim))
        print('** [{}] [Current Epoch {}] this epoch NMI values: {:.4f} ** and this epoch sil values: {}'.format(self.args.embedder, epoch, s1,silhid))
        # if s1 > self.best_dev_acc:
        #     self.best_epoch = epoch
        #     self.best_dev_acc = s1
        #     print("~~~~~~~~~~~~~~~~~~")
        #     print(self.best_dev_acc)
        #     if self._args.checkpoint_dir is not '':
        #         print('Saving checkpoint...')
        #         torch.save(embeddings, os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
        #         # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
        #         a = pd.DataFrame(self._embeddings.detach().cpu().numpy()).T
        #         a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
        #     print("save")
        #     print("~~~~~~~~~~~~~~~~~~")
        if math.floor(silhid*100) >= math.floor(self.best_dev_acc*100):

            self.best_dev_acc = round(silhid, 2)
            self.best_embeddings = embeddings
            self.best_test_acc = s1
            print("~~~~~~~~~~~~~~~~~~")
            print(silhid)
            if self._args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                torch.save(embeddings, os.path.join(self._args.checkpoint_dir,
                                                    'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
                # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
                a = pd.DataFrame(self.best_embeddings).T
                a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
            print("save")
            print("~~~~~~~~~~~~~~~~~~")
        # if abs(self.current_loss - self.last_loss) < 1e3:
        #     if self._args.checkpoint_dir is not '':
        #         print('Saving checkpoint...')
        #         torch.save(embeddings, os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
        #         # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
        #         a = pd.DataFrame(self._embeddings.detach().cpu().numpy()).T
        #         a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
        #         self.st_best = '** Finally NMI: {:.4f} **\n'.format(s1)
        #         print(self.st_best)
        #         return True
        #
        # self.last_loss = self.current_loss



    def run_similarity_search(self, epoch):

        test_embs = self._embeddings.detach().cpu().numpy()
        test_lbls = self._dataset[0].y.detach().cpu().numpy()
        numRows = test_embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
        st = []
        for N in [5, 10]:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(test_lbls, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(test_lbls, N).reshape(numRows,N)
            st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

        print("** [{}] [Current Epoch {}] sim@5 : {} | sim@10 : {} **".format(self.args.embedder, epoch, st[0], st[1]))

        if st[0] > self.best_dev_acc:
            self.best_dev_acc = st[0]
            self.best_test_acc = st[1]
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best @5 : {} | Best @10: {} **\n'.format(self.best_epoch, self.best_dev_acc, self.best_test_acc)
        print(self.st_best)

        return st


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        # self.stacked_en = nn.ModuleList(
        #     [nn.Linear(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList([nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.ReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            #x = gnn(x)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x
