import pdb
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import *
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

# def cluster_acc(y_true, y_pred):
#     y_true = y_true - np.min(y_true)
#     l1 = list(set(y_true))
#     numclass1 = len(l1)
#     l2 = list(set(y_pred))
#     numclass2 = len(l2)
#
#     ind = 0
#     if numclass1 != numclass2:
#         for i in l1:
#             if i in l2:
#                 pass
#             else:
#                 y_pred[ind] = i
#                 ind += 1
#
#     l2 = list(set(y_pred))
#     numclass2 = len(l2)
#
#     if numclass1 != numclass2:
#         print('error')
#         return
#
#     cost = np.zeros((numclass1, numclass2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
#             cost[i][j] = len(mps_d)
#
#     m = Munkres()
#     cost = cost.__neg__().tolist()
#     indexes = m.compute(cost)
#
#     new_predict = np.zeros(len(y_pred))
#     for i, c in enumerate(l1):
#         c2 = l2[indexes[i][1]]
#         ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
#         new_predict[ai] = c
#
#     acc = metrics.accuracy_score(y_true, new_predict)
#     f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
#     return acc, f1_macro

def cluster_acc(y_true, y_pred):
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

    #from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    # pdb.set_trace()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    # acc=0
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp