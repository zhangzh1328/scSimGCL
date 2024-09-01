import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
def test():
    data = np.random.rand(100, 3)

    #假如我要构造一个聚类数为3的聚类器
    estimator = KMeans(n_clusters=3)
    estimator.fit(data)#聚类
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_

def k_means(data,n_class):
    estimator = KMeans(n_class)
    estimator.fit(data)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    print(label_pred)
    return label_pred


def SC_cluster(data,n_class):
    metrics_metrix = (-1 * metrics.pairwise.pairwisGaussianMixture_clustere_distances(data)).astype(np.int32)
    metrics_metrix += -1 * metrics_metrix.min()
    n_clusters_ = n_class
    lables = spectral_clustering(metrics_metrix, n_clusters=n_clusters_)
    print(lables)
    return lables

def GaussianMixture_cluster(data,n_class):
    gmm = GaussianMixture(n_components=n_class, covariance_type='spherical').fit(data)
    y_pred = gmm.predict(data)
    print(y_pred)
    return y_pred




if __name__=="__main__":
    y_score = np.array([[
        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    ], [
        [0.7, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.56, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.5, 0.3], [0.2, 0.75, 0.12],
        [0.05, 0.05, 0.9], [0.1, 0.8, 0.6], [0.9, 0.8, 0.08],
    ], [
        [0.5, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.7, 0.3],
        [0.2, 0.4, 0.3], [0.1, 0.5, 0.3], [0.2, 0.9, 0.12],
        [0.05, 0.8, 0.9], [0.1, 0.6, 0.6], [0.9, 0.8, 0.08],
    ]])
    print(k_means(y_score,3))