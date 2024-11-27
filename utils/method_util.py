import random
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition._nmf import _initialize_nmf
from sklearn.metrics import accuracy_score

CLASSIFY, CLUSTER = 'Classify', 'Cluster'


def get_classify_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def init_wh(data, rank, seed=1, init_type='rand', print_info=True):
    """
    data (n * m)
    """
    if print_info:
        print(f'Init W and H by [{init_type}]')
    start = time()
    if init_type == 'rand':
        a, b = _initialize_nmf(data, rank, init='random', random_state=seed)
        W, H = b.T, a.T
    else:
        raise ValueError(f'Unexpected init type {init_type}')
    if print_info:
        print(f'Inited, time cost {time() - start: .4f}s')
    return W, H


def get_cluster_metrics(y_true, y_pred):
    """return cluster metrics"""
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ret_cluster_res(data, label, r, seed, n_repeat=100, ret_res=False):
    random.seed(seed)
    random_integers = random.sample(range(1, 1000), n_repeat)
    avg_nmi = 0.0
    max_nmi = 0
    max_label = None
    max_kmeans = None
    for i in range(n_repeat):
        km = KMeans(n_clusters=r, init='k-means++', n_init=1, random_state=random_integers[i])
        km.fit(data)
        labels_predict = km.labels_
        nmi_i = get_cluster_metrics(label, labels_predict)
        if nmi_i > max_nmi:
            max_label = list(labels_predict)
            max_kmeans = km
            max_nmi = nmi_i
        avg_nmi += nmi_i
    avg_nmi /= n_repeat
    if ret_res:
        return avg_nmi, max_nmi, max_label, max_kmeans
    return avg_nmi


def compute_nmf_obj(data, w, h):
    residual = data - np.dot(w, h)
    return np.sum(residual ** 2) / (2 * data.shape[1])
