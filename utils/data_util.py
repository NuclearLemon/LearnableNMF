import os
import pickle
import scipy.io as scio
import gzip

D_COIL20, D_DIGITS, D_USPS, D_OLI, D_MNIST, D_FMNIST, D_COIL100 = 'COIL20', 'Digits', 'USPS', 'Olivetti', 'MNIST', 'FMNIST', 'COIL100'

S_DATA_PATH = os.path.join(os.getcwd(), 'data')

"""
number of classes for different datasets.
"""
dataset_class_dict = {
    D_COIL20: 20,
    D_DIGITS: 10,
    D_USPS: 10,
    D_MNIST: 10,
    D_FMNIST: 10,
    D_OLI: 40,
    D_COIL100: 100,
}

net_setting_classify_dict = {
    D_COIL20: (36, 5e-2),  # 0.9884
    D_DIGITS: (50, 5e-2),  # 0.9333
    D_USPS: (75, 1e-2),  # 0.9018
    D_MNIST: (500, 1e-2),  # 8119
    D_FMNIST: (1000, 5e-2),  # 0.7668
    D_COIL100: (50, 5e-3),  # 0.9713
    D_OLI: (15, 5e-2),  # 0.9750
}

net_setting_cluster_dict = {
    D_COIL20: (54, 5e-3),  # 0.7831
    D_DIGITS: (30, 5e-2),  # 0.7361
    D_USPS: (50, 1e-3),  # 0.6571
    D_MNIST: (400, 5e-2),  # 0.4911
    D_FMNIST: (1000, 5e-2),  # 0.5746
    D_COIL100: (50, 5e-3),  # 0.7684
    D_OLI: (25, 3e-2),  # 0.8027
}


def get_mat(path, feature='feature', label='label'):
    """read mat dataset"""
    data = scio.loadmat(path)
    if label is None:
        return data[feature]
    else:
        images, labels = data[feature], data[label]
        labels = labels.reshape(-1)
        return images, labels


def get_data(dir=S_DATA_PATH, name=D_COIL20):
    """return dataset (n * m)"""
    if name in [D_MNIST, D_FMNIST, D_COIL100]:
        path = os.path.join(dir, name + '.mat.gz')
        data, label = get_gz_data(path=path)
    else:
        path = os.path.join(dir, name + '.mat')
        data, label = get_mat(path=path)
    return data, label


def get_dataset_info(dir, name):
    """get dataset info"""
    data, label = get_data(dir, name=name)
    n_components = dataset_class_dict.get(name)
    if n_components is None:
        raise ValueError(f"Dataset {name} not found")
    return data, label, n_components


def get_gz_data(path):
    with gzip.open(path, 'rb') as f:
        data, label = pickle.load(f)
    return data, label
