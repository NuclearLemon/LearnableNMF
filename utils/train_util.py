import os
import random
from time import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from method.lnmf import LNMF
from utils.data_util import get_dataset_info, D_COIL100, D_OLI, D_COIL20, D_DIGITS, D_USPS, D_MNIST, D_FMNIST
from utils.data_loader import RandDataset
from utils.method_util import get_classify_metrics, init_wh, ret_cluster_res, compute_nmf_obj

CLASSIFY, CLUSTER = 'Classify', 'Cluster'


def train(param, data_name, test_type, print_info=True, cuda=True):
    batch_size, layers = param.batch, param.layer
    lr_default, lr_rho, lr_wt = param.lr_set
    global_seed = param.seed
    rho, epoch = param.rho, param.epoch

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    random.seed(global_seed)
    device = torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")
    if print_info:
        print(f'Device: {device}')

    dataset_path = os.path.join(os.getcwd(), 'data')
    data, label, r = get_dataset_info(dataset_path, name=data_name)
    data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    n, m = data.shape
    train_loader = DataLoader(dataset=RandDataset(data.T, device=device), batch_size=batch_size,
                             num_workers=0, shuffle=True)

    random_indices = list(range(n))
    random.shuffle(random_indices)
    data = np.array([data[i] for i in random_indices])
    label = [label[i] for i in random_indices]
    data_tensor = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    n_batch_train = n // batch_size
    if 0 < n_batch_train < 5:
        n_batch_print = n_batch_train
    elif n_batch_train > 5:
        n_batch_print = n_batch_train // 5
    else:
        raise ValueError(f'batch size error: {n_batch_train}')
    w_init, h_init = init_wh(data=data, rank=r, seed=global_seed, init_type='rand', print_info=print_info)
    w_init_tensor = torch.from_numpy(w_init).to(device=device, dtype=torch.float32)

    if print_info:
        print(f"Processing {data_name} dataset")
        print(f'm: {m}, n: {n}, r: {r}, b: {batch_size}, n_layer: {layers}, \n'
              f'lr: {lr_default}(rho: {lr_rho}, fc: {lr_wt})')
    model = LNMF(m, n, batch_size, r, l=layers, rho=rho, device=device, init=w_init_tensor)
    model = model.to(device)
    data = data.T
    metric_name = 'NMI' if test_type == CLUSTER else 'ACC'
    opt_param_list = [
        {'params': [r for r in model.rho_w] + [r for r in model.rho_h], 'lr': lr_rho},
        {'params': [wi for wi in model.wtw_para], 'lr': lr_wt},
        {'params': [wi for wi in model.w_para.parameters()], 'lr': lr_wt},
    ]
    optimizer = optim.Adam(opt_param_list, weight_decay=1e-3, betas=(0.9, 0.9), lr=lr_default)
    metric_list, time_list, obj_list = [], [], []
    for e in range(epoch):
        model.train()
        if print_info:
            print("*" * 50)
        time_cost = 0.0
        for j, input_bs in enumerate(train_loader):
            optimizer.zero_grad()
            input_bs = input_bs.T
            start = time()
            W, H = model(input_bs)
            time_cost += time() - start
            # compute loss
            loss_list = list()
            total_loss = 0
            for i in range(layers):
                loss_list.append(model.get_obj(input_bs, W[i], H[i]))
                total_loss = total_loss + loss_list[-1]
            start = time()
            total_loss.backward()
            time_cost += time() - start
            if print_info and j % n_batch_print == 0:
                print(f'Epoch: {e + 1}/{epoch} [Batch: {j + 1} / {n_batch_train}]')
                print(', '.join(f"l{i + 1}:{val: .2f}" for i, val in enumerate(loss_list)))
            start = time()
            optimizer.step()
            time_cost += time() - start
        model.eval()
        start = time()
        W, H = model(data_tensor.T)
        time_cost += time() - start
        metric_epoch_list, obj_epoch_list = [], []
        for l in range(layers):
            W_train, H_train = W[l].cpu().detach().numpy(), H[l].cpu().detach().numpy()
            if test_type == CLASSIFY:
                X_train, X_test, y_train, y_test = train_test_split(
                    H_train.T, label, test_size=0.3, stratify=label, random_state=global_seed)
                classifier = LogisticRegression(max_iter=100)
                classifier.fit(X_train, y_train)
                labels_predict = classifier.predict(X_test)
                metric_epoch = get_classify_metrics(y_test, labels_predict)
            elif test_type == CLUSTER:
                metric_epoch = ret_cluster_res(H_train.T, label, r, global_seed)
            else:
                raise ValueError(f'Wrong test type {test_type}')
            metric_epoch_list.append(metric_epoch)
            obj_epoch_list.append(compute_nmf_obj(data, W_train, H_train))
        max_metric_epoch_i = np.max(metric_epoch_list)
        max_metric_epoch_i_index = metric_epoch_list.index(max_metric_epoch_i)
        metric_list.append(max_metric_epoch_i)
        time_list.append(time_cost)
        max_index = metric_epoch_list.index(max_metric_epoch_i)
        obj_list.append(obj_epoch_list[max_index])
        if print_info:
            print(f"{metric_name} list: {[f'{x:.4f}' for x in metric_epoch_list]}")
            print(f"Epoch {e + 1}, Time: {time_cost:.4f}s, "
                  f"Max {metric_name}: {max_metric_epoch_i:.4f} [Layer {max_metric_epoch_i_index + 1}]")
            print(f"Obj list: {[f'{x:.2f}' for x in obj_epoch_list]}")
    if print_info:
        print(f'Max {metric_name} value in Training: {np.max(metric_list):.4f} '
              f'(Epoch {metric_list.index(np.max(metric_list)) + 1}), \nTrain time cost: {np.sum(time_list): .4f}s')
    return np.max(metric_list)
