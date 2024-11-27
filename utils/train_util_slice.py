import os
import random
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from method.lnmf import LNMF
from utils.data_util import get_dataset_info, D_COIL100, D_OLI, D_COIL20, D_DIGITS, D_USPS, D_MNIST, D_FMNIST
from utils.data_loader import RandDataset
from utils.method_util import get_classify_metrics, init_wh, ret_cluster_res, compute_nmf_obj

CLASSIFY, CLUSTER = 'Classify', 'Cluster'


def train_slice(param, data_name, test_type=CLASSIFY, print_info=True, cuda=True):
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

    # Prepare dataset
    dataset_path = os.path.join(os.getcwd(), 'data')
    data, label, r = get_dataset_info(dataset_path, name=data_name)
    data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    data_slice = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=global_seed)
    for i, (_, test_index) in enumerate(skf.split(data, label)):
        data_slice.append((data[test_index], label[test_index]))
    data_train, label_train = data_slice[0]
    data_train = data_train.T

    train_loader = DataLoader(dataset=RandDataset(data_train, device=device), batch_size=batch_size,
                              num_workers=0, shuffle=True)

    m, n = data_train.shape
    n_batch_train = n // batch_size
    if 0 < n_batch_train < 5:
        n_batch_print = n_batch_train
    elif n_batch_train > 5:
        n_batch_print = n_batch_train // 5
    else:
        raise ValueError(f'batch size error: {n_batch_train}')
    w_init, _ = init_wh(data=data_train.T, rank=r, seed=global_seed, init_type='rand', print_info=print_info)
    w_init_tensor = torch.from_numpy(w_init).to(device=device, dtype=torch.float32)

    if print_info:
        print(f"Processing {data_name} dataset")
        print(f'm: {m}, n: {n}, r: {r}, b: {batch_size}, n_layer: {layers}, \n'
              f'lr: {lr_default}(rho: {lr_rho}, fc: {lr_wt})')

    # Prepare model
    model = LNMF(m, n, batch_size, r, l=layers, rho=rho, device=device, init=w_init_tensor)
    model = model.to(device)
    metric_name = 'NMI' if test_type == CLUSTER else 'ACC'
    opt_param_list = [
        {'params': [r for r in model.rho_w] + [r for r in model.rho_h], 'lr': lr_rho},
        {'params': [wi for wi in model.wtw_para], 'lr': lr_wt},
        {'params': [wi for wi in model.w_para.parameters()], 'lr': lr_wt},
    ]
    optimizer = optim.Adam(opt_param_list, weight_decay=1e-3, betas=(0.9, 0.9), lr=lr_default)
    metric_list, time_list, obj_list, df_list = [], [], [], []

    for e in range(epoch):
        # Train model
        model.train()
        if print_info:
            print("*" * 50)
        time_cost = 0.0
        for j, input_bs in enumerate(train_loader):
            loss_list, total_loss = [], 0
            optimizer.zero_grad()
            start = time()
            input_bs = input_bs.T
            W, H = model(input_bs)
            for i in range(layers):
                loss_list.append(model.get_obj(input_bs, W[i], H[i]))
                total_loss = total_loss + loss_list[-1]
            total_loss.backward()
            optimizer.step()
            time_cost += time() - start
            if print_info and j % n_batch_print == 0:
                print(f'Epoch: {e + 1}/{epoch} [Batch: {j + 1} / {n_batch_train}]')
                print(', '.join(f"l{i + 1}:{val: .2f}" for i, val in enumerate(loss_list)))
        # Test model
        model.eval()
        metric_slice_list, time_slice_list = [], []
        for data_test, label_test in data_slice[1:]:
            data_test_tensor = torch.from_numpy(data_test.T).to(device=device, dtype=torch.float32)
            start = time()
            W, H = model(data_test_tensor)
            test_time = time() - start
            metric_layer_list, obj_epoch_list = [], []
            for l in range(layers):
                W_test, H_test = W[l].cpu().detach().numpy(), H[l].cpu().detach().numpy()
                if test_type == CLASSIFY:
                    X_train, X_test, y_train, y_test = train_test_split(
                        H_test.T, label_test, test_size=0.3, stratify=label_test, random_state=global_seed)
                    classifier = LogisticRegression(max_iter=100)
                    classifier.fit(X_train, y_train)
                    labels_predict = classifier.predict(X_test)
                    metric_epoch = get_classify_metrics(y_test, labels_predict)
                elif test_type == CLUSTER:
                    metric_epoch = ret_cluster_res(H_test.T, label_test, r, global_seed)
                else:
                    raise ValueError(f'Wrong test type {test_type}')
                metric_layer_list.append(metric_epoch)
                obj_epoch_list.append(compute_nmf_obj(data_test.T, W_test, H_test))
            max_metric_layer_i = np.max(metric_layer_list)
            metric_slice_list.append(max_metric_layer_i)
            time_slice_list.append(test_time)
            max_index = metric_layer_list.index(max_metric_layer_i)
            obj_list.append(obj_epoch_list[max_index])
        # get metric
        time_slice_list.append(np.mean(time_slice_list))
        metric_slice_list.append(np.mean(metric_slice_list))
        metric_list.append(metric_slice_list[-1])
        res_epoch_dict = {
            'Test Time': time_slice_list,
            metric_name: metric_slice_list,
        }
        column_names = [f'data_slice{i + 2}' for i in range(len(time_slice_list) - 1)] + ['Average']
        res_epoch_df = pd.DataFrame(res_epoch_dict, index=column_names).round(5)
        df_list.append(res_epoch_df)
        if print_info:
            print(res_epoch_df)
    best_avg_res = np.max(metric_list)
    best_avg_res_index = metric_list.index(best_avg_res)
    if print_info:
        print(f"Best average res in Epoch {best_avg_res_index + 1} ({best_avg_res:.5f})")
    return best_avg_res
