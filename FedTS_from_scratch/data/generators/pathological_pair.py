'''
This python file aim at creating pairs of clients that have data on exactly the same set of labels

還需要調整一下才能 直接用來 生成 20個client的dataset
'''

import torch
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data import Subset, random_split

from util import *

n_clients = 10 # change
train_portion = 0.8 # change
val_portion = 0.1 # change
# if change the portion, previous will be overwrite
seed = 1234 # change

dataset_name = 'cifar10' # change
all_data_path = f'../../datasets/{dataset_name}/pathological_pair'
client_data_path = f'../../datasets/{dataset_name}/pathological_pair/{n_clients}'

torch.manual_seed(seed)

def prepare_data(dataset_name, all_data_path):
    
    if dataset_name == 'cifar10':
        # 加載 CIFAR-10 數據集
        transform = transforms.Compose([ # transform.Compose 允許用來將一系列transform操作寫在一起
            # 原先影像是以PIL型態抓取
            transforms.ToTensor(), # Numpy array / PIL image --> Tensor (範圍會在0~1之間)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 第一個元組是mean，第二個是標準差 (範圍會變在-1~1之間)
        ])

        # 加載 CIFAR-10 訓練集
        train_dataset = datasets.CIFAR10(root=all_data_path, train=True, download=True, transform=transform)
        # 加載 CIFAR-10 測試集
        test_dataset = datasets.CIFAR10(root=all_data_path, train=False, download=True, transform=transform)

    # 按照類別劃分數據集
    class_datasets = {i: [] for i in range(10)}
    for dataset in [train_dataset, test_dataset]:
        for img, label in dataset:
            class_datasets[label].append((img, label))

    return class_datasets

def generate_data(class_datasets, client_data_path, n_clients):
    
    client_datasets = {i: [] for i in range(n_clients)}

    # 分配數據到不同的客戶端
    for class_id, dataset in class_datasets.items():
        client_id_first = class_id // 2  # 0,1 --> 0 ； 2,3 --> 1 ； 4,5 --> 2....8,9 --> 4
        client_id_second = (client_id_first + 5) % n_clients
        client_datasets[client_id_first].extend(dataset)
        client_datasets[client_id_second].extend(dataset)

    for client_id, dataset in client_datasets.items():
        client_dataset_split = split_train_val_test(dataset, train_portion, val_portion)
        # 保存數據
        torch_save(client_data_path, f'client_{client_id}_data.pt', client_dataset_split)


class_datasets = prepare_data(dataset_name, all_data_path)
generate_data(class_datasets, client_data_path, n_clients)
print("資料切分並儲存完畢")
