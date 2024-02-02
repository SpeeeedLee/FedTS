'''
This python file should be used to generate dataset partitions for every clients
'''

import torch
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data import Subset, random_split

from util import *


n_clients = 20 # change
train_portion = 0.8 # change
val_portion = 0.1 # change
# if change the portion, previous will be overwrite
seed = 1234 # change

dataset_name = 'cifar10' # change
all_data_path = f'../../datasets/{dataset_name}/random_iid'
client_data_path = f'../../datasets/{dataset_name}/random_iid/{n_clients}'



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

        # 結合訓練集和測試集，並return
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

        return combined_dataset

    else:
        raise ValueError("No prepare for this dataset !")

    


def generate_data(combined_dataset, client_data_path, n_clients, train_portion, val_portion):

    data_per_client = len(combined_dataset) // n_clients
    client_datasets = []

    print(f"整個datasets之資料量 : {(len(combined_dataset))}")
    print(f"一個client之資料量 : {data_per_client}")

    for _ in range(n_clients):
        # 將數據隨機分割給每個客戶端
        '''
        random_split 是 pytorch的一個函數
        input : 待切割的資料集, 一個list(要切割成幾份，每份有多少個資料)
        output : 切割成果
        '''
        client_dataset, combined_dataset = random_split(combined_dataset, [data_per_client, len(combined_dataset) - data_per_client])
        client_datasets.append(client_dataset)

    for i, client_dataset in enumerate(client_datasets):
        # 將每個clients數據分成train, val, test； 並儲存成.pt檔案
        client_dataset_split = split_train_val_test(client_dataset,train_portion, val_portion)
        torch_save(client_data_path, f'client_{i}_data.pt', client_dataset_split)


combined_dataset = prepare_data(dataset_name, all_data_path)
generate_data(combined_dataset, client_data_path, n_clients, train_portion, val_portion)
print("資料切分並儲存完畢")