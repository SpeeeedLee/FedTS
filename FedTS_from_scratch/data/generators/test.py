import torch
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data import Subset, random_split, ConcatDataset
import random


n_clients = 10
dataset_name = 'cifar100'  # Dataset name
all_data_path = f'../../datasets/{dataset_name}/cluster_dirichlet_v1'
client_data_path = f'../../datasets/{dataset_name}/cluster_dirichlet_v1/{n_clients}'



def prepare_data(dataset_name, all_data_path):
    if dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加載 CIFAR-100 訓練集
        train_dataset = datasets.CIFAR100(root=all_data_path, train=True, download=True, transform=transform)
        # 加載 CIFAR-100 測試集
        test_dataset = datasets.CIFAR100(root=all_data_path, train=False, download=True, transform=transform)
        
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        
        return combined_dataset
    else:
        raise ValueError("No prepare for this dataset !")


def split_data(dataset):

    n_tot_data = len(dataset)

    # Client 0 gets 30% of the data
    split_1, dataset = random_split(dataset, [int(n_tot_data * 0.3), int(n_tot_data - int(n_tot_data * 0.3))])

    n_remain_data = len(dataset)
    # Clients 1-4 get 40% of the data
    split_2, split_3 = random_split(dataset, [int(n_tot_data * 0.4), int(n_remain_data - int(n_tot_data * 0.4))])

    return split_1, split_2, split_3


dataset = prepare_data(dataset_name, all_data_path)
split_1, split_2, split_3 = split_data(dataset)


for data, label in split_3:
    print(data)
    print(type(data))
    print('\n')
    print(label)
    print(type(label))
    