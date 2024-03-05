### Generate Data Partition which is suitable for "FedTS" ###

'''
CIFAR_100
    Random Split 30% Data --> Client 0
    Random Split 40% Data --> 先按照class分類 -->Client 1, 2, 3, 4 (Each with all of the 25 labels)
    Random Split 30% Data --> Client 5, 6, 7, 8, 9(Dirichlet(0.01))
'''

import torch
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data import Subset, random_split, ConcatDataset
import random

from util import *
from dirichlet_util import dirichlet_split_noniid

train_portion = 0.8 # change
val_portion = 0.1 # change
n_clients = 10  # Number of clients
seed = 1234  # Random seed

dataset_name = 'cifar100'  # Dataset name
all_data_path = f'../../datasets/{dataset_name}/cluster_dirichlet_v1'
client_data_path = f'../../datasets/{dataset_name}/cluster_dirichlet_v1/{n_clients}'


torch.manual_seed(seed)


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

        random.shuffle(combined_dataset)
        
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


def save_data(client_datasets, client_data_path):
    for i, client_dataset in enumerate(client_datasets):
        torch.save(client_dataset, os.path.join(client_data_path, f'client_{i}_data.pt'))




dataset = prepare_data(dataset_name, all_data_path)
split_1, split_2, split_3 = split_data(dataset)

######################################
#### 將split_1的data，分給client_0 ####
client_0_dataset_split = split_train_val_test(split_1, train_portion, val_portion)
# 保存數據
torch_save(client_data_path, f'client_{0}_data.pt', client_0_dataset_split)
print(f'client 0 之資料量 : {len(split_1)}')
print("Split 1 已經處理完畢")
######################################

################################################
#### 將split_2 依照class做分類，裝到一個字典中 ####
class_datasets = {i: [] for i in range(100)}
mapping_list = list(range(100))
random.seed(42)
random.shuffle(mapping_list)
for img, label in split_2:
    class_datasets[label].append((img, label))

# 重置隨機種子
random.seed(None)

for client_id in range(1, 5): # client 1 ~ 4
    client_dataset = []
    for key in mapping_list[(client_id - 1)*25: client_id*25]:
        client_dataset.extend(class_datasets[key])
    random.shuffle(client_dataset)
    client_dataset_split = split_train_val_test(client_dataset, train_portion, val_portion)
    # 保存數據
    torch_save(client_data_path, f'client_{client_id}_data.pt', client_dataset_split)
    print(f'client {client_id} 之資料量 : {len(client_dataset)}')
print("Split 2 已經處理完畢")
################################################
    
    
########################################
#### 將split_3 依照Dirichlet進行分類 ####
split_3_data = []
split_3_labels = []
split_3_labels_np = np.empty((0, 1), int)
for data, label in split_3:
    split_3_data.append(data)
    split_3_labels.append(label)  
    split_3_labels_np = np.append(split_3_labels_np, [[label]], axis=0)  # <<< --- 這麼做是為了滿足其格式如同tensorflow所載下來的標籤，以利之後的 "dirichlet_split_noniid" function 
# 用自定義function做Dirichlet Partition
client_idcs = dirichlet_split_noniid(split_3_labels_np, alpha = 0.01, NUM_CLIENTS = 5)
# 實際切分數據給clients並儲存
for client_id in range(5, 10): # client 5 ~ 9
    client_dataset = []
    index_list = client_idcs[client_id - 5]
    for index in index_list:
        client_dataset.append((split_3_data[index], split_3_labels[index]))
    random.shuffle(client_dataset)
    client_dataset_split = split_train_val_test(client_dataset, train_portion, val_portion)
    # 保存數據
    torch_save(client_data_path, f'client_{client_id}_data.pt', client_dataset_split)
    print(f'client {client_id} 之資料量 : {len(client_dataset)}')
print("Split 3 已經處理完畢")
#######################################


print("資料切分並儲存完畢")
