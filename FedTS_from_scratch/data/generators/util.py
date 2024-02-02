import os
import torch
from torch.utils.data import random_split


def torch_save(base_dir, filename, data):
    '''Save dataset to the directory'''
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)


def split_train_val_test(client_dataset,train_portion, val_portion):
    '''Split each client dataset to Train, Val, Test '''
    train_size = int(train_portion * len(client_dataset))
    val_size = int(val_portion * len(client_dataset))
    test_size = len(client_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(client_dataset, [train_size, val_size, test_size])
    data_partition = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return data_partition
