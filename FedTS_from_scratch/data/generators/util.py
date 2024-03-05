import os
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np 

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


def plot_label_distribution(partitions, n_clients, n_classes, alpha):
    client_label_counts = []

    # 提取所有clients的label部分并计算每个client的各标签数据量
    for partition in partitions:
        labels_original = partition[1]  # 假设partition中第二个元素是label
        labels = [label[0] for label in labels_original]
        label_counts = np.bincount(labels, minlength=n_classes)
        #print(f"第{partitions}個client持有的label 分布:\n {label_counts}")
        client_label_counts.append(label_counts)
    
    for i in range(len(client_label_counts)):
        print(f"第{i}個client : {client_label_counts[i]}\n")

    plt.figure(figsize=(12, 8))

    # 绘制柱状图
    bars = np.arange(n_clients)
    bottom = np.zeros(n_clients)
    for cls in range(n_classes):
        label_counts_at_cls = [client[cls] for client in client_label_counts]
        plt.bar(bars, label_counts_at_cls, bottom=bottom, label=f"Label {cls}")
        bottom += label_counts_at_cls

    plt.xticks(bars, [f"C{i}" for i in range(n_clients)])
    plt.xlabel("Clients")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    if alpha == 'iid':
        plt.title(f"Label Distribution of Different Clients (iid)")
    else:    
        plt.title(f"Label Distribution of Different Clients (alpha = {alpha})")

    plt.show()