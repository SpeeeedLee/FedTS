'''
這個python file的主要目的是 把某個資料夾中的所有client之.pt file load入
逐一打開並檢查裡面的label分布情況並記錄
等做完所有client後
畫出一張distribution的圖 其中 不同client用不同的顏標示!
'''

import os 
import torch
from collections import Counter
import matplotlib.pyplot as plt

dataset_name = 'cifar10'  # Dataset name，改這裡
partition_method = 'random_iid' # partition method，改這裡
n_clients = 10 # client數量，改這裡
n_classes = 10 # class數量，改這裡
all_data_path = f'../../datasets/{dataset_name}/{partition_method}/{n_clients}'


# 打印出結果
clients_label_counts = {}
for filename in sorted(os.listdir(all_data_path)):
    if filename.endswith('.pt'):
        file_path = os.path.join(all_data_path, filename)
        print(f'Counting file: {file_path}')
        dataset = torch.load(file_path)
        all_labels = [label for subset in ['train', 'val', 'test'] for _, label in dataset[subset]]
        label_counts = Counter(all_labels)
        print(label_counts)
        print('\n')
        clients_label_counts[filename] = label_counts
        

# 畫出結果

labels = list(range(n_classes))

# 初始化一个字典，用於儲存每個標籤在每個客戶端的數量
label_data = {label: [] for label in labels}

# 填充label_data
for filename, counts in clients_label_counts.items():
    for label in labels:
        label_data[label].append(counts.get(label, 0))

# 繪製堆疊柱狀圖
fig, ax = plt.subplots(figsize=(20, 10))
bottom = [0] * len(labels)
for filename, counts in clients_label_counts.items():
    data = [counts.get(label, 0) for label in labels]
    ax.bar(labels, data, bottom=bottom, label=filename)
    bottom = [sum(x) for x in zip(bottom, data)]

# 設置圖表的其他屬性
ax.set_xlabel('Label')
ax.set_ylabel('Count')
ax.set_title('Label Distribution Across Datasets')
ax.set_xticks(labels)
ax.set_xticklabels(labels, rotation=90)
ax.legend(title='Dataset')

plt.savefig(f'./examinator/{dataset_name}_{partition_method}_{n_clients}_label.png', dpi=300)

print("圖表已儲存!")