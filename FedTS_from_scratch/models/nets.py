import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    '''
    試過 兩個卷積層、一個FC --> 太簡單、一下就converge、容易overfitting
    所以決定跟隨 pFedGraph
    做三個卷積層、三個FC
    '''
    def __init__(self):
        super(CNN, self).__init__()
        # 第一個捲積層: 輸入通道 3 (RGB), 輸出通道 16, 捲積核大小 3x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 第二個捲積層: 輸入通道 16, 輸出通道 32, 捲積核大小 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 第三個捲積層: 輸入通道 32, 輸出通道 64, 捲積核大小 3x3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # 三個全連接層
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # CIFAR10 有 10 個類別

    def forward(self, x):
        # 通過捲積層和池化層
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # 攤平特徵向量
        x = x.view(-1, 4 * 4 * 64)

        # 通過全連接層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

from collections import OrderedDict
import numpy as np 
def get_state_dict(model):
    # model.state_dict 是 pytorch 的標準語法
    # 會返回一個dict，key是層的名稱，value是其中的權重
    state_dict = convert_tensor_to_np(model.state_dict()) 
    return state_dict

def convert_tensor_to_np(state_dict):
    ''' Just a simple function for converting state_dict to python OrderedDict'''
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])



def flatten_state_dict(state_dict):
    '''Convert Ordered Dict to 1D numpy Array'''
    flat_weights = [weights.flatten() for weights in state_dict.values()]
    return np.concatenate(flat_weights)
from scipy.spatial.distance import cosine

model1 = CNN()
model2 = CNN()
model1_weight = flatten_state_dict(get_state_dict(model1))
model2_weight = flatten_state_dict(get_state_dict(model2))
print(1 - cosine(model1_weight, model1_weight))