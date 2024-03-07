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

'''    
class CNN_100(nn.Module):
    def __init__(self):
        super(CNN_100, self).__init__()
        # 第一個捲積層: 輸入通道 3 (RGB), 輸出通道 16, 捲積核大小 3x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 第二個捲積層: 輸入通道 16, 輸出通道 32, 捲積核大小 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 第三個捲積層: 輸入通道 32, 輸出通道 64, 捲積核大小 3x3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # 三個全連接層
        self.fc1 = nn.Linear(4 * 4 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)  # CIFAR100 有 100 个类别

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
'''
    
class CNN_100(nn.Module):
    def __init__(self):
        super(CNN_100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(2 * 2 * 256, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 100)  # CIFAR100 有 100 個類別

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 2 * 2 * 256)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x