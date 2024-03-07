#### 創建幾個initail model版本，以利用同個初始化模型開始訓練，進行叫好的比較 ####
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from nets import CNN, CNN_100

        
folder_path = '../initial_model/CNN/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

CNN_model_1 = CNN()
torch.save(CNN_model_1.state_dict(), '../initial_model/CNN/model_1.pkl')

print('initial model 已儲存完畢')