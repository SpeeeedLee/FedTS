import time
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import os


from misc.utils import *
from data.loader import DataLoader
from modules.logger import Logger

class ServerModule:
    def __init__(self, args, sd, gpu_server):
        self.args = args # type : Namespace
        self._args = vars(self.args) # convert Namespace from argparse to dictionary 
        self.gpu_id = gpu_server
        self.sd = sd
        self.logger = Logger(self.args, self.gpu_id, is_server=True)

    def aggregate(self, local_weights, ratio=None):
        '''
        local_weights : 一個list、每一個元素為一個dict --> 代表的是每個client的model權重
        ratio : 一個list、代表每個client的聚合比重
        '''

        aggr_theta = OrderedDict([(k,None) for k in local_weights[0].keys()]) 
        # OrderdDict 不同於普通的字典，其保留的元素被添加時的順序
        # local_weights 應該是一個list，每一個元素為一個dict --> 代表的是每個client的model權重
        # [(k,None) for k in local_weights[0].keys()] 會是一個list，每一個元素為一個元組 --> [ (model key 1, None), (model key 2, None) ...]
        # aggr_theta 再把這個list轉成字典 ! 

        if ratio is not None:
            for name, params in aggr_theta.items():
                '''loop through每個model中的權重、找其他所有client及其對應的ratio、算該權重的加權平均'''
                aggr_theta[name] = np.sum([theta[name]*ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1/len(local_weights) # 就是 1 / #clients
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        return aggr_theta


class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.logger = Logger(self.args, self.gpu_id) #  will initialize a dataloader instance for every worker
       
    def switch_state(self, client_id):
        print(f"worker {self.worker_id} is switching to client {client_id}")
        self.client_id = client_id
        self.loader.switch(client_id) # switch dataloader to dersired client
        self.logger.switch(client_id) # switch logger to desired client 
        ## p.s. "switch" function is defined in DataLoder class & Logger class seperately !
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))
    
    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })
    
    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()
    
    ########## Helper Functions for Validation #####################################################
    @torch.no_grad()
    def validate(self, mode):
        if mode == 'valid':
            loader = self.loader.pa_loader_val
        elif mode == 'test':
            loader = self.loader.pa_loader_test

        self.model.eval()

        with torch.no_grad():
            target, pred, loss = [], [], []
            for _, (images, labels) in enumerate(loader):  
                images = images.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(images, labels)  # 修改这里
                pred.append(y_hat.max(1)[1])  # 修改这里：获取最可能的类别
                target.append(labels) 
                loss.append(lss)
            acc = self.accuracy(torch.cat(pred), torch.cat(target))
        return acc, np.mean(loss)


    @torch.no_grad()
    def validation_step(self, images, labels):
        y_hat = self.model(images)
        lss = F.cross_entropy(y_hat, labels)
        return y_hat, lss.item()


    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    ##################################################################################################

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state