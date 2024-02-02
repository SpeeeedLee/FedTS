import time
import torch
import torch.nn.functional as F

from misc.utils import *
from models.nets import *
from modules.federated import ClientModule



class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = CNN().cuda(self.gpu_id)
        self.parameters = list(self.model.parameters()) 

    def init_state(self):
        self.optimizer =  torch.optim.SGD(self.parameters, lr=0.01, weight_decay=1e-5, momentum=0.9) # follow pFedGraph
        self.log = {
            'lr': [],'train_lss': [],
            'ep_local_val_lss': [],'ep_local_val_acc': [],
            'rnd_local_val_lss': [],'rnd_local_val_acc': [],
            'ep_local_test_lss': [],'ep_local_test_acc': [],
            'rnd_local_test_lss': [],'rnd_local_test_acc': [],
        }

    

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd[f'personalized_{self.client_id}' \
            if (f'personalized_{self.client_id}' in self.sd) else 'global'])
        
    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train(self):
        st = time.time()
        
        print(f"worker {self.worker_id} start training & evaluation cleint {self.client_id}")
        ################################# 先針對server傳回之model 做一次evaluate --> 雖然會跟上一次一模一樣，但圖形做出來比較好跟有FL的比較########################################
        val_local_acc, val_local_lss = self.validate(mode='valid')
        test_local_acc, test_local_lss = self.validate(mode='test')
        self.logger.print(
            f'rnd: {self.curr_rnd+1}, ep: {0}, ' 
            f'train_loss: None, val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f},  {time.time()-st:.2f}s'
        )
        self.log['train_lss'].append('None') 
        self.log['ep_local_val_acc'].append(val_local_acc)
        self.log['ep_local_val_lss'].append(val_local_lss)
        self.log['ep_local_test_acc'].append(test_local_acc)
        self.log['ep_local_test_lss'].append(test_local_lss)

        ##################################################################################################################
        # Loop through all epochs
        for ep in range(self.args.n_eps):
            st = time.time()
            self.model.train()
            train_losses = []

            # Loop through all batches
            for _, (images, labels) in enumerate(self.loader.pa_loader_train):  
                images = images.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)
                self.optimizer.zero_grad() # 避免梯度累積!
                y_hat = self.model(images)
                train_lss = F.cross_entropy(y_hat, labels) 
                train_lss.backward()
                train_losses.append(train_lss.item())
                self.optimizer.step()

            #　每個epochs結束後，再做一次evaluations
            val_local_acc, val_local_lss = self.validate(mode='valid')
            test_local_acc, test_local_lss = self.validate(mode='test')

            self.logger.print(
                f'rnd:{self.curr_rnd+1}, ep:{ep+1}, '
                f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
        # 一個FL round結束後，拿client的最後一個epoch之evaluation結果出來紀錄    
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.save_log()

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.pa_loader_train)
        }
    
    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })