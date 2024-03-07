import time
import torch
import torch.nn.functional as F

from misc.utils import *
from models.nets import *
from modules.federated import ClientModule

from scipy.spatial.distance import cosine

'''
This file implement FedTS
There exist many possible ways for FedTS, 
"clients upload there model only when cosine do not change much" was implemented in this file
'''


class Client(ClientModule):
    '''
    Multi processing中、client的執行步驟:

    self.client.switch_state(client_id)
    self.client.on_receive_message(curr_rnd)
    self.client.on_round_begin()
    self.client.save_state()
    '''

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        if self.args.dataset == 'cifar100':
            self.model = CNN_100().cuda(g_id)
        elif self.args.dataset == 'cifar10':
            self.model = CNN().cuda(g_id)
            self.cluster_model = CNN().cuda(g_id)
        else:
            raise NotImplementedError('還沒Build對應的model')
        self.parameters = list(self.model.parameters()) 
        self.local_dict = {}

    def init_state(self):
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.01, weight_decay=1e-5, momentum=0.9) # follow pFedGraph
        self.log = {
            'lr': [],'train_lss': [],
            'ep_local_val_lss': [],'ep_local_val_acc': [],
            'rnd_local_val_lss': [],'rnd_local_val_acc': [],
            'ep_local_test_lss': [],'ep_local_test_acc': [],
            'rnd_local_test_lss': [],'rnd_local_test_acc': [],
            'epochs_in_FL': [], 
        }

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd[f'personalized_{self.client_id}' \
            if (f'personalized_{self.client_id}' in self.sd) else 'global']) # 如果在某輪中clients沒被選中執行FL，則使用FedAvg的結果去initialize?
        # 從server端recive這一輪FL round的新model後，將之存進sd中的last model!
        self.local_dict['last_model'] = get_state_dict(self.model)

    def update(self, update):
        # self.prev_w = convert_np_to_tensor(update['model'], self.gpu_id)
        # Maybe can use this in FedTS
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=False, skip_mask=False)
        set_state_dict(self.cluster_model, update['model'], self.gpu_id, skip_stat=False, skip_mask=False)
    
    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train(self):
        st = time.time()
        
        print(f"worker {self.worker_id} start training & evaluation cleint {self.client_id}")
        ################################# 先針對server傳回之model 做一次evaluate #########################################
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
        self.reach_cosine_criteria = False
        self.reach_epoch_limit = False
        # Loop through epochs until reach the criteria
        ep = 0
        while self.reach_cosine_criteria == False and self.reach_epoch_limit == False:
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
                train_losses.append(train_lss.item())

                regularize_loss = 0
                if self.curr_rnd >= 1 :
                    # 一開始不做正則化! 因為還沒有server回傳的anchor model !
                    # Regularization
                    # print('開始考慮正則化')
                    current_model =  flatten_state_dict_to_tensor(get_state_dict(self.model).copy(), self.gpu_id)       # 先轉成state_dict可以避免model之前的梯度累積
                    cluster_model = flatten_state_dict_to_tensor(get_state_dict(self.cluster_model).copy(), self.gpu_id)
                    regularize_loss = 0.01 * torch.dot(cluster_model, current_model) / torch.linalg.norm(current_model)
                
                total_loss = train_lss + regularize_loss
                total_loss.backward()
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
            
            # 儲存train完這個epoch的model至sd[f{c_id}_FedTS]['current_model']，以利進行cosine_criteria檢查
            self.local_dict['current_model'] = get_state_dict(self.model)
            

            ep += 1

            '''
            這邊的邏輯是:
                在matching rounds都做完之前:
                    檢查是否reach cosine criteria --> 若是、停止training、回傳給server；
                在matching rounds都結束之後:
                    不應該再檢查consine criteria --> 只檢查是否reach epoch limit
            '''
            if self.curr_rnd < self.args.matching_rounds:
                if ep > 1:
                    if self.check_cosine_criteria() == True : 
                        self.reach_cosine_criteria = True
            else:
                if self.check_epoch_limit_criteria(current_epoch = ep) == True:
                    self.reach_epoch_limit = True

            # 儲存train完這個epoch的model至local_dict['last_model']，以利下個epoch進行cosine_criteria檢查
            self.local_dict['last_model'] = get_state_dict(self.model)

        # 一個FL round結束後，拿client的最後一個epoch之evaluation結果出來紀錄    
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.log['epochs_in_FL'].append(ep) # 記錄這個client跑了幾個epochs
        self.save_log()


    def transfer_to_server(self):
        ## Below Information are stored into server ##
        if self.reach_epoch_limit == True:
            self.sd[self.client_id] = {
                'model': get_state_dict(self.model),
                'train_size': len(self.loader.pa_loader_train), 
                'client_over' : True
            }
        else:
            self.sd[self.client_id] = {
                'model': get_state_dict(self.model),
                'train_size': len(self.loader.pa_loader_train), 
                'client_over' : False
            }



    def save_state(self):
            torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
                'optimizer': self.optimizer.state_dict(),
                'model': get_state_dict(self.model),
                'log': self.log,
            })

    def check_cosine_criteria(self):
        '''For checking weather the client model do not change the direction'''
        last_model_direction =  flatten_state_dict(self.local_dict['last_model'].copy()) \
                                                        - flatten_state_dict(self.sd['initial_global_model'].copy()) 
        current_model_direction = flatten_state_dict(self.local_dict['current_model'].copy()) \
                                                        - flatten_state_dict(self.sd['initial_global_model'].copy()) 
        
        cosine_ = 1 - cosine(last_model_direction, current_model_direction)

        if cosine_ >= self.args.ready_cosine:
            print(f"{self.client_id} reach the cosine criteria")
            return True
        else:
            return False
    
    def check_epoch_limit_criteria(self, current_epoch):
        '''For checking weather the client model reach the epoch limit'''
        past_epoch = sum(self.log['epochs_in_FL'])
        if past_epoch + current_epoch >= self.args.epoch_limit: 
            print(f"{self.client_id} reach the epoch limit criteria")
            return True
        else:
            return False