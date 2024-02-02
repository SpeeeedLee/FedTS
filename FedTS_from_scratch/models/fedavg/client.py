import time
import torch.nn.functional as F


from modules.federated import ClientModule
from models.nets import *
from misc.utils import *




class Client(ClientModule):
    '''
    Call the functions in the following orders :
        1. switch_state
        2. on_receive_message
        3. on_round_begin
        4. save_state
    '''
    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = CNN().cuda(g_id) #實例化一個初始模型，並丟到gpu
        self.parameters = list(self.model.parameters()) # the elements in this list are "torch.nn.Parameter" 對象 


    ########### switch state will design whether use init_state or load_state ! #####################
    def init_state(self):
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.01, weight_decay=1e-5, momentum=0.9) # follow pFedGraph
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

    #################################################################################################


    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })


    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True)

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
                f'rnd: {self.curr_rnd+1}, ep: {ep+1}, ' 
                f'train_loss: {np.mean(train_losses):.4f}, val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f},  {time.time()-st:.2f}s'
            )
            self.log['train_lss'].append(np.mean(train_losses))

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
        # print(f"worker {self.worker_id} finish  training & evaluation cleint {self.client_id}")


    def transfer_to_server(self):
        '''since sd is shared, so below indeed means transfer to server !'''
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.pa_loader_train)
        }


    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(), # state_dict() will return the optimizer as dict
            'model': get_state_dict(self.model),
            'log': self.log,
        })