import time
import numpy as np

from scipy.spatial.distance import cosine

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule



class Server(ServerModule):
    '''
    Call the functions in the following orders :
        1. on_round_begin 
        2. on_round_complete
            update
            save_state    
    '''
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        # Create a model instance and store it in the assigned GPU
        if self.args.dataset == 'cifar100':
            self.model = CNN_100().cuda(gpu_server)
        elif self.args.dataset == 'cifar10':
            self.model = CNN().cuda(gpu_server)
        else:
            raise NotImplementedError('還沒Build對應的model')
        # store the initail model to the sd
        self.sd['anchor_global_model'] = get_state_dict(self.model)
        self.sim_matrices = []
        self.cos_matrices = []
        self.normalized_cos_matrices = []

    def on_round_begin(self, curr_rnd):
        '''
        update current model weights to the sd['global']
        if it is the first round, then the initial CNN model weights will be used
        ** Only one server instance, so the server object's weights will evolve through rounds !
        '''
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()


    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = [None]*self.args.n_clients
        local_weights_numpy = [None]*self.args.n_clients
        local_weights_numpy_dist_w_anchor = [None]*self.args.n_clients
        local_train_sizes = [None]*self.args.n_clients
        for c_id in updated:
            local_weights[c_id] = self.sd[c_id]['model'].copy() # 加入一個client的weights
            # 轉換成1D numpy array!
            local_weights_numpy[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy())
            # 利用向量之"末減初"，來計算client current model 與 anchor global model 的差向量
            local_weights_numpy_dist_w_anchor[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy()) \
                                                        - flatten_state_dict(self.sd['anchor_global_model'].copy()) 
            local_train_sizes[c_id] = self.sd[c_id]['train_size']
            del self.sd[c_id] # 刪除sd中的c_id dict 
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')
        
        n_connected = round(self.args.n_clients*self.args.frac)

        # Compute cosine & similarity matrix
        cos_matrix = np.empty(shape=(n_connected, n_connected))
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                cos_matrix[i, j] = 1 - cosine(local_weights_numpy_dist_w_anchor[i], \
                                                local_weights_numpy_dist_w_anchor[j])
        self.cos_matrices.append(cos_matrix)
        # if self.args.agg_norm == 'exp':
        #     # 做 exponential 轉換
        sim_matrix = np.exp(self.args.norm_scale * cos_matrix)
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio)) # 這邊還只是在做 FedAvg --> 但是對FedTS + FedAvg很有用欸...
        # self.sd['anchor_global_model'] = get_state_dict(self.model) # 這行在決定是否有要更新anchor model!
        # 印象中，若是更新anchor model，以FedAvg model做computation效果會很差!
        
        self.logger.print(f'anchor global model has been updated ({time.time()-st:.2f}s)')

        # 做 similarity matching
        # 改成用cosine matrix 做 update
        st = time.time()
        # 要先normalize cosine matrix
        row_sums = cos_matrix.sum(axis=1)
        normalized_cos_matrix = cos_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素
        self.normalized_cos_matrices.append(normalized_cos_matrix)
        for i in range(self.args.n_clients):
            aggr_local_model_weights = self.aggregate(local_weights, normalized_cos_matrix[i, :])
            if f'personalized_{i}' in self.sd: del self.sd[f'personalized_{i}'] 
            self.sd[f'personalized_{i}'] = {'model': aggr_local_model_weights} # 重新在sd中寫入 personalized_{c_id}，以供client讀取
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        # This will overide previous server_state.pt, so in the final, we will only have one "server_state.pt"
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model), # this is a FedAvg model，沒什麼用
            'sim_matrices': self.sim_matrices, 
            'cos_matrices' : self.cos_matrices, 
            'normalized_cos_matrices' : self.normalized_cos_matrices
        })
