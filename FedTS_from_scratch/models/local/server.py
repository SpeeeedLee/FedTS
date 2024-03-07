import time
import numpy as np

from scipy.spatial.distance import cosine

from misc.utils import *
from models.nets import CNN, CNN_100
from modules.federated import ServerModule

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        if self.args.dataset == 'cifar100':
            self.model = CNN_100().cuda(gpu_server)
        elif self.args.dataset == 'cifar10':
            initial_weights = torch.load('../initial_model/CNN/model_1.pkl')
            self.model = CNN().cuda(gpu_server)
            self.model.load_state_dict(initial_weights)
        else:
            raise NotImplementedError('還沒Build對應的model')
        # store the initail model to the sd
        self.sd['initial_global_model'] = get_state_dict(self.model)
        self.sim_matrices = []
        self.cos_matrices = []
        self.normalized_cos_matrices = []

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()
        # 其實每個clients起始時候的model都是一模一樣的!!! 因為都是拿server端一開始initialize的一個model過去

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = [None]*self.args.n_clients
        local_weights_numpy = [None]*self.args.n_clients
        local_weights_numpy_dist_w_initial = [None]*self.args.n_clients
        local_train_sizes = [None]*self.args.n_clients
        for c_id in updated:
            local_weights[c_id] = self.sd[c_id]['model'].copy() # 加入一個client的weights
            local_weights_numpy[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy()) # 轉換成1D numpy array!
            # 利用向量之"末減初"，來計算client current model 與 initial global model 的差向量
            local_weights_numpy_dist_w_initial[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy()) \
                                                        - flatten_state_dict(self.sd['initial_global_model'].copy()) 
            local_train_sizes[c_id] = self.sd[c_id]['train_size']
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        n_connected = round(self.args.n_clients*self.args.frac)

        psuedo_sim_matrix = np.eye(n_connected)

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

        st = time.time()
        for i in range(self.args.n_clients):
            aggr_local_model_weights = self.aggregate(local_weights, psuedo_sim_matrix[i, :]) # 其實就是回傳每個clients原本的model而已
            if f'personalized_{i}' in self.sd: del self.sd[f'personalized_{i}']
            self.sd[f'personalized_{i}'] = {'model': aggr_local_model_weights}

        # Really compute the cosine matrix & similarity matrix
        cos_matrix = np.empty(shape=(n_connected, n_connected))
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                cos_matrix[i, j] = 1 - cosine(local_weights_numpy_dist_w_initial[i], \
                                                local_weights_numpy_dist_w_initial[j])
        '''
        ##### 重新調整cos_matrix的維度順序 #####
        new_cos_matrix = np.empty_like(cos_matrix)
        # 遍歷 updated 列表，並將 cos_matrix 的列重新排列到 new_cos_matrix
        for old_index, new_index in enumerate(updated):
            new_cos_matrix[new_index] = cos_matrix[old_index]
        '''
        self.cos_matrices.append(cos_matrix)
        ################################################################   
        # if self.args.agg_norm == 'exp':
            # 做 exponential 轉換
        sim_matrix = np.exp(self.args.norm_scale * cos_matrix)
        row_sums = sim_matrix.sum(axis=1) # a 1D array, each element is a row sum
        sim_matrix = sim_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素
        self.sim_matrices.append(sim_matrix)
        
        row_sums = cos_matrix.sum(axis=1)
        normalized_cos_matrix = cos_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素
        self.normalized_cos_matrices.append(normalized_cos_matrix)

        self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'cos_matrices' : self.cos_matrices, 
            'normalized_cos_matrices' : self.normalized_cos_matrices
        })
