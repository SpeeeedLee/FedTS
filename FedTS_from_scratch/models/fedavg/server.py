import time
import numpy as np
from scipy.spatial.distance import cosine


from modules.federated import ServerModule
from models.nets import *
from misc.utils import *


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
        self.model = CNN().cuda(self.gpu_id)
        # store the initail model to the sd
        self.sd['initial_global_model'] = get_state_dict(self.model) 
        # 在FedAvg中不會被用到，但還是可以拿來計算我們有興趣的cosine、similarity matrix
        self.sim_matrices = []
        self.cos_matrices = []
        self.normalized_cos_matrices = []
    
    def on_round_begin(self, curr_rnd):
        '''
        update current model weights to the sd['global']
        if it is the first round, then the initial CNN model weights will be used
        ** Only one server instance, the server object's weights will evolve through rounds !
        '''
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        '''Run the update function & save the resulting global model to checkpoint path'''
        self.update(updated)
        self.save_state()

    def update(self, updated):
        ''' Upload all clients model parameters and train size from sd'''
        st = time.time()
        local_weights = [None]*self.args.n_clients
        local_weights_numpy = [None]*self.args.n_clients
        local_weights_numpy_dist_w_initial = [None]*self.args.n_clients
        local_train_sizes = [None]*self.args.n_clients
        for c_id in updated:
            local_weights[c_id] = self.sd[c_id]['model'].copy() # 加入一個client的weights
            # 轉換成1D numpy array!
            local_weights_numpy[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy())
            # 利用向量之"末減初"，來計算client current model 與 initial global model 的差向量
            local_weights_numpy_dist_w_initial[c_id] = flatten_state_dict(self.sd[c_id]['model'].copy()) \
                                                        - flatten_state_dict(self.sd['initial_global_model'].copy()) 
            local_train_sizes[c_id] = self.sd[c_id]['train_size']
            del self.sd[c_id] ### important ###
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        '''Calculate the cosine and similarity matrix, although it should be all one since constantly average'''
        n_connected = round(self.args.n_clients*self.args.frac)

        # Compute cosine & similarity matrix
        cos_matrix = np.empty(shape=(n_connected, n_connected))
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                cos_matrix[i, j] = 1 - cosine(local_weights_numpy_dist_w_initial[i], \
                                                local_weights_numpy_dist_w_initial[j])
        self.cos_matrices.append(cos_matrix)
        # if self.args.agg_norm == 'exp':
        #     # 做 exponential 轉換
        sim_matrix = np.exp(self.args.norm_scale * cos_matrix)
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素

        self.sim_matrices.append(sim_matrix)

        # 計算normalize cosine matrix
        row_sums = cos_matrix.sum(axis=1)
        normalized_cos_matrix = cos_matrix / row_sums[:, np.newaxis] # np.newaxis 只是加一個維度，並不放入任何元素
        self.normalized_cos_matrices.append(normalized_cos_matrix)

        '''Aggregate all the clients' model'''
        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio)) # this will put the aggregated new model into self.model
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')   

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }
    
    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices, 
            'cos_matrices' : self.cos_matrices, 
            'normalized_cos_matrices' : self.normalized_cos_matrices
        })