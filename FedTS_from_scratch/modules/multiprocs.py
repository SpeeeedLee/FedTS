import os 
import atexit
import time
import torch.multiprocessing as mp
import sys
import numpy as np 
import wandb
import json
import torch

class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0] # server 使用指定的第一個gpu
        self.proc_id = os.getppid()
        print(f'main process id: {self.proc_id}')
        print(f"server, gpu_id:{self.gpu_server}")

        self.sd = mp.Manager().dict() # ParentProcess類別實例之 "sd" 是一個 multiprocessing manager dict，可以共享多個進程之間的數據!
        self.sd['is_done'] = False
        self.create_workers(Client)
        self.server = Server(args, self.sd, self.gpu_server) #### Create a Sever instance ####
        atexit.register(self.done) # 註冊self.done函數，使得其在程式要退出(有人執行sys.exit)之前會被執行

    def create_workers(self, Client):
        '''Assigning workers(children process) specific gpus and start them'''
        
        self.processes = []
        self.q = {}
        # loop through every worker
        for worker_id in range(self.args.n_workers):
            # since the first gpu id is used for server, so avialbe gpu number will need to minus 1 !
            if worker_id < len(self.gpus)-1 :
                gpu_id = self.gpus[worker_id+1] 
            else:
                gpu_id = self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            print(f'worker_id: {worker_id}, gpu_id:{gpu_id}')

            # Create a mp.Queue instance for every worker 
            self.q[worker_id] = mp.Queue()
            # Create a process for each worker
            p = mp.Process(target=WorkerProcess, args=(self.args, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            self.processes.append(p)
        
        for p in self.processes:
            # .start() to actually start the process
            p.start() # this start is a build-in function ! not the start() we define in the following !

    def start(self):
        '''This start funciton is called by main.py'''
        self.sd['is_done'] = False

        # 創建check_path, log_path 目錄，如果它們還不存在的話
        if os.path.isdir(self.args.checkpt_path) == False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)

        self.n_connected = round(self.args.n_clients*self.args.frac)
        
        # 開始迴圈每一輪
        if self.args.model == 'FedTS' or self.args.model == 'custom_sim' or self.args.model == 'FedTS_v2':
            for curr_rnd in range(self.args.matching_rounds):
                self.curr_rnd = curr_rnd
                self.updated = set() # 每一輪剛開始皆創建一個空集合 (set)，在"wait"中會加入東西，server在aggregate時會用到
                
                ''' 每一輪sample的client都不一樣。可是不同次實驗中、同一輪sample到的client是同樣的'''
                np.random.seed(self.args.seed+curr_rnd) 
                self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
                
                st = time.time()
                print(f'[main] server starts round {curr_rnd}')
                ##################################################
                self.server.on_round_begin(curr_rnd)
                ##################################################
                while len(self.selected)>0:
                    '''當 #worker < #client 時、這個while loop有可能會執行多次'''
                    _selected = []
                    for worker_id, q in self.q.items():
                        c_id = self.selected.pop(0) # 此輪選重要train的client list中pop出第一個 (pop完後，這個迴圈也就結束了)
                        _selected.append(c_id) 
                        q.put((c_id, curr_rnd)) # 在這裡把每個worker待解決的client端行為 放入 其 mp.Queue中 !!
                        if len(self.selected) == 0:
                            break

                    self.wait(curr_rnd, _selected) # 等client端把被選種的client都train好
                print(f'[main] all clients updated at round {curr_rnd}')
                ###########################################
                self.server.on_round_complete(self.updated)
                ###########################################
                print(f'[main] round {curr_rnd} server aggregation done ({time.time()-st:.2f} s)')
            ##################################################
            curr_rnd += 1
            np.random.seed(self.args.seed+curr_rnd) 
            self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
            while len(self.selected)>0:
                _selected = []
                for worker_id, q in self.q.items():
                    c_id = self.selected.pop(0) # 此輪選重要train的client list中pop出第一個 (pop完後，這個迴圈也就結束了)
                    _selected.append(c_id) 
                    q.put((c_id, curr_rnd)) # 在這裡把每個worker待解決的client端行為 放入 其 mp.Queue中 !!
                    if len(self.selected) == 0:
                        break

                self.wait(curr_rnd, _selected) # 等client端把被選種的client都train好
            print(f'[main] all clients finish final local tuning')
            ###########################################
        
        else:
            for curr_rnd in range(self.args.n_rnds):
                self.curr_rnd = curr_rnd
                self.updated = set() # 每一輪剛開始皆創建一個空集合 (set)，在"wait"中會加入東西，server在aggregate時會用到
                
                ''' 每一輪sample的client都不一樣。可是不同次實驗中、同一輪sample到的client是同樣的'''
                np.random.seed(self.args.seed+curr_rnd) 
                self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
                
                st = time.time()
                print(f'[main] server starts round {curr_rnd}')
                ##################################################
                self.server.on_round_begin(curr_rnd)
                ##################################################
                while len(self.selected)>0:
                    '''當 #worker < #client 時、這個while loop有可能會執行多次'''
                    _selected = []
                    for worker_id, q in self.q.items():
                        c_id = self.selected.pop(0) # 此輪選重要train的client list中pop出第一個 (pop完後，這個迴圈也就結束了)
                        _selected.append(c_id) 
                        q.put((c_id, curr_rnd)) # 在這裡把每個worker待解決的client端行為 放入 其 mp.Queue中 !!
                        if len(self.selected) == 0:
                            break

                    self.wait(curr_rnd, _selected) # 等client端把被選種的client都train好
                print(f'[main] all clients updated at round {curr_rnd}')
                    
                ###########################################
                self.server.on_round_complete(self.updated)
                ###########################################
                print(f'[main] round {curr_rnd} server aggregation done ({time.time()-st:.2f} s)')

        self.sd['is_done'] = True # only when the sd["is_done"] == True can the main sys be exited
        for worker_id, q in self.q.items():
            q.put(None) # None 本身也是一個東西，讓每個worker的Queue可以get到這個None並跳入下一個迴圈，並終止。
        print('[main] server done')

        if self.args.model == 'FedTS' or self.args.model == 'custom_sim' or self.args.model == 'FedTS_v2':
            self.wandb_FedTS()
        else:            
            self.wandb()

        sys.exit() # 中斷父進程，這會觸發 atexit 註冊的函數

    def wandb_FedTS(self):
        print("saving logs to wandb ......")
        for c_id in range(self.args.n_clients):
            log_file_path = os.path.join(self.args.log_path, f"client_{c_id}.txt")
            with open(log_file_path, 'r') as file:
                log_data = json.load(file)
                epochs_in_FL = log_data["log"]["epochs_in_FL"]
                print(f"Client {c_id} : {epochs_in_FL}")
        
        ''' 每一個client在每個FL Matching Round中都執行了幾個epochs?'''
        epochs_in_FL_record_matrix = np.empty(shape=(self.args.n_clients, self.args.matching_rounds+1))
        for client_id in range(self.args.n_clients):
            log_file_path = os.path.join(self.args.log_path, f"client_{client_id}.txt")
            with open(log_file_path, 'r') as file:
                log_data = json.load(file)
            epochs_in_FL_record_matrix[client_id, :] =  log_data["log"]["epochs_in_FL"]
        max_epochs_in_matching_rounds = np.amax(epochs_in_FL_record_matrix, axis=0).tolist()
        min_epochs_in_matching_rounds = np.amin(epochs_in_FL_record_matrix, axis=0).tolist()
        
        '''
        將每個client在FL Mathing Round中、沒有trainnig到的epoch部分補上前一個的vlaue、所以會有平行直線、以利視覺化對比
        最後收尾會有些client比較長、有些client比較短、
            --> 幫比較短的cliet，延長其最後Accuracy、loss到self.args.epoch_limit + self.args.matching_rounds + 1 + (max_epoch_round - min_epoch_round)!
        而平均值就直接用這個計算，反正都是看最後一個FL round中找最大值，因此不會錯的 
        '''
        n_epochs = int(self.args.epoch_limit + self.args.matching_rounds + 1 + (max_epochs_in_matching_rounds[0] - min_epochs_in_matching_rounds[0]))# 看多少個epoch的test accuracy
        total_val_acc  = [0]*n_epochs 
        total_val_lss = [0]*n_epochs
        total_test_acc = [0]*n_epochs
        total_test_lss = [0]*n_epochs
        for client_id in range(self.args.n_clients):
            # 初始化 wandb 運行
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name=f"client_{client_id}") 
            log_file_path = os.path.join(self.args.log_path, f"client_{client_id}.txt")
            with open(log_file_path, 'r') as file:
                log_data = json.load(file)
            epochs_in_FL = log_data["log"]["epochs_in_FL"]
            val_acc = []
            val_lss = []
            test_lss = []
            test_acc = []
            # 紀錄最後各自train到epoch limit之前的數據
            for matching_round in range(self.args.matching_rounds):
                client_epochs = epochs_in_FL[matching_round] + 1 # "+1" is for evaluating the model sent back from server
                max_epoch = max_epochs_in_matching_rounds[matching_round] + 1
                # for val_acc
                val_acc.extend(log_data["log"]["ep_local_val_acc"][:client_epochs])
                last_val_acc = log_data["log"]["ep_local_val_acc"][client_epochs-1]
                val_acc.extend([last_val_acc]*(int(max_epoch-client_epochs)))
                log_data["log"]["ep_local_val_acc"] = log_data["log"]["ep_local_val_acc"][client_epochs:]
                # for val_lss
                val_lss.extend(log_data["log"]["ep_local_val_lss"][:client_epochs])
                last_val_lss = log_data["log"]["ep_local_val_lss"][client_epochs-1]
                val_lss.extend([last_val_lss]*(int(max_epoch-client_epochs)))
                log_data["log"]["ep_local_val_lss"] = log_data["log"]["ep_local_val_lss"][client_epochs:]
                # for test_acc
                test_acc.extend(log_data["log"]["ep_local_test_acc"][:client_epochs])
                last_test_acc = log_data["log"]["ep_local_test_acc"][client_epochs-1]
                test_acc.extend([last_test_acc]*(int(max_epoch-client_epochs)))
                log_data["log"]["ep_local_test_acc"] = log_data["log"]["ep_local_test_acc"][client_epochs:]
                # for test_lss
                test_lss.extend(log_data["log"]["ep_local_test_lss"][:client_epochs])
                last_test_lss = log_data["log"]["ep_local_test_lss"][client_epochs-1]
                test_lss.extend([last_test_lss]*(int(max_epoch-client_epochs)))
                log_data["log"]["ep_local_test_lss"] = log_data["log"]["ep_local_test_lss"][client_epochs:]
            
            # 紀錄最後一次similarity matching後的數據
            curr_records = len(val_acc)
            final_extend = n_epochs - curr_records
            # for val_acc
            val_acc.extend(log_data["log"]["ep_local_val_acc"]) # 這裡面的第一筆數據應該會是最後一次similarity matching後回傳的model之evaluation
            last_val_acc = log_data["log"]["ep_local_val_acc"][-1]
            val_acc.extend([last_val_acc]*(int(final_extend)))
            # for val_lss
            val_lss.extend(log_data["log"]["ep_local_val_lss"])
            last_val_lss = log_data["log"]["ep_local_val_lss"][-1]
            val_lss.extend([last_val_lss]*(int(final_extend)))
            # for test_acc
            test_acc.extend(log_data["log"]["ep_local_test_acc"])
            last_test_acc = log_data["log"]["ep_local_test_acc"][-1]
            test_acc.extend([last_test_acc]*(int(final_extend)))
            # for test_lss
            test_lss.extend(log_data["log"]["ep_local_test_lss"])
            last_test_lss = log_data["log"]["ep_local_test_lss"][-1]
            test_lss.extend([last_test_lss]*(int(final_extend)))

            # 記錄到wandb
            for epoch in range(n_epochs):
                wandb.log({"Epoch": epoch + 1, "Validation Accuracy": val_acc[epoch], 
                        "Validation Loss": val_lss[epoch], "Test Loss": test_lss[epoch], 
                        "Test Accuracy": test_acc[epoch]})
                
            total_val_acc = [a + b for a, b in zip(total_val_acc, val_acc)]
            total_val_lss = [a + b for a, b in zip(total_val_lss, val_lss)]
            total_test_acc = [a + b for a, b in zip(total_test_acc, test_acc)]
            total_test_lss = [a + b for a, b in zip(total_test_lss, test_lss)]

            # 結束當前客戶端之wandb運行
            wandb.finish()


        # 計算平均值
        mean_val_acc = [x / self.args.n_clients for x in total_val_acc] 
        mean_val_lss = [x / self.args.n_clients for x in total_val_lss] 
        mean_test_acc = [x / self.args.n_clients for x in total_test_acc] 
        mean_test_lss = [x / self.args.n_clients for x in total_test_lss]

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_val_acc")
        for epoch in range(n_epochs):
            wandb.log({"Average Validation Accuracy": mean_val_acc[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_val_lss")
        for epoch in range(n_epochs):
            wandb.log({"Average Validation Loss": mean_val_lss[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_test_acc")
        for epoch in range(n_epochs):
            wandb.log({"Average Test Accuracy": mean_test_acc[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_test_lss")
        for epoch in range(n_epochs):
            wandb.log({"Average Test Loss": mean_test_lss[epoch]})
        wandb.finish()


        print("saving cosine matrix and similarity matrix in every FL round to wandb...")
        file_path = os.path.join(self.args.checkpt_path, f"server_state.pt")
        saved_state = torch.load(file_path) 
        cos_matrices = saved_state['cos_matrices']
        sim_matrices = saved_state['sim_matrices']
        normalized_cos_matrices = saved_state['normalized_cos_matrices']
        graph_matrices = saved_state['graph_matrices']
    
        fl_round = 0
        labels_row = [f"c_{i}" for i in range(self.args.n_clients)]
        labels_column = [f"c_{i}" for i in range(self.args.n_clients)] + ['mim'] + ['max']
        wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_cos")
        for cos_matrix in cos_matrices:
            fl_round += 1
            # 創建並上傳熱力圖
            # 先替cos_matrix 加兩個column，全0、全1
            zeros_column = np.zeros((cos_matrix.shape[0], 1))    # 創建一個全0的行
            ones_column = np.ones((cos_matrix.shape[0], 1))      # 創建一個全1的行
            cos_matrix = np.hstack((cos_matrix, zeros_column, ones_column))
            wandb.log({f"cosine_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = cos_matrix,  show_text=True)})
        wandb.finish()

        fl_round = 0
        wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_normalized_cos")
        for normalized_cos_matrix in normalized_cos_matrices:
            fl_round += 1
            # 創建並上傳熱力圖
            # 先替cos_matrix 加兩個column，全0、全1
            zeros_column = np.zeros((normalized_cos_matrix.shape[0], 1))    # 創建一個全0的行
            ones_column = np.ones((normalized_cos_matrix.shape[0], 1))      # 創建一個全1的行
            normalized_cos_matrix = np.hstack((normalized_cos_matrix, zeros_column, ones_column))
            wandb.log({f"normalized_cos_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = normalized_cos_matrix,  show_text=True)})
        wandb.finish()
        
        fl_round = 0
        wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_sim")
        for sim_matrix in sim_matrices:
            fl_round += 1
            # 創建並上傳熱力圖
            # 先替cos_matrix 加兩個column，全0、全1
            zeros_column = np.zeros((sim_matrix.shape[0], 1))    # 創建一個全0的行
            ones_column = np.ones((sim_matrix.shape[0], 1))      # 創建一個全1的行
            sim_matrix = np.hstack((sim_matrix, zeros_column, ones_column))
            wandb.log({f"similarity_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = sim_matrix,  show_text=True)})
        wandb.finish()

        fl_round = 0
        wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_graph")
        for graph_matrix in graph_matrices:
            fl_round += 1
            # 創建並上傳熱力圖
            # 先替cos_matrix 加兩個column，全0、全1
            zeros_column = np.zeros((graph_matrix.shape[0], 1))    # 創建一個全0的行
            ones_column = np.ones((graph_matrix.shape[0], 1))      # 創建一個全1的行
            graph_matrix = np.hstack((graph_matrix, zeros_column, ones_column))
            wandb.log({f"graph_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = graph_matrix,  show_text=True)})
        wandb.finish()
        


    def wandb(self):
        print("saving logs to wandb ......")

        log_file_path = os.path.join(self.args.log_path, f"client_{0}.txt")
        with open(log_file_path, 'r') as file:
            log_data = json.load(file)
            n_epochs = len(log_data["log"]["ep_local_val_acc"])
        total_val_acc  = [0]*n_epochs 
        total_val_lss = [0]*n_epochs
        total_test_acc = [0]*n_epochs
        total_test_lss = [0]*n_epochs

        for client_id in range(self.args.n_clients):
            # 初始化 wandb 運行
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name=f"client_{client_id}")                
            
            log_file_path = os.path.join(self.args.log_path, f"client_{client_id}.txt")
            with open(log_file_path, 'r') as file:
                log_data = json.load(file)
            for epoch in range(n_epochs):
                val_acc = log_data["log"]["ep_local_val_acc"][epoch]
                val_lss = log_data["log"]["ep_local_val_lss"][epoch]
                test_lss = log_data["log"]["ep_local_test_lss"][epoch]
                test_acc = log_data["log"]["ep_local_test_acc"][epoch]

                total_val_acc[epoch] += val_acc 
                total_val_lss[epoch] += val_lss
                total_test_acc[epoch] += test_acc
                total_test_lss[epoch] += test_lss

                # 記錄到wandb
                wandb.log({"Epoch": epoch + 1, "Validation Accuracy": val_acc, 
                        "Validation Loss": val_lss, "Test Loss": test_lss, 
                        "Test Accuracy": test_acc})

            # 結束當前客戶端之wandb運行
            wandb.finish()

        # 計算平均值
        mean_val_acc = [x / self.args.n_clients for x in total_val_acc] 
        mean_val_lss = [x / self.args.n_clients for x in total_val_lss] 
        mean_test_acc = [x / self.args.n_clients for x in total_test_acc] 
        mean_test_lss = [x / self.args.n_clients for x in total_test_lss]

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_val_acc")
        for epoch in range(n_epochs):
            wandb.log({"Average Validation Accuracy": mean_val_acc[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_val_lss")
        for epoch in range(n_epochs):
            wandb.log({"Average Validation Loss": mean_val_lss[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_test_acc")
        for epoch in range(n_epochs):
            wandb.log({"Average Test Accuracy": mean_test_acc[epoch]})
        wandb.finish()

        wandb.init(project = self.args.exp_name, entity="speeeedlee", name="average_test_lss")
        for epoch in range(n_epochs):
            wandb.log({"Average Test Loss": mean_test_lss[epoch]})
        wandb.finish()

        if self.args.model == 'similarity' or self.args.model == 'local' or self.args.model == 'fedavg' or self.args.model == 'FedTS':
            print("saving cosine matrix and similarity matrix in every FL round to wandb...")
            file_path = os.path.join(self.args.checkpt_path, f"server_state.pt")
            saved_state = torch.load(file_path) 
            cos_matrices = saved_state['cos_matrices']
            sim_matrices = saved_state['sim_matrices']
            normalized_cos_matrices = saved_state['normalized_cos_matrices']
            graph_matrices = saved_state['graph_matrices']
        
            fl_round = 0
            labels_row = [f"c_{i}" for i in range(self.args.n_clients)]
            labels_column = [f"c_{i}" for i in range(self.args.n_clients)] + ['mim'] + ['max']
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_cos")
            for cos_matrix in cos_matrices:
                fl_round += 1
                # 創建並上傳熱力圖
                # 先替cos_matrix 加兩個column，全0、全1
                zeros_column = np.zeros((cos_matrix.shape[0], 1))    # 創建一個全0的行
                ones_column = np.ones((cos_matrix.shape[0], 1))      # 創建一個全1的行
                cos_matrix = np.hstack((cos_matrix, zeros_column, ones_column))
                wandb.log({f"cosine_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = cos_matrix,  show_text=True)})
            wandb.finish()

            fl_round = 0
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_normalized_cos")
            for normalized_cos_matrix in normalized_cos_matrices:
                fl_round += 1
                # 創建並上傳熱力圖
                # 先替cos_matrix 加兩個column，全0、全1
                zeros_column = np.zeros((normalized_cos_matrix.shape[0], 1))    # 創建一個全0的行
                ones_column = np.ones((normalized_cos_matrix.shape[0], 1))      # 創建一個全1的行
                normalized_cos_matrix = np.hstack((normalized_cos_matrix, zeros_column, ones_column))
                wandb.log({f"normalized_cos_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = normalized_cos_matrix,  show_text=True)})
            wandb.finish()
            
            fl_round = 0
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_sim")
            for sim_matrix in sim_matrices:
                fl_round += 1
                # 創建並上傳熱力圖
                # 先替cos_matrix 加兩個column，全0、全1
                zeros_column = np.zeros((sim_matrix.shape[0], 1))    # 創建一個全0的行
                ones_column = np.ones((sim_matrix.shape[0], 1))      # 創建一個全1的行
                sim_matrix = np.hstack((sim_matrix, zeros_column, ones_column))
                wandb.log({f"similarity_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = sim_matrix,  show_text=True)})
            wandb.finish()

            fl_round = 0
            wandb.init(project = self.args.exp_name, entity="speeeedlee", name = "heatmap_graph")
            for graph_matrix in graph_matrices:
                fl_round += 1
                # 創建並上傳熱力圖
                # 先替cos_matrix 加兩個column，全0、全1
                zeros_column = np.zeros((graph_matrix.shape[0], 1))    # 創建一個全0的行
                ones_column = np.ones((graph_matrix.shape[0], 1))      # 創建一個全1的行
                graph_matrix = np.hstack((graph_matrix, zeros_column, ones_column))
                wandb.log({f"graph_heatmap_{fl_round}_": wandb.plots.HeatMap(x_labels = labels_column, y_labels = labels_row, matrix_values = graph_matrix,  show_text=True)})
            wandb.finish()

            


        return True



    def wait(self, curr_rnd, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if not c_id in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(0.1)

    def done(self):
        for p in self.processes:
            p.join()
        print('[main] All children have joined. Destroying main process ...')





class WorkerProcess:
    def __init__(self, args, worker_id, gpu_id, q, sd, Client):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False
        self.pid = os.getpid()
        print(f"A worker is created ! PID : {self.pid}")
        self.client = Client(self.args, self.worker_id, self.gpu_id, self.sd) #這裡才真的替每個worker實例化一個client instance !
        self.listen() 
        # 這裡非常關鍵...，每個worker的target都被指定到這個WorkerProcess class(也就是會實例化一個object)；
        # 而實例化過程一定會執行__init__()
        # __init__中又叫他去執行listen() !

    def listen(self):
        while not self.sd['is_done']:
            mesg = self.q.get() # 從q中獲取並移除一個項目，若沒有東西，會一直等。
            if not mesg == None:
                client_id, curr_rnd = mesg 
                ##################################
                self.client.switch_state(client_id) # 這裡會包含呼叫init_state 或 load_state的函數，client就可以追蹤自己目前的optimizer了! 好強大
                self.client.on_receive_message(curr_rnd)
                self.client.on_round_begin()
                self.client.save_state()
                ##################################
            time.sleep(1.0) # every worker check every one sec whether a new work is needed, unitl sd['is_done'] == True

        print('[main] Terminating worker processes ... ')
        sys.exit() # 中斷子進程，不影響父進程





