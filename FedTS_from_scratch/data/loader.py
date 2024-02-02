from misc.utils import *

class DataLoader:
    '''
    這是一個自定義的DataLoader class
    其instance中會有個實例是真正的pyTorch DataLoader ....
    '''
    def __init__(self, args):
        self.args = args
        self.n_workers = 1 # 如果設多一點呢? 雙層multi process ?
        self.client_id = None #  一個worker第一次被創建時，其DataLoader中的cid是None

        from torch.utils.data import DataLoader # 修改成適合影像資料集的data loader
        self.DataLoader = DataLoader 
        # 這不是在創建一個實例，只是在pass入這個class而已!
        # 可以想成是別名、指針的概念
        # self.DataLoader = DataLoader()，這樣才是真的在實例化 


    def switch(self, client_id):
        if not self.client_id == client_id: # 原先client_id都會被設為None，所以第一次跑switch函數時，一定會進入if
            self.client_id = client_id
            self.partition_train = get_data(self.args, client_id=client_id)[0]
            self.partition_val = get_data(self.args, client_id=client_id)[1]
            self.partition_test = get_data(self.args, client_id=client_id)[2]
            # pa_loader stands for partition dataset loader 
            self.pa_loader_train = self.DataLoader(dataset=self.partition_train, batch_size=32,
                shuffle=True, num_workers=self.n_workers, pin_memory=False) # pin_memory = True 會更快，但是需要更多cpu內存
            self.pa_loader_val = self.DataLoader(dataset=self.partition_val, batch_size=32,
                shuffle=False, num_workers=self.n_workers, pin_memory=False)
            self.pa_loader_test = self.DataLoader(dataset=self.partition_test, batch_size=32,
                shuffle=False, num_workers=self.n_workers, pin_memory=False)
            '''
            print(len(self.pa_loader_train))
            print(len(self.pa_loader_val))
            
            # 获取第一个 batch
            first_batch = next(iter(self.pa_loader_train))

            # 假设图像和标签分别存储在 batch 的两个部分
            images, labels = first_batch

            # 选择第一个数据项
            first_image = images[0]
            first_label = labels[0]
            print(first_image.size(), first_label)
            '''



def get_data(args, client_id):
    # 事先會先generate data，那些data已經存在對應路徑中的.pt檔案中了! 
    # (.pt file 不只可以拿來存模型，也可拿來存data)
    data = torch_load(
        args.data_path, 
        f'{args.dataset}/{args.mode}/{args.n_clients}/client_{client_id}_data.pt'
    )
    return [data['train'], data['val'], data['test']]