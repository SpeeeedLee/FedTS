'''
Put all useful helper functions here !

'''



from collections import OrderedDict
import numpy as np 
import os
import torch
import json

def flatten_state_dict(state_dict):
    '''Convert Ordered Dict to 1D numpy Array'''
    flat_weights = [weights.flatten() for weights in state_dict.values()]
    return np.concatenate(flat_weights)

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))


def convert_tensor_to_np(state_dict):
    '''
    state_dict : Pytorch model, a dict {'conv1.weight' : tensor1, 'conv1.bias' : tensor2 ...}
    return : still a model with same key, but the value is now numpy !
    '''
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict

def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    '''查一下這是不是tensor跟numpy間轉換常用的函數、而不是作者自己寫的'''
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id) # 如果是scalar --> 一些hyperparameter --> 轉為正常的pytorch張量，然後放到gpu
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id) # 如果是vector --> model parameters --> 轉為可微分的pytorch張量，然後放到gpu
    return _state_dict

def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False):
    '''將state_dict(包含key, numpy weights pair)設置到model中、再放到指定的gpu_id上'''
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask, model=model.state_dict())
    model.load_state_dict(state_dict) # load_state_dict 是 Pytorch的預定義函數
    
def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)