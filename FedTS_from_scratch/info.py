'''
如果先用全部的data train好幾輪 train到最好、
然後把這個model當作anchor model
其他所有clients、在不同FL round之model、都用cosine similarity來跟anchor model看變化
是不是很好 ??

應該是 !! 因為用所有data、就是最佳解、也是FL的初衷
'''



'''
sd : 應該是包含 server, clients 的 models 的一個雙層dict :
{'global' : {'model': ...numpy weights... }, 
    '1' : {'model' : ...numpy weights ..., 'train_size' : 1000}, 
    '2' : {'model' : ...numpy weights ..., 'train_size' : 999}, 
    ...
}

'''


'''
PID = Process ID
PPID = Parent Process ID (當前進程之父進程的ID)


'''


'''
每一個client會keep一個 {client_id}_state.pt、其中有以下資訊:
        {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        }




'''