import os
from datetime import datetime

from my_parser import myParser
from modules.multiprocs import ParentProcess

def main(args):

    args = set_config(args)

    if args.model == 'fedavg':    
        print(f"Using {args.model} Algotihm")
        from models.fedavg.server import Server
        from models.fedavg.client import Client

    elif args.model == 'similarity':
        print(f"Using {args.model} Algorithm")
        from models.similarity.server import Server
        from models.similarity.client import Client

    elif args.model == 'local':
        print(f"Using {args.model} Algorithm")
        from models.local.server import Server
        from models.local.client import Client

    elif args.model == 'FedTS':
        print(f"Using {args.model} Algotihm")
        from models.FedTS.server import Server
        from models.FedTS.client import Client

    else:
        print(f'incorrect model was given: {args.model}')
        os._exit(0)
    pp = ParentProcess(args, Server, Client) 
    # 這會創造(#workers個子進程)，並且每個子進程都在"listen"，並且都擁有一個Client instance
    # 還會創造一個Server instance
    
    
    pp.start() 
    # 這會開始迴圈每一輪FL訓練
    '''
    for :
        Server : "on_round_begin"
        丟給workers一些task去做 (利用Queue、且每個Worker每隔一秒都一直在listen)
        "wait" --> 等所有client都train完 (所有workers都做完事情)
        Server : "on_round_complete"
    '''


def set_config(args):

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    if args.model == 'FedTS': 
        args.exp_name = f'{args.dataset}_{args.mode}_c_{args.n_clients}_{args.model}_cos{args.ready_cosine}_r{args.matching_rounds}'
    else:
        args.exp_name = f'{args.dataset}_{args.mode}_c_{args.n_clients}_{args.model}_e{args.n_eps}_r{args.n_rnds}'
    args.data_path = f'{args.base_path}/datasets'
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    # args.weight_decay = 1e-6
    # args.base_lr = 1e-3


    if args.dataset == 'cifar10':
        args.n_clss = 10

    start_time =  datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"start time : {start_time}")

    return args

if __name__ == '__main__':
    main(myParser().parse())
