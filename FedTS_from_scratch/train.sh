# Record important script

##### Pathological Pair, cifar10 #####

# FedTS
python main.py --model FedTS --dataset cifar10 --mode pathological_pair --n-clients 10 --ready-cosine 0.99 --matching-rounds 1 --epoch-limit 200

# FedTS_v2
python main.py --model FedTS_v2 --dataset cifar10 --mode pathological_pair --n-clients 10 --ready-cosine 0.99 --epoch-limit 200 --matching-rounds (要用估的，估高一些，至少要使大家會train超過200 eps)
# 不對 應該改成每次reach cosine criteria 就做similarity matching，沒有epoch limit，而是限制matching_rounds；但是最後結果必須看200+matching_round以內的最高accuracy

# similarity
python main.py --model similarity --dataset cifar10 --mode pathological_pair --n-clients 10 --n-rnds 10 --n-eps 20

# custom_sim
python main.py --model custom_sim --dataset cifar10 --mode pathological_pair --n-clients 10 --custom-epochs-list 10 20 --epoch-limit 200

# FedAvg
python main.py --model fedavg --dataset cifar10 --mode pathological_pair --n-clients 10 --n-eps 4 --n-rnds 50 


# local 
python main.py --model local --dataset cifar10 --mode pathological_pair --n-clients 10 --n-eps 10 --n-rnds 20 # 其實不同設法只會變動多久做一次similarity matching檢查，都還是在做local training



##### cluster_dirichlet_v1, cifar100 #####

# local 
python main.py --model local --dataset cifar100 --mode cluster_dirichlet_v1 --n-clients 10 --n-eps 4 --n-rnds 50

# FedAvg
python main.py --model fedavg --dataset cifar100 --mode cluster_dirichlet_v1 --n-clients 10 --n-eps 4 --n-rnds 50 

# similarity
python main.py --model similarity --dataset cifar100 --mode cluster_dirichlet_v1 --n-clients 10 --n-eps 4 --n-rnds 50

# FedTS
python main.py --model FedTS --dataset cifar100 --mode cluster_dirichlet_v1 --n-clients 10 --ready-cosine 0.98 --matching-rounds 1 --epoch-limit 200

#### Generate Data ####
cd data
python generators/cluster_dirichlet_v1.py