### Below is the code that I write for Dirichlet Partition a long time ago ###
import matplotlib.pyplot as plt
import numpy as np

import os

'''
in shorts: 如果有10個label，必須有剛好1/10的client要第一個label、剛好1/10的client要第二個label...
也就是說如果我alpha要超級小的話，我就直接上帝視角分就好了...用alpha去調沒有意義，也到不了那裏

假設有100個clients，
從Dirichlet中抽distribution，
當alpha設得很低的時候，probability會是[0,1,0,0,0,0,0,0,0,0]

因此在此情況下，本來就無法保證抽到的100個client有10個會在label 1有1；10個在label 2有1....
所以很有可能某個client機率值為1所要的那個label，會被其他client抽完，
若遇到此情況，我只能random給這個client剩餘的label了

--> 要避免alpha值過低到出現[0,1,0,0,0,0,0,0,0,0]....，否則有可能降低non iid (就算client數量比class數量低也沒用)

10類別的情況下，0.01就很極限了；
0.1剛好
而且再越變越小也沒有用，probability distribution都已經是1了...

ex:
超過10個clients只想抽類別1，
當類别1被抽完後，這些clients沒有其他類別的機率，一定得用一些random給予的方法，就失去了一開始Dirichlet的性質

'''


def dirichlet_split_noniid(train_labels, alpha, NUM_CLIENTS, seed = 42):

    np.random.seed(seed)
    train_labels = np.array(train_labels) # convert list to numpy, so that below code can run successfully
    n_classes = train_labels.max()+1
    n_data = len(train_labels)


    class_idcs = [[] for _ in range(n_classes)]

    for index in range(len(train_labels)):
        y = train_labels[index][0]
        class_idcs[y].append(index)    
    
    for lst in class_idcs:
        np.random.shuffle(lst)

    original_class_count = np.array( [len(class_idcs[i]) for i in range(n_classes)])
    remain_class_count = original_class_count
    print(f"original class count in dataset:{original_class_count}")

    diri_prior = np.random.uniform(size=n_classes)
    p = [x * alpha for x in diri_prior]
    rng = np.random.default_rng()
    client_label_prob = rng.dirichlet(p, size = NUM_CLIENTS)

    '''
    client_label_prob += 1e-15
    row_sums = client_label_prob.sum(axis=1, keepdims=True)                
    client_label_prob = client_label_prob / row_sums
    '''

    label_distribution = np.zeros((NUM_CLIENTS, n_classes))

    # 進行抽取的過程
    items = [labels for labels in range(n_classes)]

    n_client_data = int(np.floor(n_data /NUM_CLIENTS)) #500


    client_number = 0
    for i in range(len(train_labels)):
        # print(f"開始第幾次抽取:{i+1}")
        pick_label = np.random.choice(items , 1 , p = client_label_prob[client_number,:])

        label_distribution[client_number, pick_label] += 1

        remain_class_count[pick_label] -= 1 

        if (i == len(train_labels)-1):
            break

        if remain_class_count[pick_label] <= 0:
            column_to_remove = items.index(pick_label)
            items.remove(pick_label)    
            client_label_prob = np.delete(client_label_prob, column_to_remove, axis = 1)
            row_sums = client_label_prob.sum(axis=1, keepdims=True)     
            for x in range(NUM_CLIENTS):
                if(row_sums[x] == 0):
                    p_new = [1 / len(items)] * len(items)
                    p_new = [g * alpha for g in p_new]
                    client_label_prob[x] = rng.dirichlet(p_new, size = 1)
                else:
                    client_label_prob[x,:] = client_label_prob[x,:] / row_sums[x]
        

        client_number += 1
        if client_number == NUM_CLIENTS:
             client_number = 0


    client_idcs = [[] for _ in range(NUM_CLIENTS)] 
    for i in range(NUM_CLIENTS):
        for j in range(n_classes):
            end_index = int(label_distribution[i,j])
            client_idcs[i].extend(class_idcs[j][: end_index])
            class_idcs[j] = class_idcs[j][end_index :]

    return client_idcs








