#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

[user_train, user_validation, user_test, usernum, itemnum] = np.load('data/' + args.dataset + 'Partitioned.npy', allow_pickle=True)


owner = dict()
for user in range(usernum):
    for item in set(user_train[user]['produce']):
        if item in owner:
            print("multiple_creators!")
        owner[item] = user


consumed_items = defaultdict(int)
consumed_item_owners = {}
for user in range(usernum):
    consumed_item_owners[user] = set()
    for item in user_train[user]['consume']:        
        consumed_items[item] += 1
        consumed_item_owners[user].add(owner[item])

def sample_users_validation(K):
    sample = set()
    for i in range(K):
        user = np.random.randint(0, usernum)
        while len(user_validation[user]['consume']) < 1 or user in sample or user_validation[user]['consume'][0] not in consumed_items: 
            user = np.random.randint(0, usernum)
        sample.add(user)
    return sample

def sample_items(u, K):    
    sample = set()
    
    sample_list = []
    
    for i in range(K):
        item = np.random.randint(0, itemnum)
   
        while item in user_train[u]['consume'] or item in user_train[u]['produce'] or item in user_validation[u]['consume'] or item in user_test[u]['consume'] or item in sample or item not in consumed_items: 
            item = np.random.randint(0, itemnum)
        sample.add(item)
        sample_list.append(item) 
        
    return sample_list


############################# PREDICTIONS

f = open(
    'out/%s/%s/%d_%g_%g_%g.log' % (args.dataset, 'pop', 1, 1, 1, 1), 'w')    

mrr = 0
ndcg = 0
recall_1 = 0
recall_5 = 0
recall_10 = 0
recall_25 = 0
recall_50 = 0
n_users_test = 0




for u in range(usernum):
    if(len(user_test[u]['consume']) < 1):
        continue
    if(user_test[u]['consume'][0] not in consumed_items):
        continue
    
        
    n_users_test = n_users_test + 1
    
    item_sample = sample_items(u, 99)
    item_sample += user_test[u]['consume']
    
    scores = [consumed_items[item] for item in item_sample]    
    scores = np.array(scores)
    
    idx = np.argsort(-scores)            
    ranking = np.array(item_sample)[idx]
    rank  = np.where(ranking == user_test[u]['consume'])[0][0]+1           
    
    _mrr = 1.0/rank
    mrr += _mrr
    
    _ndcg = 1.0/np.log2(rank+1)
    ndcg = ndcg + _ndcg
    
    if rank == 1:
        recall_1 += 1.0
    if rank <= 5:
        recall_5 += 1.0
    if rank <= 10:
        recall_10 += 1.0
    if rank <= 25:
        recall_25 += 1.0
    if rank <= 50:
        recall_50 += 1.0
    
    f.write('%d, %d, %d, %d, %f, %f\n'%(u, len(user_train[u]['consume']), len(user_train[u]['produce']), len(consumed_item_owners[u]), _ndcg, _mrr))
    f.flush()
f.close()
   
mrr /=  n_users_test
ndcg /= n_users_test 
recall_1  /= n_users_test
recall_5  /= n_users_test
recall_10  /= n_users_test
recall_25  /= n_users_test
recall_50 /= n_users_test

f = open(
    'out/%s/%s/%d_%g_%g_%g.out' % (args.dataset, 'pop', 1, 1, 1, 1),'w') 
f.write('%f, %f, %f, %f, %f, %f, %f, %f\n' %(0, recall_1, recall_5, recall_10, recall_25, recall_50, ndcg, mrr))
f.flush()
f.close()
