#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
from collections import defaultdict

from sampler import WarpSampler
from nn import get_model

from keras.optimizers import Adagrad, Adam, SGD, RMSprop
#sudo pip install h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--alg', default='nn')
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--save', action='store_true')
parser.add_argument('--min_delta', default=0.01, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--latent_dimension', default=10, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--maximum_epochs', default=2000, type=int)
parser.add_argument('--lambda1', default=0.0, type=float)
parser.add_argument('--lambda2', default=0.0, type=float)
args = parser.parse_args()

[user_train, user_validation, user_test, usernum, itemnum] = np.load(args.dataset + 'Partitioned.npy', allow_pickle=True)

np.random.seed(7)


# count positive events
owner = dict()
oneiteration = 0

for user in range(usernum):
    oneiteration += len(user_train[user]['consume'])            
    for item in set(user_train[user]['produce']):
        if item in owner:
            print("multiple_creators!")
        owner[item] = user

for item in range(itemnum):
    if item not in owner:
        print("missing creator!")
        break

#### RAMON
for user in range(usernum):       
    user_train[user]['consume_owners'] = set()
    for item in user_train[user]['consume']:
        user_train[user]['consume_owners'].add(owner[item])
#### RAMON

consumed_items = defaultdict(int)
for user in range(usernum):
    for item in user_train[user]['consume']:        
        consumed_items[item] += 1

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
    owner_list = []
    
    for i in range(K):
        item = np.random.randint(0, itemnum)
   
        while item in user_train[u]['consume'] or item in user_train[u]['produce'] or item in user_validation[u]['consume'] or item in user_test[u]['consume'] or item in sample or item not in consumed_items: 
            item = np.random.randint(0, itemnum)
        sample.add(item)
        sample_list.append(item)
        owner_list.append(owner[item])
        
    return sample_list, owner_list

sampler = WarpSampler(user_train, user_validation, user_test, owner, usernum, itemnum, consumed_items, batch_size=args.batch_size, n_workers=1)

            
best_iter = 0   
best_valid_mrr = 1e-6
prev_valid_mrr = 1e-6

f = open('%s_%s_%d_%d_%g_%g.txt' % (args.dataset, args.alg, args.n_layers, args.latent_dimension, args.lambda1, args.lambda2),'w')

############# Architecture
hidden_layers = []
initial = 2*args.latent_dimension
factor = 1
for i in range(0,args.n_layers):
	hidden_layers.append(initial/factor)
	factor = factor*2
reg_layers = [args.lambda1]*args.n_layers

model = get_model(usernum, itemnum, mf_dim=args.latent_dimension, layers=hidden_layers, reg_layers=reg_layers, reg_emb=args.lambda2)
model.compile(optimizer=Adam(), loss='binary_crossentropy')
############
for i in range(args.maximum_epochs):    
    for _ in range(int(oneiteration / args.batch_size)):
        batch_u, batch_i, batch_j, batch_oi, batch_oj =  sampler.next_train_batch()
        
        user_input = batch_u + batch_u
        item_input = batch_i + batch_j
        owner_input = batch_oi + batch_oj
        
        labels = [1]*len(batch_u) + [0]*len(batch_u)
        
        hist = model.fit([np.array(user_input), np.array(item_input), np.array(owner_input)], np.array(labels), batch_size=args.batch_size, verbose=0, shuffle=True)
    
    
    if i % 10 == 0:
        n_sample = int(0.01*usernum)
        user_sample = sample_users_validation(n_sample)
        mrr = 0.0
        
        for u in user_sample:
            item_sample, owner_sample = sample_items(u, 99)
            item_sample += [a for a in user_validation[u]['consume']]
            owner_sample += [owner[a] for a in user_validation[u]['consume']]
            
            users = np.full(len(item_sample), u, dtype = 'int32')
                        
            scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)
            scores = scores.T[0]
            
            idx = np.argsort(-scores)            
            ranking = np.array(item_sample)[idx]
            
            rank  = np.where(ranking == user_validation[u]['consume'])[0][0]+1
            mrr += 1.0/rank     
        
        mrr /= n_sample
        pct = (mrr/best_valid_mrr - 1)
        f.write('%d, %f, %f, %f\n' % (i, mrr, best_valid_mrr, pct))
        f.flush()

        if mrr > best_valid_mrr and pct >= args.min_delta:
            best_valid_mrr = mrr            
            best_iter = i
            #model.save_weights('weights.h5')
            best_weights = model.get_weights()
        elif i >= best_iter + 50:
            f.write('break\n')
            f.flush()
            break            

        prev_valid_mrr = mrr        

f.close()

if(args.save):
    np.save('DCPRec_%s_%d_%g_%g.npy' % (args.dataset, args.latent_dimension, args.learning_rate, args.lambda1), [model.best_u, model.best_i])


############################# PREDICTIONS

f = open('%s_%s_%d_%d_%g_%g.log' % (args.dataset, args.alg, args.n_layers, args.latent_dimension, args.lambda1, args.lambda2),'w')

mrr = 0
ndcg = 0
recall_1 = 0
recall_5 = 0
recall_10 = 0
recall_25 = 0
recall_50 = 0
n_users_test = 0


model.set_weights(best_weights)

for u in range(usernum):
    if(len(user_test[u]['consume']) < 1):
        continue
    if(user_test[u]['consume'][0] not in consumed_items):
        continue
   
    n_users_test = n_users_test + 1
    
    item_sample, owner_sample = sample_items(u, 99)
    item_sample += [a for a in user_test[u]['consume']]
    owner_sample += [owner[a] for a in user_test[u]['consume']]
    
    users = np.full(len(item_sample), u, dtype = 'int32')
                        
    scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=len(item_sample), verbose=0)
    scores = scores.T[0]            
    
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
    
    f.write('%d, %d, %d, %f, %f\n'%(u, len(user_train[u]['consume']), len(user_train[u]['produce']), _ndcg, _mrr))
    f.flush()
f.close()
   
mrr /=  n_users_test
ndcg /= n_users_test 
recall_1  /= n_users_test
recall_5  /= n_users_test
recall_10  /= n_users_test
recall_25  /= n_users_test
recall_50 /= n_users_test

f = open('%s_%s_%d_%d_%g_%g.out' % (args.dataset, args.alg, args.n_layers, args.latent_dimension, args.lambda1, args.lambda2),'w')
f.write('%f, %f, %f, %f, %f, %f, %f, %f\n' %(best_valid_mrr, recall_1, recall_5, recall_10, recall_25, recall_50, ndcg, mrr))
f.flush()
f.close()

