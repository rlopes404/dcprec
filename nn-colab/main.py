#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import keras

from collections import defaultdict
from sampler import WarpSampler
from nn import get_model_pair, get_model_triple, get_model_pos_neg
#sudo pip install h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--alg', default='pair', choices=['pair', 'triple', 'posneg'])
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--save', action='store_true')
parser.add_argument('--min_delta', default=0.0, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--emb_dim', default=10, type=int)
parser.add_argument('--n_neurons', default=10, type=int)
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

f = open('out/%s_%s_%d_%d_%d_%g_%g.txt' % (args.dataset, args.alg, args.n_layers, args.emb_dim, args.n_neurons, args.lambda1, args.lambda2),'w')

############# Architecture

#alg', default='pair', choices=['pair', 'triple'])
if args.alg == 'pair':
    model = get_model_pair(usernum, itemnum, mf_dim=args.emb_dim, n_neurons=args.n_neurons, num_layer=args.n_layers, reg_layers=args.lambda1, reg_emb=args.lambda2)
elif args.alg == 'triple':
    model = get_model_triple(usernum, itemnum, mf_dim=args.emb_dim, n_neurons=args.n_neurons, num_layer=args.n_layers, reg_layers=args.lambda1, reg_emb=args.lambda2)
elif args.alg == 'posneg':
    model, model_eval = get_model_pos_neg(usernum, itemnum, mf_dim=args.emb_dim, n_neurons=args.n_neurons, num_layer=args.n_layers, reg_layers=args.lambda1, reg_emb=args.lambda2)
else:
    model = None    


############

u_train = []
i_train = []
oi_train = []

#for u in user_train.keys():
#    n = len(user_train[u]['consume'])    
#    u_train += [u]*n
#    i_train += user_train[u]['consume']
#    oi_train += [owner[a] for a in user_train[u]['consume']]

#def sample_neg(user, itemnum, user_train, user_validation, user_test, consumed_items):
#    item_ip = np.random.randint(0, itemnum)
#    while item_ip in user_train[user]['consume'] or item_ip in user_train[user]['produce'] or item_ip not in consumed_items or item_ip in user_validation[user]['consume'] or item_ip in user_test[user]['consume']:  
#        item_ip = np.random.randint(0, itemnum)
#    return item_ip

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = 0.0

    def on_batch_end(self, batch, logs={}):
        self.losses += logs.get('loss')

for i in range(args.maximum_epochs):

    history = LossHistory()
    loss = 0.0 
    for _ in range(int(oneiteration / args.batch_size)):
        batch_u, batch_i, batch_j, batch_oi, batch_oj =  sampler.next_train_batch()
        
        if(args.alg == 'posneg'):
            labels = [0]*len(batch_u)
            margin = np.full(len(batch_u), 0.2)
            
            hist = model.fit([np.array(batch_u), np.array(batch_i), np.array(batch_j), np.array(batch_oi), np.array(batch_oj), margin], labels, batch_size=args.batch_size, verbose=0, shuffle=False, callbacks=[history])
        
            #hist = model.fit([np.array(batch_u), np.array(batch_i), np.array(batch_j), np.array(batch_oi), np.array(batch_oj)], batch_size=args.batch_size, verbose=0, shuffle=False)
        else:
            user_input = batch_u + batch_u
            item_input = batch_i + batch_j
            owner_input = batch_oi + batch_oj
            labels = [1]*len(batch_u) + [0]*len(batch_u)
            hist = model.fit([np.array(user_input), np.array(item_input), np.array(owner_input)], np.array(labels), batch_size=args.batch_size, verbose=0, shuffle=True, callbacks=[history])
    #j_train = []
    #oj_train = []
    #for u in u_train:
    #    j = sample_neg(u, itemnum, user_train, user_validation, user_test, consumed_items)
    #    
    #    j_train.append(j)
    #    oj_train.append(owner[j])
    
        #user_input = u_train + u_train
        #item_input = i_train + j_train
        #owner_input = oi_train + oj_train        
        # labels = [1]*len(i_train) + [0]*len(j_train)

        
    
    loss += history.losses
    if i % 1 == 0:
        n_sample = int(0.01*usernum)
        user_sample = sample_users_validation(n_sample)
        mrr = 0.0
        
        for u in user_sample:
            item_sample, owner_sample = sample_items(u, 99)
            item_sample += [a for a in user_validation[u]['consume']]
            owner_sample += [owner[a] for a in user_validation[u]['consume']]
            users = np.full(len(item_sample), u, dtype = 'int32') 

            if(args.alg == 'posneg'):
                #[np.array(batch_u), np.array(batch_i), np.array(batch_j), np.array(batch_oi), np.array(batch_oj)]
                #scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)       
                scores = model_eval.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)
            else:                                       
                scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)

            scores = scores.T[0]            
            idx = np.argsort(-scores)            
            ranking = np.array(item_sample)[idx]
            
            rank  = np.where(ranking == user_validation[u]['consume'])[0][0]+1
            mrr += 1.0/rank     
        
        mrr /= n_sample
        pct = (mrr/best_valid_mrr - 1)
        f.write('%d, %f, %f, %f, %f\n' % (i, loss, mrr, best_valid_mrr, pct))
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
    np.save('DCPRec_%s_%d_%g_%g.npy' % (args.dataset, args.emb_dim, args.learning_rate, args.lambda1), [model.best_u, model.best_i])


############################# PREDICTIONS

f = open('out/%s_%s_%d_%d_%d_%g_%g.log' % (args.dataset, args.alg, args.n_layers, args.emb_dim, args.n_neurons, args.lambda1, args.lambda2),'w')

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
    
    if(args.alg == 'posneg'):
        #scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)       
        scores = model_eval.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)
    else:                                       
        scores = model.predict([users, np.array(item_sample), np.array(owner_sample)], batch_size=100, verbose=0)
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

f = open('out/%s_%s_%d_%d_%d_%g_%g.out' % (args.dataset, args.alg, args.n_layers, args.emb_dim, args.n_neurons, args.lambda1, args.lambda2),'w')
f.write('%f, %f, %f, %f, %f, %f, %f, %f\n' %(best_valid_mrr, recall_1, recall_5, recall_10, recall_25, recall_50, ndcg, mrr))
f.flush()
f.close()

