#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
from collections import defaultdict

from sampler import WarpSampler
from ball import Ball
from dcprec import Dcprec
from dist import Dist
from fm import FM


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--alg', required=True, choices=['dcprec','ball', 'dist', 'distball', 'fm'])
parser.add_argument('--margin', default=0.5, type=float) #{0.1, 0.2, 0.25, 0.5}
parser.add_argument('--save', action='store_true')
parser.add_argument('--assymetric', default=0, type=int)
parser.add_argument('--min_delta', default=0.01, type=float)
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--latent_dimension', default=20, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--maximum_epochs', default=2000, type=int)
# need tuning for new data, the default value is for reddit
# best parameter for Pinerest data is also 0.1
parser.add_argument('--lambda1', default=0.1, type=float)
# need tuning for new data, the default value is for reddit
# best parameter for Pinerest data is 0.001
#parser.add_argument('--lambda2', default=0.0001, type=float)
args = parser.parse_args()

[user_train, user_validation, user_test, usernum, itemnum] = np.load('data/' + args.dataset + 'Partitioned.npy', allow_pickle=True)


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
oneiteration = min(1000000, oneiteration)

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

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

#['dcprec','ball', 'dist', 'distball']
if args.alg == 'dcprec':
    model = Dcprec(usernum, itemnum, args)
elif args.alg == 'ball':
    model = Ball(usernum, itemnum, args)
elif args.alg == 'dist' or args.alg == 'distball':
    model = Dist(usernum, itemnum, args)
elif args.alg == 'fm':
    model = FM(usernum, itemnum, args)
else:
    model = None
            
sess.run(tf.initialize_all_variables())


best_iter = 0   
best_valid_mrr = 1e-6
prev_valid_mrr = 1e-6

f = open('out/%s/%s/%d_%g_%g_%g.txt' % (args.dataset, args.alg, args.latent_dimension, args.learning_rate, args.lambda1, args.margin),'w')

for i in range(args.maximum_epochs):    
    for _ in range(int(oneiteration / args.batch_size)):
        batch = sampler.next_train_batch()
        batch_u, batch_i, batch_j, batch_oi, batch_oj = batch
        
        
        _, train_loss= sess.run((model.gds, model.loss),{model.batch_u: batch_u, model.batch_i: batch_i, model.batch_j: batch_j, model.batch_oi: batch_oi, model.batch_oj: batch_oj})
    
    
    if i % 10 == 0:
        n_sample = int(0.01*usernum)
        user_sample = sample_users_validation(n_sample)
        mrr = 0.0
        
        for u in user_sample:
            item_sample, owner_sample = sample_items(u, 99)
            item_sample += [a for a in user_validation[u]['consume']]
            owner_sample += [owner[a] for a in user_validation[u]['consume']]
                        
            scores = sess.run((model.scores), {model.user_eval: [u], model.item_eval: item_sample, model.owner_eval: owner_sample})
                        
            idx = np.argsort(scores)            
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
            model.save(sess, args)            
        elif i >= best_iter + 50:
            f.write('break\n')
            f.flush()
            break            

        prev_valid_mrr = mrr        

f.close()

if(args.save):
    np.save('out/%s/%s/%d_%g_%g_%g.npy' % (args.dataset, args.alg, args.latent_dimension, args.learning_rate, args.lambda1, args.margin), [model.best_u, model.best_i])


############################# PREDICTIONS
def make_pred(u_emb, i_emb, o_emb, alg):
    pred = np.sum(np.square(u_emb - i_emb), axis=1, keepdims=True)
    if not alg == 'dist':
        pred += np.sum(np.square(u_emb - o_emb), axis=1, keepdims=True)
        if alg == 'fm':
            pred += np.sum(np.square(i_emb - o_emb), axis=1, keepdims=True)
        
    return pred

f = open(
    'out/%s/%s/%d_%g_%g_%g.log' % (args.dataset, args.alg, args.latent_dimension, args.learning_rate, args.lambda1, args.margin),
    'w')    

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
    
    item_sample, owner_sample = sample_items(u, 99)
    item_sample += [a for a in user_test[u]['consume']]
    owner_sample += [owner[a] for a in user_test[u]['consume']]
    
    if not args.assymetric:
        scores = make_pred(model.best_u[u,:], model.best_i[item_sample,:], model.best_u[owner_sample,:], args.alg)[:,0]
    else:
        scores = make_pred(model.best_u[u,:], model.best_i[item_sample,:], model.best_o[owner_sample,:], args.alg)[:,0]
                    
    idx = np.argsort(scores)            
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

f = open(
    'out/%s/%s/%d_%g_%g_%g.out' % (args.dataset, args.alg, args.latent_dimension, args.learning_rate, args.lambda1, args.margin),
    'w') 
f.write('%f, %f, %f, %f, %f, %f, %f, %f\n' %(best_valid_mrr, recall_1, recall_5, recall_10, recall_25, recall_50, ndcg, mrr))
f.flush()
f.close()
