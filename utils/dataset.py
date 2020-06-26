#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


def compute_stats(dataset):
    [user_train, user_validation, user_test, usernum, itemnum] = np.load('data/' + dataset + 'Partitioned.npy', allow_pickle=True)

    consume=[]
    produce=[]
    
    consumed = set()
    produced = set()
    total = set()
    n = 0.0

    for user in xrange(usernum):
        if len(user_train[user]['consume']) == 0:
            continue
            
        n += 1.0
        if len(user_train[user]['consume']) > 0:
            consume.append(len(user_train[user]['consume']))
            consumed.update(user_train[user]['consume'])
            total.update(user_train[user]['consume'])
        
        if len(user_train[user]['produce']) > 0:
            produce.append(len(user_train[user]['produce']))
            produced.update(user_train[user]['produce'])     
            total.update(user_train[user]['produce'])


    print('%s & %d & %d & %d & %.2f & %.2f \\\\' %(dataset, n, len(total), np.sum(consume), np.mean(consume), np.mean(produce)))
    
compute_stats('RedditCore')
compute_stats('PinterestCore')

