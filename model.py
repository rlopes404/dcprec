#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf

class Model():
    
    def save(self, sess, args):
        #self.saver.save(sess, self.model_name)
        self.best_u = sess.run(self.u_embeddings)
        self.best_i = sess.run(self.i_embeddings)
        if args.assymetric:
            self.best_o = sess.run(self.o_embeddings)
        
    
    def ball_normalize_all(self, sess):
        _u_embeddings = tf.assign(self.u_embeddings, tf.clip_by_norm(self.u_embeddings, 1.0, axes=1))
        _i_embeddings = tf.assign(self.i_embeddings, tf.clip_by_norm(self.i_embeddings, 1.0, axes=1))
        sess.run((_u_embeddings, _i_embeddings))        
        
    def __init__(self, usernum, itemnum, args):
        self.n_users = usernum
        self.n_items = itemnum
        self.best_u = None
        self.best_i = None
        
        
        d_emb = args.latent_dimension
        learning_rate = args.learning_rate
                
        self.model_name = 'R_%s_%d_%g_%g.ckpt' % (args.dataset, d_emb, learning_rate, args.lambda1)
              
        #self.u_embeddings = tf.Variable(tf.random_uniform([self. n_users, d_emb], maxval=0.1))
        #self.i_embeddings = tf.Variable(tf.random_uniform([self.n_items, d_emb], maxval=0.1))
        
        self.u_embeddings = tf.Variable(tf.random.normal([self.n_users, d_emb], stddev=0.1))
        self.i_embeddings = tf.Variable(tf.random.normal([self.n_items, d_emb], stddev=0.1))
        if args.assymetric:
            self.o_embeddings = tf.Variable(tf.random.normal([self.n_users, d_emb], stddev=0.1))
       
        self.batch_u = tf.placeholder(tf.int32, [None])
        self.batch_i = tf.placeholder(tf.int32, [None]) #positive items
        self.batch_j = tf.placeholder(tf.int32, [None]) #negative items
        self.batch_oi = tf.placeholder(tf.int32, [None]) #owner positive items
        self.batch_oj = tf.placeholder(tf.int32, [None]) #owner negative items
        

        
        self.batch_u_emb = tf.gather(self.u_embeddings, self.batch_u)
        self.batch_i_emb = tf.gather(self.i_embeddings, self.batch_i)
        self.batch_j_emb = tf.gather(self.i_embeddings, self.batch_j)        
        
        if not args.assymetric:        
            self.batch_oi_emb = tf.gather(self.u_embeddings, self.batch_oi)        
            self.batch_oj_emb = tf.gather(self.u_embeddings, self.batch_oj)
        else:
            self.batch_oi_emb = tf.gather(self.o_embeddings, self.batch_oi)        
            self.batch_oj_emb = tf.gather(self.o_embeddings, self.batch_oj)
        
        
        ### model
        pos_distances = tf.reduce_sum(tf.math.squared_difference(self.batch_u_emb, self.batch_i_emb), axis=1)
        neg_distances = tf.reduce_sum(tf.math.squared_difference(self.batch_u_emb, self.batch_j_emb), axis=1)
        if not args.alg == 'dist':
            pos_distances += tf.reduce_sum(tf.math.squared_difference(self.batch_u_emb, self.batch_oi_emb), axis=1)
            if(args.alg == 'fm'):
                pos_distances += tf.reduce_sum(tf.math.squared_difference(self.batch_i_emb, self.batch_oi_emb), axis=1)
            
            neg_distances +=  tf.reduce_sum(tf.math.squared_difference(self.batch_u_emb, self.batch_oj_emb), axis=1)
            if(args.alg == 'fm'):
                neg_distances += tf.reduce_sum(tf.math.squared_difference(self.batch_j_emb, self.batch_oj_emb), axis=1)
        
        
        ### loss function       
        if args.alg == 'ball': 
            self.loss =  tf.reduce_sum(tf.nn.relu(args.margin + pos_distances - neg_distances))            
        else:
            if not args.alg == 'dist':
                self.loss = -tf.reduce_sum(tf.log_sigmoid(neg_distances - pos_distances)) + sum(map(tf.nn.l2_loss, [self.batch_u_emb, self.batch_i_emb, self.batch_j_emb, self.batch_oi_emb, self.batch_oj_emb]))* args.lambda1                
            else:
                self.loss = -tf.reduce_sum(tf.log_sigmoid(neg_distances - pos_distances)) + sum(map(tf.nn.l2_loss, [self.batch_u_emb, self.batch_i_emb, self.batch_j_emb]))* args.lambda1  
       
        #self.auc = tf.reduce_mean((tf.sign(neg_distances - pos_distances) + 1) / 2)

        self.gds = []
        self.gds.append(tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(self.loss))
        if args.alg == 'ball':
            self.gds.append(tf.assign(self.u_embeddings, tf.clip_by_norm(self.u_embeddings, 1.0, axes=1)))
            self.gds.append(tf.assign(self.i_embeddings, tf.clip_by_norm(self.i_embeddings, 1.0, axes=1)))



        self.user_eval = tf.placeholder(tf.int32, [None])
        self.item_eval = tf.placeholder(tf.int32, [None])
        self.owner_eval = tf.placeholder(tf.int32, [None])
        
        self.batch_user_eval = tf.gather(self.u_embeddings, self.user_eval)
        self.batch_item_eval = tf.gather(self.i_embeddings, self.item_eval)
        if not args.assymetric:
            self.batch_owner_eval = tf.gather(self.u_embeddings, self.owner_eval)
        else:
            self.batch_owner_eval = tf.gather(self.o_embeddings, self.owner_eval)

        
        self.scores =  tf.reduce_sum(tf.math.squared_difference(self.batch_user_eval, self.batch_item_eval), axis=1)
        if not args.alg == 'dist':
            self.scores +=  tf.reduce_sum(tf.math.squared_difference (self.batch_user_eval, self.batch_owner_eval), axis=1)
            if(args.alg == 'fm'):
                self.scores += tf.reduce_sum(tf.math.squared_difference (self.batch_item_eval, self.batch_owner_eval), axis=1)

        #self.saver = tf.train.Saver({'u': self.u_embeddings, 'i': self.i_embeddings})
