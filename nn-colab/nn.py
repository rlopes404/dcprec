#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, Lambda
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l1, l2
from keras.initializers import RandomNormal
from keras.utils import plot_model
import numpy as np


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)
    
def identity_loss2(y_true, y_pred):
    return K.sum(K.maximum(0.0, 1.0 + y_pred[1] - y_pred[0]), axis=-1) 
    
def identity_loss3(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)    


def get_model_pos_neg(num_users, num_items, mf_dim=25, n_neurons=25, num_layer=1, reg_layers=0.01, reg_emb=0.01):
    
    layers = [n_neurons]*num_layer
    factor = 2
    for i in range(1, num_layer):
        layers[i] = int(layers[i] / factor)
        factor *= 2
    
    def get_user_embed_module():
        user = Input(shape=(1,), dtype='int32', name = 'user')
        user_embedder = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'user_embedder', embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
        user_out = Flatten()(user_embedder(user))
        user_module = Model(user, user_out)
        return user_module
        
    def get_item_embed_module():
        item = Input(shape=(1,), dtype='int32', name = 'item')
        item_embedder = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'item_embedder', embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
        item_out = Flatten()(item_embedder(item))
        item_module = Model(item, item_out)
        return item_module
    
    def get_user_item_model():
        user = Input(shape=(mf_dim,), dtype='float', name = 'user')
        item = Input(shape=(mf_dim,), dtype='float', name = 'item')

        ui_vector = keras.layers.Concatenate()([user, item])
        for idx in range(0, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="ui_layer%d_%d" %(idx, layers[idx]))
            ui_vector = layer(ui_vector)
        return Model([user, item], ui_vector)
        
    def get_user_producer_model():
        user = Input(shape=(mf_dim,), dtype='float', name = 'user')
        producer = Input(shape=(mf_dim,), dtype='float', name = 'producer')
        
        up_vector =  keras.layers.Concatenate()([user, producer])        
        for idx in range(0, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="up_layer%d_%d" %(idx, layers[idx]))
            up_vector = layer(up_vector)
        return Model([user, producer], up_vector)  

    def get_triplet_model():    
        user = Input(shape=(1,), dtype='int32', name = 'user')
        item = Input(shape=(1,), dtype='int32', name = 'item')
        producer = Input(shape=(1,), dtype='int32', name = 'producer')
        
        user_embedder = get_user_embed_module()
        item_embedder = get_item_embed_module()

        u_emb = user_embedder(user)
        i_emb = item_embedder(item)
        p_emb = user_embedder(producer)
                      
        user_item_model = get_user_item_model()
        user_producer_model = get_user_producer_model()
        
        ui_vector = user_item_model([u_emb, i_emb])
        up_vector = user_producer_model([u_emb, p_emb])
        
        merged = keras.layers.Concatenate()([ui_vector, up_vector])
        
        #merged = keras.layers.Concatenate()([u_emb, i_emb, p_emb])
        
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(merged)
        
        return Model(input=[user, item, producer], output=prediction)
     
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    producer_input = Input(shape=(1,), dtype='int32', name = 'producer_input')
    
    neg_item_input = Input(shape=(1,), dtype='int32', name = 'neg_item_input')
    neg_producer_input = Input(shape=(1,), dtype='int32', name = 'neg_producer_input')
    
    triplet_model = get_triplet_model()    
    
    pos = triplet_model([user_input, item_input, producer_input])
    neg = triplet_model([user_input, neg_item_input, neg_producer_input])
    
    #def triplet_loss(inputs):
    #    pos, neg = inputs
    #    return K.sum(K.maximum(0.0, 1.0 + neg - pos), axis=-1)    

    constant = Input(shape=(1,), dtype='float', name = 'margin')    
    sub = keras.layers.Subtract()([neg, pos])
    sub = keras.layers.Add()([constant, sub])
    
    #todo: Constant
    #added = keras.layers.Add()([K.constant(1.0, dtype=float, shape=(1,)), sub])
    out = keras.layers.ReLU()(sub)

    
    model = Model(input=[user_input, item_input, neg_item_input, producer_input, neg_producer_input, constant], output=out)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    
    #loss = Lambda(triplet_loss)([pos,neg])
    #model = Model(input=[user_input, item_input, neg_item_input, producer_input, neg_producer_input], output=loss)
    #model.compile(optimizer=Adam(), loss=identity_loss)

#    model = Model(input=[user_input, item_input, neg_item_input, producer_input, neg_producer_input], output=[pos,neg])
#    model.compile(optimizer=Adam(), loss=identity_loss2)
    
    #model = Model(input=[user_input, item_input, neg_item_input, producer_input, neg_producer_input], output=pos)
    #model.compile(optimizer=Adam(), loss='binary_crossentropy')
    
    plot_model(model, to_file='model_posneg.png')

    
    return model, triplet_model    


#this model is completely assymetric in that we have an embedding matrix for each role: user, item, consumer, producer
#model: <user, item> (+) <consumer, producer>
def get_model_assymetric(num_users, num_items, mf_dim=25, layers=[50], reg_layers=0.01, reg_emb=0.01):
        
    layers = [n_neurons]*num_layer
    factor = 2
    for i in range(1,num_layer):
        layers[i] = int(layers[i] / factor)
        factor *= 2
 
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    producer_input = Input(shape=(1,), dtype='int32', name = 'producer_input')
 
    # Embedding layer
    UI_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'ui_emb_u_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
    UI_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'ui_emb_i_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
  
    UO_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'uo_emb_u_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
    UO_Embedding_producer = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'uo_emb_o_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
  
    # UI Part
    ui_user_latent = Flatten()(UI_Embedding_User(user_input))
    ui_item_latent = Flatten()(UI_Embedding_Item(item_input))
    ui_vector = keras.layers.Concatenate()([ui_user_latent, ui_item_latent])
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="ui_layer%d_%d" %(idx, layers[idx]))
        ui_vector = layer(ui_vector)
  
    # UO part
    uo_user_latent = Flatten()(UO_Embedding_User(user_input))
    uo_producer_latent = Flatten()(UO_Embedding_producer(producer_input))
    uo_vector =  keras.layers.Concatenate()([uo_user_latent, uo_producer_latent])
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="uo_layer%d_%d" %(idx, layers[idx]))
        uo_vector = layer(uo_vector)
    
    # Concatenate parts
    predict_vector = keras.layers.Concatenate()([ui_vector, uo_vector])
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input, producer_input], output=prediction)
    plot_model(model, to_file='asymmetric_model.png')
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model


def get_model_triple(num_users, num_items, mf_dim=25, n_neurons=25, num_layer=1, reg_layers=0.01, reg_emb=0.01):
    
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    producer_input = Input(shape=(1,), dtype='int32', name = 'producer_input')
 
    # Embedding layer
    user_embedder = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'u_embedder_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
    item_embedder = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'i_embedder_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
  
    # embedding
    user_emb = Flatten()(user_embedder(user_input))
    item_emb = Flatten()(item_embedder(item_input))  
    producer_emb = Flatten()(user_embedder(producer_input))
      
    # triple interaction    
    layers = [n_neurons]*num_layer
    factor = 2
    for i in range(1,num_layer):
        layers[i] = int(layers[i] / factor)
        factor *= 2
    
    ui_vector = keras.layers.Concatenate()([user_emb, item_emb, producer_emb])
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="ui_layer%d_%d" %(idx, layers[idx]))
        ui_vector = layer(ui_vector)

    ui_vector = Dropout(0.5)(ui_vector)
 #   ui_vector = Dense(5, kernel_regularizer=l2(reg_layers), activation='relu')(ui_vector)
 #   ui_vector = Dropout(0.5)(ui_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(ui_vector)
    
    model = Model(input=[user_input, item_input, producer_input], output=prediction)
    plot_model(model, to_file='triple_model.png')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model    

#this model is symetric in that we have two embedding matrices: user, item
#model: <user, item> (+) <user, user>
def get_model_pair(num_users, num_items, mf_dim=25, n_neurons=25, num_layer=1, reg_layers=[0.01], reg_emb=0.01):
 
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    producer_input = Input(shape=(1,), dtype='int32', name = 'producer_input')
 
    # Embedding layer
    user_embedder = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'u_embedder_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
    item_embedder = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'i_embedder_%d' %(mf_dim), embeddings_initializer = RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_emb), input_length=1)
  
    # embedding
    user_emb = Flatten()(user_embedder(user_input))
    item_emb = Flatten()(item_embedder(item_input))  
    producer_emb = Flatten()(user_embedder(producer_input))
    
    layers = [n_neurons]*num_layer
    factor = 2
    for i in range(1,num_layer):
        layers[i] = int(layers[i] / factor)
        factor *= 2
      
    # user item interaction
    ui_vector = keras.layers.Concatenate()([user_emb, item_emb])
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="ui_layer%d_%d" %(idx, layers[idx]))
        ui_vector = layer(ui_vector)
  
    # user producer interaction    
    up_vector =  keras.layers.Concatenate()([user_emb, producer_emb])
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers), activation='relu', name="up_layer%d_%d" %(idx, layers[idx]))
        up_vector = layer(up_vector)
    
    # Concatenate parts
    predict_vector = keras.layers.Concatenate()([ui_vector, up_vector])
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input, producer_input], output=prediction)
    plot_model(model, to_file='pair_model.png')
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model    
