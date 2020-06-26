import tensorflow as tf
import numpy as np


def __unit_ball_normalize(v):
    norm = tf.norm(v, axis=1, keepdims=True)
    aux = tf.math.maximum(norm, [1])
    return v/aux  

u = tf.Variable(np.array([[1.0, -1.0], [1.0, 0.0]]))
uu = tf.expand_dims(u,1)


v = tf.Variable(np.array([[-0.5,0.0], [3.0,4.0], [0, -2]]))
vv = tf.expand_dims(v,0)
#norm = tf.norm(v, axis=1, keepdims=True)
#aux = tf.math.maximum(b,[1])
#r = v/aux
r = tf.math.squared_difference(uu,vv)
rr = tf.reduce_sum(r,2)


k = tf.math.top_k(tf.constant([4,1,10,-1,2]), 2)


init = tf.global_variables_initializer()
sess =  tf.Session()
sess.run(init)

a = sess.run(u)
print(a)
b = sess.run(v)
print(b)   
c = sess.run(r)
print(c)
d = sess.run(rr)
print(d)

e,f = sess.run(k)
    #b = sess.run(norm)
    #aux = tf.reduce_max(b, reduction_indices=[1])
    #c = sess.run(aux)
    d = sess.run(r)
    print(a)
    #print(b)
    #print(c)
    print(d)
    print('#########')

    kk = tf.scatter_update(v, [0,2], [[-10, -10],[-20,-20]])
    tf.assign(v,kk)
    teste2 = sess.run(v)   
    print(teste2)


#####

d_emb = 5
p = tf.Variable(np.array([[3,3],[4,5],[7,8],[9,10]]))
q = tf.Variable(np.array([[2,1],[1,-1],[-4,5],[4,5]]))

r = tf.reduce_sum(tf.square(p-q), axis=1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    v = sess.run(r)
    print(v)


##

import numpy as np

DATA_PATH = '/home/ramon/Dropbox/CPRec/data/'
dataset = np.load(DATA_PATH + 'RedditCorePartitioned.npy')
[user_train, user_validation, user_test, usernum, itemnum] = dataset


# count positive events
owner = dict()
oneiteration = 0

for user in range(usernum):
    oneiteration += len(user_train[user]['consume'])            
    for item in set(user_train[user]['produce']):
        if item in owner:
            print "multiple_creators!"
        owner[item] = user

#### RAMON
data_test = {}
for user in range(usernum):
    user_test[user]['sample'] = set()
    
    for i in range(99):
        item = np.random.randint(0, itemnum)
        while item in user_train[user]['consume'] or item in user_test[user]['consume'] or item in user_test[user]['sample']: item = np.random.randint(0, itemnum)
        user_test[user]['sample'].add(item)
        
    user_train[user]['consume_owners'] = set()
    for item in user_train[user]['consume']:
        user_train[user]['consume_owners'].add(owner[item])
#### RAMON

for item in range(itemnum):
    if item not in owner:
        print "missing creator!"
        break
