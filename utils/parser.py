import os
import glob
import shutil


#alg = ['ball', 'dcprec', 'dist', 'distball']
#alg = ['bpr', 'cprec', 'fm', 'vista']
#alg = ['bpr', 'cprec', 'cprec2', 'cprec3', 'cprec3sym', 'fm', 'vista']
alg = ['dcprec']
dataset = ['RedditCore', 'PinterestCore']


for d in dataset:
    for a in alg:    
        max_v = float('-inf')
        max_f = ''
        
        path  = './%s/%s/' %(d, a)
        print('path: '+path)

        for i in glob.glob(path+'*.out'):
            print(i)
            with open(i) as f:
                line = f.readline()
                value = float(line.split(',')[0])
                if value > max_v:
                    max_v = value
                    max_f = i

        max_f = max_f.replace('.out', '')
        print('max_f: '+max_f)
        
       
        shutil.copyfile(max_f+".out" , max_f.replace(a, a+'/final')+'.out')  
        shutil.copyfile(max_f+".log" , max_f.replace(a, a+'/final')+'.log')                  
        shutil.copyfile(max_f+".txt" , max_f.replace(a, a+'/final')+'.txt')

#####################


import shutil
import os

factors = [10,15,20,25,30]
lambdas1 = [0.001, 0.01, 0.1, 1]
lambdas2 = [0.001, 0.01, 0.1, 1]

#RedditCore_10_0.01_0.001_0.0001
 
normalized = 0

dataset='RedditCore'
learning_rate=0.01

max_v = float('-inf')
max_f = ''

for latent_dimension in factors:
    for lambda1 in lambdas1:
        for lambda2 in lambdas2:
            s  = '%s_%d_%g_%g_%g' % (dataset, latent_dimension, learning_rate, lambda1, lambda2)
            with open(s+'.txt') as f:
                line = f.readlines()
                line = line[len(line)-1]
                value = float(line.split(',')[0])
                if value > max_v:
                    max_v = value
                    max_f = s
                
shutil.copyfile (max_f+".out" , "./final/"+max_f+".out")  
shutil.copyfile (max_f+".log" , "./final/"+max_f+".log")                    
shutil.copyfile (max_f+".txt" , "./final/"+max_f+".txt")
