import pandas as pd
import os
import glob
from scipy import stats
import numpy as np

def breakdown(a, idx=1):
    cuts = [a[idx].min()-1] + list(a[idx].quantile([0.1, 0.25, 0.5, 0.75])) + [a[idx].max()+1]
    #cuts = [a[idx].min()-1] + list([10,50, 100, 500, 1000]) + [a[idx].max()+1]    
    #cuts = [a[idx].min()-1, 10, a[idx].max()+1]
    a['cat'] = pd.cut(a[idx], cuts)
    values = a.groupby('cat').mean()[4]   
    
    s = []
    for i in values:
        s.append('%.2f'% (i))
    return s

def symbol_t_test(x,y):
    is_greater = True if np.mean(x) > np.mean(y) else False
    prob = stats.ttest_ind(x, y, equal_var=False)[1]
    if(prob >= 0.01):
        if(prob >= 0.05):
           return "$\circ$"
        else:
            if(is_greater):
                return "$\\vartriangle$"
            else:
                return "$\\triangledown$"            
    else:
        if(is_greater):
            return "$\\blacktriangle$"
        else:
            return "$\\blacktriangledown$"



def read(folder, dataset, alg, idx):
    
    for d in dataset:
        print('#######'+d)
        
        path = './dcprec/out/%s/dcprec/final/' %(d)
        base_f = glob.glob(path+'*.out')[0].replace('.out','')
           
        base_pd = pd.read_csv(base_f+'.log', header=None)
        base_mrr = base_pd[4].mean()
        
        output = ['dcprec', '%.2f' %(base_mrr)]
        output += breakdown(base_pd, idx)
        print(' & '.join(output) + '\\\\')
        
        for a in alg:    
            max_v = float('-inf')
            max_f = ''

            path  = './%s/out/%s/%s/' %(folder, d, a)
            #print('path: '+path)

            for i in glob.glob(path+'*.out'):
                #print(i)
                with open(i) as f:
                    line = f.readline()
                    value = float(line.split(',')[0])
                    if value > max_v:
                        max_v = value
                        max_f = i
            
            max_f = max_f.replace('.out', '')
            
            max_f_pd = pd.read_csv(max_f+'.log', header=None)
            max_f_mrr = max_f_pd[4].mean()
                
            mrr_improv = (base_mrr/max_f_mrr - 1)*100
            symbol = symbol_t_test(max_f_pd[4], base_pd[4])    
            output = ['%s'%(a), '%.2f (%.2f) %s'%(max_f_mrr, mrr_improv, symbol)]
            output += breakdown(max_f_pd, idx)
            print(' & '.join(output) + '\\\\')
    
read('baseline', ['PinterestCore', 'RedditCore'], ['bpr', 'cprec2', 'fm', 'vista'], idx=1)
read('dcprec', ['PinterestCore', 'RedditCore'], ['dist'], idx=1)

d = pd.read_csv('./finals/DCPRec/RedditCore_30_0.01_0.01.log', header=None)
c = pd.read_csv('./finals/CPRec/RedditCore_30_0.01_0.01_0.001.log', header=None)


for a in [d,c]:
    cuts = [a[1].min()-1] + list(a[1].quantile([0.1, 0.25, 0.5, 0.75, 0.9999])) + [a[1].max()+1]
    a['cat'] = pd.cut(a[1], cuts)
    values = a.groupby('cat').mean()[3]   
    
    s = ''
    for i in values:
        s += ' %.2f & '% (i)
    print(s)      

#http://benalexkeen.com/bucketing-continuous-variables-in-pandas/


#############


import pandas as pd

d = pd.read_csv('./finals/DCPRec/RedditCore_30_0.01_0.01.log', header=None)
c = pd.read_csv('./finals/CPRec/RedditCore_30_0.01_0.01_0.001.log', header=None)


for a in [d,c]:
    cuts = [a[1].min()-1] + list(a[1].quantile([0.1, 0.25, 0.5, 0.75, 0.9999])) + [a[1].max()+1]
    a['cat'] = pd.cut(a[1], cuts)
    values = a.groupby('cat').mean()[3]   
    
    s = ''
    for i in values:
        s += ' %.2f & '% (i)
    print(s)      

#http://benalexkeen.com/bucketing-continuous-variables-in-pandas/


