#!/bin/bash

dataset=$1
factors=($2)
lambdas=(0.001 0.01 0.1 1)

for f in ${factors[*]};
do
    for l1 in ${lambdas[*]};
    do
    	for l2 in ${lambdas[*]};
    	do
            python2 main-nn.py --dataset=$dataset --latent_dimension=$f --lambda1=$l1 --lambda2=$l2
        done
    done
done
