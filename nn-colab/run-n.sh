#!/bin/bash

dataset=$1
nlayers=$2

./n.sh $dataset $nlayers 10 &
./n.sh $dataset $nlayers 25 &
./n.sh $dataset $nlayers 50 &
./n.sh $dataset $nlayers 75 &
./n.sh $dataset $nlayers 100 &
