#!/bin/bash

INFO='my_fantastic_model'
EPOCHS=10
LR=0.00005

python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO 
