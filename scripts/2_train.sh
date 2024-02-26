#!/bin/bash

INFO='my_fantastic_model'
EPOCHS=50
LR=0.0001

python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO 
