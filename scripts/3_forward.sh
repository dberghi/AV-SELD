#!/bin/bash

INFO='my_fantastic_model'
EPOCH=10 #use your best epoch
LR=0.00005


python forward.py --epoch=$EPOCH --lr=$LR --info=$INFO


