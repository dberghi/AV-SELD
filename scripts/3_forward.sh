#!/bin/bash

INFO='my_fantastic_model'
EPOCH=50 #use your best epoch
LR=0.0001


python forward.py --epoch=$EPOCH --lr=$LR --info=$INFO


