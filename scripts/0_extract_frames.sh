#!/bin/bash

CASE=3

python core/extract_frames_and_AVCS.py --ACS-case=$CASE

for CASE in {1,2,4,5,6,7,8}
     do
      echo $'\nACS-case: '$((CASE)) 
      python core/extract_frames_and_AVCS.py --ACS-case=$CASE
  done
