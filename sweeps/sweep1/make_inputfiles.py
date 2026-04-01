#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:17:45 2023

@author: gregg
"""

import numpy as np
from datetime import datetime
import os
import h5py
import sys
sys.path.append('../..')
sys.path.append('../../helper_libs')
import pickle_helpers as ph

tic = datetime.now()

max_t_shift_list = [100,200,400,600,800,1000]
num_t0_list = [int(x/20) for x in max_t_shift_list]

inputfile_template = "./input_template.py"

round_decimal = 2

for index,max_t_shift in enumerate(max_t_shift_list):
    iter_string = "max_t_shift_"+str(round(max_t_shift,round_decimal))
    iter_string = iter_string.replace(".","_")
        
    inputfile_mod = "./input_files/"+iter_string+".py"
    with open(inputfile_template,"r") as fi, open(inputfile_mod,"w") as fo:
        for line in fi:
            if "num_t_steps =" in line:
                writeline = 'num_t_steps = '+str(num_t0_list[index])+'\n'
            elif "t_shift_superblock_size =" in line:
                writeline = 't_shift_superblock_size = '+str(num_t0_list[index])+'\n'
            elif "max_t_shift =" in line:
                writeline = 'max_t_shift = '+str(max_t_shift)+'\n'
            else:
                writeline = line
                    
            fo.write(writeline)
                
    print("made "+inputfile_mod)
                
toc = datetime.now()
print("Execution time: ")
print(toc-tic)
