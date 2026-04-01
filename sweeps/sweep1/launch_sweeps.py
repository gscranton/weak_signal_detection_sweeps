# exec(open("launch_sweeps.py").read())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:42:47 2023

@author: gregg
"""

import numpy as np
import glob
import os
import sys
sys.path.append('../..')

run_flag = True

inputfile_directory = "./input_files/"
#inputfile_directory = ""
launcher_path = "python ../../launcher.py"
num_threads = 1

inputfile_list_raw = glob.glob(inputfile_directory+"*")

inputfile_list = []
for inputfile in inputfile_list_raw:
    output_dir = './output_dirs/output_data_'+inputfile[:-4]
    if not os.path.exists(output_dir):
        inputfile_list.append(inputfile.split('/')[-1])

inputfile_lists = np.array_split(np.array(inputfile_list),num_threads)

for idx,nlist in enumerate(inputfile_lists):
    #command_string = 'gnome-terminal -x sh -c "'
    command_string = 'gnome-terminal --wait -- sh -c "'
    
    for n in nlist:
        command_string += launcher_path+' -d '+n+'; '
    
    command_string += '\n"'
    
    if run_flag:
        os.system(command_string)
    else:
        print(command_string)
