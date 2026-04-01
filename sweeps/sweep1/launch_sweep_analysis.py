"""
exec(open("launch_sweep_analysis.py").read())
"""
import numpy as np
import glob
import os
import sys

run_flag = True

output_dirs = "./output_dirs/"

output_dir_list = glob.glob(output_dirs+"*")

run_num_list = list(range(1,6))

for n in range(len(output_dir_list)):
    rname = output_dir_list[n].split('/')[-1][5:]
    for m in range(len(run_num_list)):
        command_string = "python ../../sweep_analysis.py -sp -srd -d "+rname+" -r "+str(run_num_list[m])
        if run_flag:
            os.system(command_string)
        else:
            print(command_string)
