#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:07:20 2022

@author: gregg
"""

import os
import numpy as np
import sys
import time
import random
import argparse
import importlib.util

def load_dynamic_variables(folder_path, file_name):
    # Ensure the file_name ends with .py
    if not file_name.endswith('.py'):
        file_name += '.py'
        
    file_path = os.path.join(folder_path, file_name)
    
    # 1. Create a module name (can be anything, e.g., "dynamic_mod")
    module_name = "custom_config"
    
    # 2. Load the file structure
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    
    if spec is None:
        raise FileNotFoundError(f"Could not find the file: {file_path}")
        
    # 3. Create the module object from the spec
    module = importlib.util.module_from_spec(spec)
    
    # 4. Execute the module to populate the variables
    spec.loader.exec_module(module)
    
    return module

parser = argparse.ArgumentParser()
parser.add_argument('-rl','--min_run_index',default=1,type=int)
parser.add_argument('-ru','--max_run_index',default=6,type=int)
parser.add_argument('-d','--data_folder_name',default='1',type=str)
parser.add_argument('-ts','--timestep',default=30,type=int)
parser.add_argument('-t1','--timeoffset1',default=0,type=int)
parser.add_argument('-t2','--timeoffset2',default=0,type=int)
parser.add_argument('-as','--a2_superblock_size',default=2,type=int)
parser.add_argument('-rf','--run_flag',default=1,type=int)
args = parser.parse_args()

min_run_index = int(args.min_run_index)
max_run_index = int(args.max_run_index)
data_folder_name = str(args.data_folder_name)
timestep = int(args.timestep)
timeoffset1 = int(args.timeoffset1)
timeoffset2 = int(args.timeoffset2)
A2_superblock_size = int(args.a2_superblock_size)
run_flag = bool(args.run_flag)

if data_folder_name[-3:] == '.py':
    data_folder_name = data_folder_name[:-3]

print('input file name')
print(str(data_folder_name))
os.system('ls '+str(data_folder_name))
os.system('ls '+str(data_folder_name)+'0')

current_dir = os.path.dirname(os.path.abspath(__name__))
relative_path = "./input_files"
folder_path = os.path.abspath(os.path.join(current_dir,relative_path))
f = load_dynamic_variables(folder_path,data_folder_name+'.py')

phi1 = f.phi1
max_t_shift = f.max_t_shift
save_verbosity_index = f.save_verbosity_index
pulse_filename = f.pulse_filename
x10 = f.x10
z10 = f.z10
x20 = f.x20
z20 = f.z20
x30 = f.x30
z30 = f.z30
t_sig_start = f.t_sig_start
noise_flag = f.noise_flag
gamma1 = f.gamma1
gamma2 = f.gamma2
alpha1 = f.alpha1
alpha2 = f.alpha2
beta1 = f.beta1
C1 = f.C1
kappa12 = f.kappa12
sig_repeat_timestep_factor = f.sig_repeat_timestep_factor
sensor_increment = f.sensor_increment
phi3 = f.phi3
pulse_type = f.pulse_type
del_t_ff = f.del_t_ff
t_offset1 = f.t_offset1
t_offset2 = f.t_offset2
t_offset3 = f.t_offset3
input_filename = f.input_filename
A1 = f.A1
total_t = f.total_t
signal_mag = f.signal_mag
af_list = f.af_list
pulse1_height = f.pulse1_height
pulse1_width = f.pulse1_width
pulse1_type = f.pulse1_type
pulse1_center = f.pulse1_center
pulse2_height = f.pulse2_height
pulse2_width = f.pulse2_width
pulse2_type = f.pulse2_type
pulse2_center = f.pulse2_center
pulse3_height = f.pulse3_height
pulse3_width = f.pulse3_width
pulse3_type = f.pulse3_type
pulse3_center = f.pulse3_center
spect_num = f.spect_num
Cff1 = f.Cff1
Cff2 = f.Cff2
Cff3 = f.Cff3
t_offset2 = f.t_offset2
max_t_shift = f.max_t_shift
num_t_steps = f.num_t_steps
t_shift_superblock_size = f.t_shift_superblock_size
x10 = f.x10
z10 = f.z10
x20 = f.x20
z20 = f.z20
x30 = f.x30
z30 = f.z30
frequency = f.frequency
alpha1 = f.alpha1
alpha2 = f.alpha2
kappa12 = f.kappa12
gamma1 = f.gamma1
gamma2 = f.gamma2
abs_pulse_location_flag = f.abs_pulse_location_flag
snap_flag = f.snap_flag

#A2_block_list = list(range(0,len(af_list.split(' ')),A2_superblock_size))
af_len = len([float(x) for x in af_list.split(' ')])
A2_block_list = list(range(0,af_len,A2_superblock_size))

t_shift_block_list = list(range(num_t_steps))

num_t_shift_superblocks = int(np.ceil(len(t_shift_block_list)/t_shift_superblock_size))
t_shift_superblock_lists = []
for i in range(num_t_shift_superblocks):
    t_shift_superblock_list = list(range(i*t_shift_superblock_size,(i+1)*t_shift_superblock_size))
    t_shift_superblock_lists.append(t_shift_superblock_list)
    
timestep_list = list(range(0, 2*len(A2_block_list)*len(t_shift_superblock_lists)*timestep,timestep))
random.shuffle(timestep_list)

if not os.path.isdir('./data/data_'+str(data_folder_name)):
    os.system('mkdir data/data_'+str(data_folder_name))

relative_path = "../.."
folder_path = os.path.abspath(os.path.join(current_dir,relative_path))

ind=0
for i in range(len(A2_block_list)):
    for j in range(len(t_shift_superblock_lists)):
        command_string = ''
        command_string += 'gnome-terminal -x sh -c "'
        command_string += 'sleep '+str(timestep_list[ind]+timeoffset1)+'s; '
        command_string += 'python '+folder_path+'/run_sweep.py '
        command_string += '--af_block_lim '+str(A2_block_list[i])+' '+str(A2_block_list[i]+A2_superblock_size)+' '
        command_string += '--t_shift_block_lim '+str(t_shift_superblock_lists[j][0])+' '+str(int(t_shift_superblock_lists[j][-1])+1)+' '
        command_string += '--num_t_shift_steps '+str(len(t_shift_block_list))+' '
        command_string += '--phi1 '+str(phi1)+' '
        command_string += '--signal_mag 0 '
        command_string += '--min_run_index '+str(min_run_index)+' '
        command_string += '--max_run_index '+str(max_run_index)+' '
        command_string += '--data_folder_name '+str(data_folder_name)+' '
        command_string += '--max_t_shift '+str(max_t_shift)+' '
        command_string += '--frequency '+str(frequency)+' '
        command_string += '--af_list '+af_list+' '
        command_string += '--spect_num '+str(spect_num)+' '
        command_string += '--save_verbosity_index '+str(save_verbosity_index)+' '
        command_string += '--input_filename '+input_filename+' '
        command_string += '--pulse_filename '+pulse_filename+' '
        command_string += '--total_t '+str(total_t)+' '
        command_string += '--x10 '+str(x10)+' '
        command_string += '--z10 '+str(z10)+' '
        command_string += '--x20 '+str(x20)+' '
        command_string += '--z20 '+str(z20)+' '
        command_string += '--x30 '+str(x30)+' '
        command_string += '--z30 '+str(z30)+' '
        command_string += '--t_sig_start '+str(t_sig_start)+' '
        if not noise_flag:
            command_string += '--noise_flag '
        command_string += '--A1 '+str(A1)+' '
        command_string += '--pulse1_width '+str(pulse1_width)+' '
        command_string += '--pulse1_center '+str(pulse1_center)+' '
        command_string += '--pulse1_height 0 '
        command_string += '--pulse1_type '+str(pulse1_type)+' '
        command_string += '--pulse2_width '+str(pulse2_width)+' '
        command_string += '--pulse2_center '+str(pulse2_center)+' '
        command_string += '--pulse2_height 0 '
        command_string += '--pulse2_type '+str(pulse2_type)+' '
        command_string += '--pulse3_width '+str(pulse3_width)+' '
        command_string += '--pulse3_center '+str(pulse3_center)+' '
        command_string += '--pulse3_height 0 '
        command_string += '--pulse3_type '+str(pulse3_type)+' '
        command_string += '--sensor_increment '+str(sensor_increment)+' '
        command_string += '--phi3 '+str(phi3)+' '
        command_string += '--del_t_ff '+str(del_t_ff)+' '
        command_string += '--Cff1 '+str(Cff1)+' '
        command_string += '--Cff2 '+str(Cff2)+' '
        command_string += '--Cff3 '+str(Cff3)+' '
        command_string += '--t_offset1 '+str(t_offset1)+' '
        command_string += '--t_offset2 '+str(t_offset2)+' '
        command_string += '--t_offset3 '+str(t_offset3)+' '
        command_string += '--gamma1 '+str(gamma1)+' '
        command_string += '--gamma2 '+str(gamma2)+' '
        command_string += '--alpha1 '+str(alpha1)+' '
        command_string += '--alpha2 '+str(alpha2)+' '
        command_string += '--beta1 '+str(beta1)+' '
        command_string += '--C1 '+str(C1)+' '
        command_string += '--kappa12 '+str(kappa12)+' '
        command_string += '--abs_pulse_location_flag '+str(abs_pulse_location_flag)+' '
        command_string += '--snap_flag '+str(snap_flag)+' '
        
        command_string += ' | tee ./output_dirs/data_'+str(data_folder_name)+'/logfile_all_runs_'+str(A2_block_list[i])+\
                      '_'+str(t_shift_superblock_lists[j][0])+'_'+str(t_shift_superblock_lists[j][-1])+'.txt"'
        
        if run_flag:
            os.system(command_string)
        else:
            print(str(ind+1))
            print(command_string)
            print("\n")
        ind += 1
        
for i in range(len(A2_block_list)):
    for j in range(len(t_shift_superblock_lists)):
        command_string = ''
        command_string += 'gnome-terminal -x sh -c "'
        command_string += 'sleep '+str(timestep_list[ind]+timeoffset1)+'s; '
        command_string += 'python '+folder_path+'/run_sweep.py '
        command_string += '--af_block_lim '+str(A2_block_list[i])+' '+str(A2_block_list[i]+A2_superblock_size)+' '
        command_string += '--t_shift_block_lim '+str(t_shift_superblock_lists[j][0])+' '+str(int(t_shift_superblock_lists[j][-1])+1)+' '
        command_string += '--num_t_shift_steps '+str(len(t_shift_block_list))+' '
        command_string += '--phi1 '+str(phi1)+' '
        command_string += '--signal_mag '+str(signal_mag)+' '
        command_string += '--min_run_index '+str(min_run_index)+' '
        command_string += '--max_run_index '+str(max_run_index)+' '
        command_string += '--data_folder_name '+str(data_folder_name)+' '
        command_string += '--max_t_shift '+str(max_t_shift)+' '
        command_string += '--frequency '+str(frequency)+' '
        command_string += '--af_list '+af_list+' '
        command_string += '--spect_num '+str(spect_num)+' '
        command_string += '--save_verbosity_index '+str(save_verbosity_index)+' '
        command_string += '--input_filename '+input_filename+' '
        command_string += '--pulse_filename '+pulse_filename+' '
        command_string += '--total_t '+str(total_t)+' '
        command_string += '--x10 '+str(x10)+' '
        command_string += '--z10 '+str(z10)+' '
        command_string += '--x20 '+str(x20)+' '
        command_string += '--z20 '+str(z20)+' '
        command_string += '--x30 '+str(x30)+' '
        command_string += '--z30 '+str(z30)+' '
        command_string += '--t_sig_start '+str(t_sig_start)+' '
        if not noise_flag:
            command_string += '--noise_flag '
        command_string += '--A1 '+str(A1)+' '
        command_string += '--pulse1_width '+str(pulse1_width)+' '
        command_string += '--pulse1_center '+str(pulse1_center)+' '
        command_string += '--pulse1_height '+str(pulse1_height)+' '
        command_string += '--pulse1_type '+str(pulse1_type)+' '
        command_string += '--pulse2_width '+str(pulse2_width)+' '
        command_string += '--pulse2_center '+str(pulse2_center)+' '
        command_string += '--pulse2_height '+str(pulse2_height)+' '
        command_string += '--pulse2_type '+str(pulse2_type)+' '
        command_string += '--pulse3_width '+str(pulse3_width)+' '
        command_string += '--pulse3_center '+str(pulse3_center)+' '
        command_string += '--pulse3_height '+str(pulse3_height)+' '
        command_string += '--pulse3_type '+str(pulse3_type)+' '
        command_string += '--sensor_increment '+str(sensor_increment)+' '
        command_string += '--phi3 '+str(phi3)+' '
        command_string += '--del_t_ff '+str(del_t_ff)+' '
        command_string += '--Cff1 '+str(Cff1)+' '
        command_string += '--Cff2 '+str(Cff2)+' '
        command_string += '--Cff3 '+str(Cff3)+' '
        command_string += '--t_offset1 '+str(t_offset1)+' '
        command_string += '--t_offset2 '+str(t_offset2)+' '
        command_string += '--t_offset3 '+str(t_offset3)+' '
        command_string += '--gamma1 '+str(gamma1)+' '
        command_string += '--gamma2 '+str(gamma2)+' '
        command_string += '--alpha1 '+str(alpha1)+' '
        command_string += '--alpha2 '+str(alpha2)+' '
        command_string += '--beta1 '+str(beta1)+' '
        command_string += '--C1 '+str(C1)+' '
        command_string += '--kappa12 '+str(kappa12)+' '
        command_string += '--abs_pulse_location_flag '+str(abs_pulse_location_flag)+' '
        command_string += '--snap_flag '+str(snap_flag)+' '
        
        command_string += ' | tee ./output_dirs/data_'+str(data_folder_name)+'/logfile_all_runs_sig_'+str(A2_block_list[i])+\
                      '_'+str(t_shift_superblock_lists[j][0])+'_'+str(t_shift_superblock_lists[j][-1])+'.txt"'
        
        if run_flag:
            os.system(command_string)
        else:
            print(str(ind+1))
            print(command_string)
            print("\n")
        ind += 1
            
            
            
            
            
            
