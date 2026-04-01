"""
exec(open("sweep_over_t_analysis.py").read())
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import argparse
import os
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
parser.add_argument('-s','--sweep_num',default=1,type=int) #22
parser.add_argument('-r','--run_num',default=1,type=int)
parser.add_argument('-sf','--save_flag',default=1,type=int)
parser.add_argument('-ts','--time_shift_flag',default=1,type=int)
parser.add_argument('-af','--average_flag',default=1,type=int)
args = parser.parse_args()

sweep_num = int(args.sweep_num)
run_num = int(args.run_num)
save_flag = bool(args.save_flag)
time_shift_flag = bool(args.time_shift_flag)
average_flag = bool(args.average_flag)

dpath = './sweeps/sweep'+str(sweep_num)+'/figs/'
indirpath = './sweeps/sweep'+str(sweep_num)+'/input_files/'

flist = glob.glob(dpath+'*')
t_list = []
for n in range(len(flist)):
    if flist[n].split('.')[-1] != 'png':
        t = int(flist[n].split('_')[-1])
        t_list.append(t)

t_list.sort()

pn = []
ps = []
t_sublist = []
for t in t_list:
    fpath = dpath+'figs_data_max_t_shift_'+str(t)+'/figs_run'+str(run_num)+'/figs_run_max_t_shift_'+str(t)+'_results.hdf'

    f = h5py.File(fpath,'r')
    
    t0_per_amp_fact = int(len(f['parameter_list0'])/len(f['parameter_avg_list0']))
    if average_flag:
        pn.append(f['parameter_avg_list0'][-1])
        ps.append(f['parameter_avg_list_max_t_shift_'+str(t)][-1])
    else:
        
        
        pn.extend(f['parameter_list0'][-1*t0_per_amp_fact:])
        ps.extend(f['parameter_list_max_t_shift_'+str(t)][-1*t0_per_amp_fact:])
    
    f.close()
    
    if not average_flag:
        fi = load_dynamic_variables(indirpath,'max_t_shift_'+str(t))
        max_t_shift = fi.max_t_shift
        num_t_steps = fi.num_t_steps
        t_step = max_t_shift/num_t_steps
        t_sublist_i = list(np.arange(t,t+max_t_shift,t_step))
        t_sublist.extend(t_sublist_i)
        t_sublist,pn,ps = zip(*sorted(zip(t_sublist,pn,ps)))
        t_sublist = list(t_sublist)
        pn = list(pn)
        ps = list(ps)

pltcolors = ['tab:blue','tab:orange','tab:green']

if average_flag:
    t_plot_list = t_list
else:
    t_plot_list = t_sublist

plt.figure()
plt.plot(t_plot_list,pn,color=pltcolors[0])
plt.plot(t_plot_list,ps,color=pltcolors[1])
plt.plot(t_plot_list,pn,'o',color=pltcolors[0])
plt.plot(t_plot_list,ps,'o',color=pltcolors[1])
#if not average_flag:
#    plt.xlabel('data point number')
if time_shift_flag:
    plt.xlabel('Time shift (s)')
else:
    plt.xlabel('Total time (s)')
plt.ylabel('Mean detection coefficient')
plt.legend(["noise","noise+signal"])
if save_flag:
    if average_flag:
        plt.savefig(dpath+'sweep_over_t_avg_sweep'+str(sweep_num)+'_run'+str(run_num)+'.png',bbox_inches='tight')
    else:
        plt.savefig(dpath+'sweep_over_t_sweep'+str(sweep_num)+'_run'+str(run_num)+'.png',bbox_inches='tight')

plt.show(block=False)










