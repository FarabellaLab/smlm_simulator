
# Intro
# 
# Generate a dataset containing the 4 scenarios that we use for assessment of clustering algorithms



import numpy as np
import pandas as pd
import os
import numpy as np
from smlm_simulator.various import *
from smlm_simulator.simulation import *
import multiprocessing as mp
import sys

import warnings
warnings.filterwarnings("ignore")

# Overwrite stdout and stderr functions so that each line in the ouput starts with a string representing the current time. It's just for easier debugging.
overwriteOuts()

# Specify all the parameters. You will change some of them as needed for each of scenarios.
probes_parameters={
        'probes_per_mb': 5000*0.75, # <5000 # efficiency 25,50,75,100
        'segment_width_nm': 500 # fixed for biological reasons (computed from the average folding ratio of chromatin (diameter=0.01*genomic_length))
    }
combination_parameters={
        'segment_length_nm': 1000000, # same as the target
        'segments_per_sim': 2, # same as the target
        'segs_per_comb_is_random': False, # set false to have always same number
        'selection_procedure': 'sparse_percentile',
        'num_simulations': 20,
        'max_shift_amount_nm': 1000,
        'arrangement': 'random',
        'random_seed': 0
    }
localization_parameters={
        'fov_size': np.array([7000, 7000, 7000]), # same as the target
        'labels_per_probe': 2, # fixed for methodological reasons
        'precision_mean': np.array([10.,10.,50.]),
        'precision_sd': np.array([5.,5.,20.]), # same as the target
        'bleaching_prob': 0.25, # 0.1,0.2,0.5
        'unbound_probes_per_bound': 160,
        'attraction_towards_bound': False,
        'attraction_iterations': 7,
        'attraction_radius_nm': 1000,
        'probe_width_nm': 120,
        'attraction_factor': 10, # lennard jones epsilon=10
        'random_noise_per_blinks': 50,
        'detection_rate': 1
    }

# output_dir = '/home/ipiacere@iit.local/Desktop/data/synth_data/scenarios2'
# output_dir = '/work/ipiacere/data/synth_data/scenarios2'
output_dir = '/home/ipiacere@iit.local/synth1'

if(not os.access(output_dir, os.W_OK)):
    raise ValueError('Can\'t write in the specified folder: ' + output_dir)


# Get low-resolution traces of real chromosomes 
dls, lengths = getChr21DLs()


# Copy the generic parameters defined before.
single_probes_parameters = probes_parameters.copy()
single_localization_parameters = localization_parameters.copy()
single_combination_parameters = combination_parameters.copy()

# This function is used to generate a set of simulated experiments and saving them in the specified folder.
def doSingle(dls1, lengths1, outdir, probes_params, combination_params, localization_params):
    sims_dfs = simulate(dls1, lengths1, verbose=1,
                                    probes_parameters=probes_params,
                                    combination_parameters=combination_params,
                                    localization_parameters=localization_params)
    pars_dict = {**probes_params, **combination_params, **localization_params}
    saveSimulatedPointClouds(sims_dfs, destination_folder=outdir, parameters_dict=pars_dict)

pool = mp.Pool(16)
jobs = {}

# Varying the length (in basepairs) of the segments
for seg_len_bp in [1000000,500000][:1]:
    print('seg_len_bp: ', seg_len_bp)
    single_combination_parameters['segment_length_nm'] = seg_len_bp

    # varying distance
    for dist in [100,500,1000,2000,3000][:1]:
        print('dist: ', dist)
        single_combination_parameters['max_shift_amount_nm'] = dist
        single_combination_parameters['arrangement'] = 'lattice'
        single_localization_parameters['unbound_probes_per_bound'] = localization_parameters['unbound_probes_per_bound']*1.0
        single_localization_parameters['random_noise_per_blinks'] = localization_parameters['random_noise_per_blinks']*1.0
        single_combination_parameters['segments_per_sim'] = 2

        output_dir2 = output_dir + '/seg_len_bp_%i/varying_distance/distance_%inm'%(seg_len_bp, dist)
        jobs[output_dir2] = pool.apply_async(doSingle, (dls, lengths, output_dir2, single_probes_parameters, single_combination_parameters, single_localization_parameters))
    
    # varying noise
    for noise_level in [0.25,0.5,1.0,1.5]:
        print('noise_level: ', noise_level)
        single_combination_parameters['max_shift_amount_nm'] = 1000
        single_combination_parameters['arrangement'] = 'lattice'
        single_localization_parameters['unbound_probes_per_bound'] = localization_parameters['unbound_probes_per_bound']*noise_level
        single_localization_parameters['random_noise_per_blinks'] = localization_parameters['random_noise_per_blinks']*noise_level
        single_combination_parameters['segments_per_sim'] = 2

        output_dir2 = output_dir + '/seg_len_bp_%i/varying_noise/noise_level_%.2f'%(seg_len_bp, noise_level)
        jobs[output_dir2] = pool.apply_async(doSingle, (dls, lengths, output_dir2, single_probes_parameters, single_combination_parameters, single_localization_parameters))
    
    # varying num
    for num in [2,3,5,10,15,20,40,80]:
        print('num: ', num)
        single_combination_parameters['max_shift_amount_nm'] = 1000
        single_combination_parameters['arrangement'] = 'random'
        single_localization_parameters['unbound_probes_per_bound'] = localization_parameters['unbound_probes_per_bound']*1.0
        single_localization_parameters['random_noise_per_blinks'] = localization_parameters['random_noise_per_blinks']*1.0
        single_combination_parameters['segments_per_sim'] = num

        output_dir2 = output_dir + '/seg_len_bp_%i/varying_num/num_%i'%(seg_len_bp, num)
        jobs[output_dir2] = pool.apply_async(doSingle, (dls, lengths, output_dir2, single_probes_parameters, single_combination_parameters, single_localization_parameters))


ress = {}
for k,v in jobs.items():
    res1 = v.get()
    ress[k] = res1
    print(k)
    print(res1)
    print('-----')




