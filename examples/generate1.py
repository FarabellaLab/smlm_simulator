
# Intro
# 
# Generate a single set of simulated experiments and save them



import numpy as np
import pandas as pd
import os
import numpy as np
from smlm_simulator.various import *
from smlm_simulator.simulation import *
# from pipeline.utils.simulation import *
import multiprocessing as mp
# from CIMA.TEMPy.ScoringFunctions import ScoringFunctions
# from CIMA.segments.SegmentGaussian import TransformBlurrer
# from CIMA.utils.WritePDB import *
# from CIMA.utils.Visualization import *
import sys

import warnings
warnings.filterwarnings("ignore")

overwriteOuts()

probes_parameters={
        'probes_per_mb': 5000*0.75, # <5000 # efficiency 25,50,75,100
        'segment_width_nm': 500 # fixed for biological reasons (computed from the average folding ratio of chromatin (diameter=0.01*genomic_length))
    }
combination_parameters={
        'segment_length_nm': 1000000, # same as the target
        'segments_per_sim': 2, # same as the target
        'segs_per_comb_is_random': False, # set false to have always same number
        'num_simulations': 20,
        'max_shift_amount_nm': 3000,
        'arrangement': 'lattice',
        'random_seed': 0
    }
localization_parameters={
        'fov_size': np.array([10000, 4000, 4000]), # same as the target
        'labels_per_probe': 2, # fixed for methodological reasons
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

output_dir = '/home/ipiacere@iit.local/Desktop/tmp/generated_data'
os.makedirs(output_dir, exist_ok=True)

dls, lengths = getChr21DLs()



single_probes_parameters = probes_parameters.copy()
single_localization_parameters = localization_parameters.copy()
single_combination_parameters = combination_parameters.copy()

sims_dfs = simulate(dls, lengths, verbose=0,
                                        probes_parameters=probes_parameters,
                                        combination_parameters=combination_parameters,
                                        localization_parameters=localization_parameters)

pars_dict = {**probes_parameters, **combination_parameters, **localization_parameters}

saveSimulatedPointClouds(sims_dfs, destination_folder = output_dir, parameters_dict=pars_dict)



