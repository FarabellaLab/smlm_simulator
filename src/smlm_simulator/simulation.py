import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
# from pipeline.utils.various import *
import opensimplex
from smlm_simulator.various import *

# from numba import jit

def getChr21DLs(min_seg_length=1000000):
    '''
    Gives a set of low resolution ball-and-stick representations of real chromosomes 21,
    which you can use as starting points for simulations.


    Returns:
    * a list of numpy array of shape (n,3) representing diffraction limited coordinates of the segments
    * a list of floats representing lengths of the segments in terms of basepairs
    '''
    if(os.path.isfile('/home/ipiacere@iit.local/Desktop/data/from_Irene/ball_and_stick/chromosome21.tsv')):
        df0 = pd.read_csv('/home/ipiacere@iit.local/Desktop/data/from_Irene/ball_and_stick/chromosome21.tsv', sep='\t')
    elif(os.path.isfile('/work/ipiacere/data/from_Irene/ball_and_stick/chromosome21.tsv')):
        df0 = pd.read_csv('/work/ipiacere/data/from_Irene/ball_and_stick/chromosome21.tsv', sep='\t')
    else:
        df0 = pd.read_csv('https://zenodo.org/records/3928890/files/chromosome21.tsv?download=1', sep='\t')
    df = df0[['X(nm)', 'Y(nm)', 'Z(nm)','Genomic coordinate','Chromosome copy number']] \
        .rename({
            'X(nm)': 'x',
            'Y(nm)': 'y',
            'Z(nm)': 'z',
            'Genomic coordinate':'gen_coord',
            'Chromosome copy number': 'chr_copy_num'},
        axis=1)
    
    # identify chr copies without missing x,y,z values
    is_row_null = df[['x','y','z']].isnull().any(axis=1)
    df2 = pd.DataFrame(is_row_null)
    df2['chr_copy_num'] = df['chr_copy_num']
    null_counts = df2.groupby(['chr_copy_num']).sum()
    good_copies = np.array(null_counts.index[(null_counts==0).values.flatten()])
    # good_copies = [g for g in good_copies if g not in [3250]]

    # parse genetic start, end and center
    df_cleaned = df.copy()
    df_cleaned['genetic_start'] = df_cleaned['gen_coord'].apply(lambda x: int(x.split(':')[1].split('-')[0]))
    df_cleaned['genetic_end'] = df_cleaned['gen_coord'].apply(lambda x: int(x.split('-')[1]))
    df_cleaned['genetic_center'] = ((df_cleaned['genetic_start'] + df_cleaned['genetic_end'])/2).astype('int')
    df_cleaned = df_cleaned.sort_values('genetic_start', kind='stable')

    # collect positions and lengths
    # avoid sections with resolution larger than 50000bp
    dls = []
    lengths = []
    for i in good_copies:
        chrcopy_subdf = df_cleaned[df_cleaned['chr_copy_num'] == i].copy()
        diffs_bp = chrcopy_subdf['genetic_start'].diff().values
        split_inds = np.where(diffs_bp!=50000)[0][1:]
        subdfs = [chrcopy_subdf.iloc[split_inds[i]:split_inds[i+1],:].copy() for i in range(len(split_inds)-1)]
        for df1 in subdfs:
            length = df1['genetic_end'].max() - df1['genetic_end'].min()
            if(len(df1)<=1 or length<min_seg_length):
                continue
            dls.append(df1[['x','y','z']].values)
            lengths.append(length)
    return dls, lengths

def getRandomIntsAtPercentileRegions(inds, values, num):
    '''
    Given 'indices' and corresponding 'values', divide the distribution of 'values' in 'num' quantiles and select randomly one index from each of them.
    '''
    df = pd.DataFrame()
    df['inds'] = inds
    df['values'] = values
    df['percs'] = df['values'].rank(pct=True)
    bot = 0.25
    top = 0.75
    perc_steps = [bot + (top-bot)*i/num for i in range(num+1)]
    selected_inds = []
    for i in range(num):
        dfslice = df.loc[((df['percs']>=perc_steps[i])*(df['percs']<=perc_steps[i+1])).astype('bool')]
        selected = random.choice(dfslice['inds'].tolist())
        selected_inds.append(selected)
    return selected_inds

def saveSimulatedPointClouds(sims_dfs, destination_folder = '/home/ipiacere@iit.local/Desktop/data/synth_data/test', parameters_dict={}, remove_old=True):
    '''
    Save the provided dataframes in the destination folder. Moreover write the simulation parameters in a file in the same folder.
    '''
    import os
    import glob

    os.makedirs(destination_folder, exist_ok=True)

    if(remove_old):
        files = glob.glob(destination_folder+'/*')
        for f in files:
            os.remove(f)


    import json
    with open(destination_folder+'/config.json', 'w') as f:
        json.dump({k:(v.tolist() if type(v)==np.ndarray else v) for k,v in parameters_dict.items()}, f, indent=2)

    for i, sim_df in enumerate(sims_dfs):
        sim_df.to_csv(destination_folder+'/sim%i.csv'%i, index=False)

def simulate(dl_list, lengths,
    probes_parameters={},
    combination_parameters={},
    localization_parameters={},
    verbose=0):
    '''
    Generate a defined number of simulated SMLM experiments and return them as pandas dataframes.


    Arguments:
    * dl_list: list, list of np arrays of shape (n,3)
    * lengths: list, lengths in bp of the dls
    * ...parameters
    * probes_per_mb: int, how many probes to position in 1megabase of chromatin
    * segment_width_nm: unused, is is actually variable along the segment and is automatically computed with another function (getWidths)
    * segment_length_nm: int, length in basepairs (error in the variable name) of the segment you generate
    * segments_per_sim: int, how many segments to insert in a single experiment
    * segs_per_comb_is_random: bool, should the number of segments be chosen randomly or should it be exaclty 'segments_per_sim'
    * selection_procedure: ['random', 'sparse_percentile'], whether to select the segments randomly or making sure that the entire distribution of radii of gyration is represented
    * num_simulations: int, how many experiments to generate
    * max_shift_amount_nm: float, distance between segments. Has a different meaning depending on 'arrangement'
    * arrangement: ['lattice', 'tetrahedron', 'random'], how to arrange segments in the field of view.
        If lattice the centers of mass of the segments are positioned on a lattice in which successice elements along an axis are spaced 'max_shift_amount_nm' nm apart.
        If tetrahedron the CoMs of the segments are positioned on a tetrahedron and are all 'max_shift_amount_nm' nms apart.
        If random they are randomly positioned in the field of view, with the constraint that mean nearest neighbor CoM distance is not more than 50nm from 'max_shift_amount_nm'/
    * random_seed: int, random seed for reproducibility of the the simulation
    * fov_size: np.array of shape (3,), size of the field of view along x,y,z axes in nm
    * labels_per_probe: int, how many labels (those which emit blinks) should on average be assigned to a probe
    * precision_mean: float, the mean precision (nm) of localizations
    * precision_sd: float, the standard deviation (nm) of the precision of localizations
    * bleaching_prob: float in range 0.0-1.0, bleaching probability. the probability of each blink of being the last one for that label
    * unbound_probes_per_bound: float, how many unbound probes are positioned in 1 cubic megabase of FoV (misleading variable name)
    * is_unbound_rate_random: bool, is unbound rate random (normal distribution with mean 'unbound_probes_per_bound' and std 52) or is it fixed to 'unbound_probes_per_bound'
    * attraction_towards_bound: bool, are unbound probes attracted also to bound ones, or just to other unbound probes
    * attraction_iterations: int, how many iterations of attraction computation should be performed
    * attraction_radius_nm: float, maximum distance of two probes to be subject to mutual attraction
    * probe_width_nm: float, distance below which two probes are not subject to mutual attraction
    * attraction_factor: float, amount of attraction between probes
    * random_noise_per_blinks: float, how many false localizations probes are positioned in 1 cubic megabase of FoV (misleading variable name)
    * detection_rate: unused, the detection rate is actually always 100%

    

    Returns:
    * list of pandas dataframes representing simulated experiments
        each df with have columns
            x,y,z: coordinates of the point
            type:
                0: localization coming from a bound probe,
                1: localization coming from an unbound probe (noise)
                2: false localization (noise)
            cluster-ID: index of the chromatin segment from which the localization was generated. -1 if the localization if false or it was generated from an unbound probe (noise)
            moleculeIndex:
                0...: index of the (bound) probe from which the localization was generated
                -1: false localization or localization coming from an unbound probe (noise)
            precisionx, precisiony, precisionz: mean localization precision for all localizations in the experiment
            precisionx_actual, precisiony_actual, precisionz_actual: localization precision of the probe from which the localization was generated. All the localizations from the same probe have the same localization precision.

    '''

    probes_parameters = dict({
        'probes_per_mb': 200,
        'segment_width_nm': 500
    }, **probes_parameters)

    combination_parameters = dict({
        'segment_length_nm': 1000000,
        'segments_per_sim': 10,
        'segs_per_comb_is_random': False,
        'selection_procedure': 'random',
        'num_simulations': 1,
        'max_shift_amount_nm': 4000,
        'arrangement': 'lattice',
        'random_seed': 9
    }, **combination_parameters)

    localization_parameters = dict({
        'fov_size': np.array([8000, 15000, 5000*2]),
        'labels_per_probe': 5,
        'precision_mean': np.array([14.,15.,48.]),
        'precision_sd': np.array([6.,7.,19.]),
        'bleaching_prob': 0.5,
        'unbound_probes_per_bound': 5,
        'is_unbound_rate_random': True,
        'attraction_towards_bound': True,
        'attraction_iterations': 10,
        'attraction_radius_nm': 5000,
        'probe_width_nm': 50,
        'attraction_factor': 5e7,
        'random_noise_per_blinks': 0.1,
        'detection_rate': 0.7
    }, **localization_parameters)

    np.random.seed(combination_parameters['random_seed'])
    random.seed(combination_parameters['random_seed'])

    copies_probes_dfs = {}

    if(verbose>0): print('interpolating')
    for i,dl in enumerate(tqdm(dl_list, disable=verbose<1)):
        length_bp = lengths[i]
        probes_num = int((length_bp/1000000)*probes_parameters['probes_per_mb'])

        probes_df = getInterpolatedCoords(dl,
                                        probes_num,
                                        length_bp,
                                        segment_width_nm=probes_parameters['segment_width_nm'])
        copies_probes_dfs[i] = probes_df
    
    if(verbose>0): print('combining')
    combs_dfs, combs = getChromosomeSegmentsCombinations(list(copies_probes_dfs.values()), segment_length_bp=combination_parameters['segment_length_nm'],
                                      approx_extracted_segments=3500,
                                      remove_extremes=False,
                                      num_combinations=combination_parameters['num_simulations'],
                                      avg_segs_per_combination=combination_parameters['segments_per_sim'],
                                      selection_procedure=combination_parameters['selection_procedure'],
                                      segs_per_comb_is_random=combination_parameters['segs_per_comb_is_random'],
                                      max_shift_amount=combination_parameters['max_shift_amount_nm'],
                                      arrangement=combination_parameters['arrangement'],
                                      return_definitions=True,
                                      seed=combination_parameters['random_seed'],
                                      verbose=verbose)

    if(verbose>0): print('localizing')
    simulated_dfs = []
    for comb_ind, comb_df in enumerate(combs_dfs):
        if(verbose>0): print('started SIMULATION %i'%comb_ind)
        params = {
            'inDf': comb_df,
            'size':localization_parameters['fov_size'],
            'detectionRate':localization_parameters['detection_rate'],
            'noiseLevel':localization_parameters['random_noise_per_blinks'],
            'unboundMoleculesRate':localization_parameters['unbound_probes_per_bound'],
            'isUnboundRateRandom': localization_parameters['is_unbound_rate_random'],
            'noisescales':[5],
            'noiseamplitudes':[1],
            'noisesmoothness':5,
            'wanted_mean' : localization_parameters['precision_mean'],
            'wanted_sd' : localization_parameters['precision_sd'],
            'averageLabelsPerMol' : localization_parameters['labels_per_probe'],
            'blinkGeomProb' : localization_parameters['bleaching_prob'],
            'gravity_steps':localization_parameters['attraction_iterations'],
            'gravity_amount':localization_parameters['attraction_factor'],
            'gravity_radius':localization_parameters['attraction_radius_nm'],
            'boundMoleculesAttract':localization_parameters['attraction_towards_bound'],
            'min_dist':localization_parameters['probe_width_nm'],
            'verbose':verbose-1
            }
        simulated_df = getStormSimulatedPoints(**params)
        if(verbose>0): print('SIMULATION %i completed'%comb_ind)
        simulated_dfs.append(simulated_df)
    return simulated_dfs



def getBoxSideGivenNumAndNNDist(num, expected_nn_dist):
    from sklearn.neighbors import NearestNeighbors
    box_sides = np.linspace(1000,8000,30)
    df = pd.DataFrame(columns=[''])
    sample_size=10
    for side in box_sides:
        for sample_i in range(sample_size):
            points = np.random.uniform(np.array([0]*3), np.array([side]*3), (num,3))
            nn_dists = NearestNeighbors(n_jobs=8).fit(points).kneighbors(n_neighbors=1, return_distance=True)[0][:,0]
            df = df.append({'side':side,
                            'sample_i': sample_i,
                            'nn_dist_mean':nn_dists.mean()}, ignore_index=True)
    df = df.groupby('side').mean().reset_index()
    df['deviation'] = np.abs(df['nn_dist_mean']-expected_nn_dist)
    return df['side'][np.argmin(df['deviation'])]

def getPositionsGivenNumAndNNDist(num, expected_nn_dist):
    from sklearn.neighbors import NearestNeighbors
    box_sides = np.linspace(1000,8000,30)
    df = pd.DataFrame(columns=[''])
    sample_size=30
    for side in box_sides:
        for sample_i in range(sample_size):
            points = np.random.uniform(np.array([0]*3), np.array([side]*3), (num,3))
            nn_dists = NearestNeighbors(n_jobs=8).fit(points).kneighbors(n_neighbors=1, return_distance=True)[0][:,0]
            if(np.abs(nn_dists.mean()-expected_nn_dist)<50):
                return points
    raise ValueError('No appropriate arrangement found')



def getLocalizationsFromCoords(moleculeCoords, averageLabelsPerMol, blinkGeomProb,
    precisionMeanLog, precisionSdLog, per_probe_precision=True):
    '''
    Given probes positions, returns localizations' positions
    '''

    numDimensions = moleculeCoords.shape[1]

    numMolecules = moleculeCoords.shape[0]
    numLabels = averageLabelsPerMol*numMolecules
    label_to_mol_index = np.random.choice(np.arange(numMolecules), numLabels, replace=True)
    label_to_mol_index.sort()
    labelsPerMol = np.zeros(numMolecules, dtype='int')
    inds, counts = np.unique(label_to_mol_index, return_counts=True)
    labelsPerMol[inds] = counts
    labelCoords = np.repeat(moleculeCoords, labelsPerMol, axis=0)

    numBlinksPerLab = np.random.geometric(blinkGeomProb, numLabels)
    num_total_blinks = numBlinksPerLab.sum()
    
    blink_to_mol_index = np.repeat(label_to_mol_index, numBlinksPerLab, axis=0)
    if(per_probe_precision):
        locPrec = np.random.lognormal(precisionMeanLog, precisionSdLog, (numLabels, numDimensions))
        blink_localization_precision = np.repeat(locPrec, numBlinksPerLab, axis=0)
    else:
        blink_localization_precision = np.random.lognormal(precisionMeanLog, precisionSdLog, (num_total_blinks, numDimensions))

    mean_blink_location_wrt_label = np.zeros((num_total_blinks, numDimensions))
    blink_shift_wrt_label = np.random.normal(mean_blink_location_wrt_label, blink_localization_precision, mean_blink_location_wrt_label.shape)
    detectionCoords = np.repeat(labelCoords, numBlinksPerLab, axis=0) + blink_shift_wrt_label

    return pd.DataFrame(np.concatenate((detectionCoords, blink_to_mol_index.reshape(-1,1), blink_localization_precision), axis=1),
        columns=['x','y','z','moleculeIndex','precisionx_actual','precisiony_actual','precisionz_actual'])

def simulateSTORMAdvanced(moleculeCoords, averageLabelsPerMol, blinkGeomProb,
                  precisionMeanLog, precisionSdLog, detectionRate,
                  falseDetectionRate, unboundMoleculesRate, fieldLimit, isUnboundRateRandom=True,
                  noisescales=[3], noiseamplitudes=[1], noisesmoothness=10,
                  gravity_steps=5, gravity_amount=1/2, gravity_radius=100,
                  boundMoleculesAttract=False, min_dist=500,
                  verbose=0):
    '''
    Port of the simulateSTORM function from R to python with some modifications.
    The function performs the folllowing steps:
    - Adding unbound probes
    - Computing localizations' coordinates
    - Removing detections outside fov
    - Adding false detections
    '''

    numDimensions = moleculeCoords.shape[1]
    assert(numDimensions == 3)
    assert(precisionMeanLog.shape[0] == numDimensions)
    assert(precisionSdLog.shape[0] == numDimensions)
    assert(fieldLimit.shape[0] == 2)
    assert(fieldLimit.shape[1] == numDimensions)

    if(verbose>0): print('Adding unbound molecules')
    if(unboundMoleculesRate is None):
        sampledUnboundMoleculesRate=0
    else:
        if(isUnboundRateRandom):
            sampledUnboundMoleculesRate = max(np.random.normal(unboundMoleculesRate, 52),0)
        else:
            sampledUnboundMoleculesRate = unboundMoleculesRate
        unboundMoleculesNum = int(sampledUnboundMoleculesRate*(fieldLimit[1]-fieldLimit[0]).prod()*1e-9)
    unboundMoleculesCoords = np.empty(shape=(0,3))
    if(unboundMoleculesNum>0):
        unboundMoleculesCoords = getRandomPointsInCube(unboundMoleculesNum, fieldLimit, scales=noisescales, amplitudes=noiseamplitudes, smoothness=noisesmoothness)

        if(boundMoleculesAttract):
            all_coords = np.concatenate((unboundMoleculesCoords,moleculeCoords),axis=0)
            which_fixed_coords = np.full(len(all_coords), False)
            which_fixed_coords[len(unboundMoleculesCoords):] = True
        else:
            all_coords = unboundMoleculesCoords
            which_fixed_coords = np.full(len(all_coords), False)
        if(verbose>0): print('Computing unbound particles\' physics')
        allMoleculesCoordsGravitated = applyGravitySteps(all_coords, which_fixed_coords=which_fixed_coords,
                                                         iters_num=gravity_steps, shift_amount=gravity_amount,
                                                         radius=gravity_radius, min_dist=min_dist,
                                                         verbose=verbose>1)
        unboundMoleculesCoords = allMoleculesCoordsGravitated[np.invert(which_fixed_coords)]
    allMoleculeCoords = np.concatenate((moleculeCoords, unboundMoleculesCoords), axis=0)


    if(verbose>0): print('Computing particles\' localizations')
    detectionCoords = getLocalizationsFromCoords(allMoleculeCoords, averageLabelsPerMol,
            blinkGeomProb, precisionMeanLog, precisionSdLog, per_probe_precision=True)
    if(len(detectionCoords.loc[detectionCoords['moleculeIndex']<len(moleculeCoords)])>0):
        detectionCoords.loc[detectionCoords['moleculeIndex']<len(moleculeCoords),'type'] = 0
    if(len(detectionCoords.loc[detectionCoords['moleculeIndex']>=len(moleculeCoords)])>0):
        detectionCoords.loc[detectionCoords['moleculeIndex']>=len(moleculeCoords),'type'] = 1
        detectionCoords.loc[detectionCoords['moleculeIndex']>=len(moleculeCoords),'moleculeIndex'] = -1


    if(verbose>0): print('Removing detections outside fov')
    is_detection_inside = ((detectionCoords[['x','y','z']].values > fieldLimit[0])*(detectionCoords[['x','y','z']].values < fieldLimit[1])).prod(axis=1).astype('bool')
    detectionCoords = detectionCoords.loc[is_detection_inside,:]
        

    if(verbose>0): print('Adding false detections')
    if(falseDetectionRate is None):
        pass
    else:
        numFalseDetections = int(falseDetectionRate*(fieldLimit[1]-fieldLimit[0]).prod()*1e-9)
    if(numFalseDetections > 0):
        accepted_false_detections = np.random.uniform(fieldLimit[0], fieldLimit[1], (numFalseDetections,numDimensions))
        accepted_false_detections_df = pd.DataFrame(accepted_false_detections,
            columns = ['x','y','z'])
        accepted_false_detections_df['moleculeIndex'] = -1
        accepted_false_detections_df['type'] = 2
        detectionCoords = pd.concat((detectionCoords, accepted_false_detections_df), axis=0)
    

    if(len(detectionCoords)==0):
        detectionCoords['type'] = None
    detectionCoords['moleculeIndex'] = detectionCoords['moleculeIndex'].astype('int')
    detectionCoords['type'] = detectionCoords['type'].astype('int')
    return detectionCoords.reset_index(drop=True)

def getStormSimulatedPoints(inDf, size=33000, noiseLevel=0.1, unboundMoleculesRate=0.5, isUnboundRateRandom=True, detectionRate=0.7, verbose=0,
    noisescales=[3], noiseamplitudes=[1], noisesmoothness=10,
    wanted_mean = np.array([13.865, 14.55, 48.004]),
    wanted_sd = np.array([6.102, 6.966, 18.972]),
    averageLabelsPerMol = 5,
    blinkGeomProb = 0.5,
    gravity_steps=5, gravity_amount=1/2, gravity_radius=100, boundMoleculesAttract=False,
    min_dist=500):

    df = pd.DataFrame(inDf)
    moleculeCoords = df[['x','y','z']].values
    
    # How to find log-normal parameters based on the mean and std you want to obtain
    # (numpy log-normal function takes the norm and sd of the underlying normal distribution,
    # but we're interested in obtaining a specific mean and sd from the log-normal distriubtion itself):
    # https://www.johndcook.com/blog/2022/02/24/find-log-normal-parameters/
    
    falseDetectionRate = noiseLevel
    
    wanted_var = wanted_sd**2
    var = np.log((wanted_var/(wanted_mean**2)) + 1)
    mu = np.log(wanted_mean) - var/2
    sd = np.sqrt(var)
    precisionMeanLog = mu
    precisionSdLog = sd

    if(len(moleculeCoords)>0):
        center = getBoxCenter(moleculeCoords)
    else:
        center = np.array([10000,10000,10000])
    
    if isinstance(size, list): size = np.array(size)
    top = center + size/2
    bottom = center - size/2
    fieldLimits = np.stack((bottom, top), axis=0)

    detectionList_df = simulateSTORMAdvanced(moleculeCoords,
                                averageLabelsPerMol, blinkGeomProb, precisionMeanLog,
                                  precisionSdLog, detectionRate, falseDetectionRate, unboundMoleculesRate, fieldLimits, isUnboundRateRandom=isUnboundRateRandom, verbose=verbose,
                                  noisescales=noisescales, noiseamplitudes=noiseamplitudes, noisesmoothness=noisesmoothness,
                                  gravity_steps=gravity_steps, gravity_amount=gravity_amount, gravity_radius=gravity_radius,
                                  boundMoleculesAttract=boundMoleculesAttract, min_dist=min_dist)

    
    moleculeToCluster = df['cluster-ID'].values
    detectionList_df['cluster-ID'] = -1 # for noise
    where_not_noise = (detectionList_df['moleculeIndex']!=-1).values
    molInd = detectionList_df.loc[where_not_noise, 'moleculeIndex'].values
    detectionList_df.loc[where_not_noise, 'cluster-ID'] = moleculeToCluster[molInd]

    detectionList_df[['precisionx','precisiony','precisionz']] = wanted_mean

    return detectionList_df

def getWidths(num, min_width=12, sampling_dist=100):
    '''
    Given the number of sampling points along a chromatin segment, returns randomly generate chromatin widths at those points.
    This is done by fixing to random values a subset (sefined by 'sampling_dist') of the sampling points,
    and interpolating linearly the others.

    Arguments:
        * num: number of sampling points
        * min_width: width to which the sampling points after the last fixed point are interpolated
        * sampling_dist: distance between successive fixed points
    
    Returns:
        * np.array of shape (num,) where each element represents the generated width of the corresponding sampling point on the chromatin segment
    '''
    widths = np.zeros(num)
    where_fixed_inds = range(0,num,sampling_dist)
    where_fixed_arr = np.zeros(num, dtype='bool')
    where_fixed_arr[where_fixed_inds] = True
    widths[where_fixed_arr] = np.random.normal(250, 80, size=where_fixed_arr.sum())
    # first and central sampling points along the segment
    for i in range(len(where_fixed_inds)-1):
        weights = np.linspace(1,0,sampling_dist+1)[1:-1]
        indstart = where_fixed_inds[i]
        indend = where_fixed_inds[i+1]
        widths[indstart+1:indend] = widths[indstart]*weights + widths[indend]*(1-weights)
    # sampling points after the last fixed point
    if(num>where_fixed_inds[-1]+1):
        weights = np.linspace(1,0,num - where_fixed_inds[-1]+1)[1:-1]
        widths[where_fixed_inds[-1]+1:] = widths[where_fixed_inds[-1]]*weights + min_width*(1-weights)
    return widths

def getInterpolatedCoords(coords, num=3000,
                          length_bp=45000000, segment_width_nm=20):
    '''
    Given a low resolution representation of a chromatin segment, returns a higher resolution representation of it.
    'num' and 'length_bp' define how high the resolution will be.

    Arguments:
        coords: np array in which each row is the set of coordinates of a point.
        Represents the points to interpolate
        num: number of points you want to have in the final representation of the segment
        length_bp: length that the segment is assumed to have

    '''

    from scipy import interpolate
    numdims = coords.shape[1]
    data = coords.transpose()
    #now we get all the knots and info about the interpolated spline
    tck, u= interpolate.splprep(data, k=3)
    #here we generate the new interpolated dataset, 
    #increase the resolution by increasing the spacing
    new = interpolate.splev(np.linspace(0,1,num), tck)
    interp_to_join = [np.array(n) for n in new]
    interp_coords_nm = np.stack(interp_to_join).T
    dist_bp = float(length_bp)/num
    interp_coords_bp = (np.arange(num)*dist_bp).astype('int')
    random_shifts = np.random.uniform(0,1, len(interp_coords_nm))
    random_dirs = np.random.normal(0.0, 1.0, interp_coords_nm.shape)
    random_dirs = random_dirs/np.linalg.norm(random_dirs, axis=1).reshape(-1,1)
    random_vecs = random_dirs*random_shifts.reshape(-1,1)*(getWidths(len(interp_coords_nm))).reshape(-1,1) # *(segment_width_nm/2)
    interp_coords_nm_with_shift = interp_coords_nm+random_vecs

    columns = ['x','y','z','bp'] if numdims == 3 else ['x','y','bp']
    return pd.DataFrame(np.concatenate((interp_coords_nm_with_shift, interp_coords_bp.reshape((-1,1))), axis=1), columns=columns)


def getOpenSimplexNoiseArray(cube_side=30, coords=None, scales=[1], amplitudes=[1], seed=0):
    '''
    Evaluate a simplex noise landscape at 'coords' coordinates, and normalize it.
    '''
    opensimplex.seed(seed)
    if(coords is None):
        coords = np.stack(np.where(np.zeros(shape=[cube_side]*3) == 0), axis=0).T
    val = np.zeros(coords.shape[0])
    amplitudes_sum = sum(amplitudes)
    amplitudes = [amp/amplitudes_sum for amp in amplitudes]
    for scale, amp in zip(scales, amplitudes):
        for i, coord in enumerate(coords):
            x,y,z = coord
            val[i] += opensimplex.noise3((x/cube_side)*scale, (y/cube_side)*scale, (z/cube_side)*scale)*amp
    return (val+len(amplitudes))/(2*len(amplitudes)) # normalization

def probToRange(x,rangemax=6):
    return (x-0.5)*rangemax
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getRandomPointsInCube(numFalseDetections, fieldLimit, scales=[1], amplitudes=[1], seed=0, smoothness=6):
    '''
    Get randomly positioned points in the cube defined by 'fieldLimit'.
    This uses simplex noise to have noise not uniformly distributed.
    '''
    numDimensions=3
    cube_side = (fieldLimit[1]-fieldLimit[0]).max()

    # compute the required number of samples needed to obtain about 'numFalseDetections' localizations
    rand_locs = np.random.uniform(fieldLimit[0], fieldLimit[1], (numFalseDetections,numDimensions))
    noise_at_rand_points = getOpenSimplexNoiseArray(cube_side, rand_locs, scales=scales, amplitudes=amplitudes, seed=seed)
    acceptance_probab = noise_at_rand_points.mean()
    num_trials = int(numFalseDetections/acceptance_probab)


    possible_false_detections = np.random.uniform(fieldLimit[0], fieldLimit[1], (num_trials,numDimensions))
    noise_at_possible_detections = getOpenSimplexNoiseArray(cube_side, possible_false_detections, scales=scales, amplitudes=amplitudes, seed=seed)

    noise_at_possible_detections = sigmoid(probToRange(noise_at_possible_detections, smoothness))
    random_vals = np.random.uniform(0,1, len(noise_at_possible_detections))
    accepted_false_detections = possible_false_detections[random_vals<noise_at_possible_detections,:]
    return accepted_false_detections


def getChromosomeSegmentsCombinations(chrs, segment_length_bp=1000000,
                                      approx_extracted_segments=5000,
                                      remove_extremes=False,
                                      num_combinations=10,
                                      avg_segs_per_combination=5,
                                      segs_per_comb_is_random=False,
                                      selection_procedure='random',
                                      max_shift_amount=3000,
                                      arrangement='lattice',
                                      return_definitions=False,
                                      seed = 0,
                                      verbose=0):
    '''

    Arguments:
        *chrs*: list of DataFrames each of which has x,y,z and bp columns.
        *approx_extracted_segments*: the number of segment to consider among all the possible ones.
            This is useful to limit the computations in case of a very large amount of possible segments.
    
            
    Return:
        a list of DataFrames with columns x,y,z,bp,cluster. Each df represents a set of segments.
    '''

    portions_indices = []
    segs_num = {}
    for copy_id,_ in enumerate(chrs):
        copy_df = chrs[copy_id]
        segs_num[copy_id] = int(np.ceil(copy_df['bp'].max()/segment_length_bp))
    tot_num = sum(segs_num.values())

    for copy_id,_ in enumerate(chrs):
        randvec = np.random.uniform(0,1,segs_num[copy_id])
        for seg_id in range(segs_num[copy_id]):
            if(randvec[seg_id] > (approx_extracted_segments/tot_num)): continue
            portions_indices.append((copy_id, seg_id))
    

    segments_features_dicts = []

    for copy_id,seg_id in tqdm(portions_indices, disable=verbose<1):
        copy_df = chrs[copy_id]
        segs_num = int(np.ceil(chrs[copy_id]['bp'].max()/segment_length_bp))
        where_in_range = (copy_df['bp']>= seg_id*segment_length_bp) * \
                copy_df['bp'] < (seg_id+1)*segment_length_bp
        where_in_range = where_in_range.astype('bool')
        selected_seg_df = copy_df[where_in_range].copy()
        segments_features_dicts.append(getClusterFeatures(selected_seg_df[['x','y','z']].values, selected_features=['num','gyr']))
    segments_features_df = pd.DataFrame(segments_features_dicts, index=portions_indices)

    selected_df = segments_features_df.copy()

    
    if(remove_extremes):
        perc25 = int(len(selected_df)*0.25)
        perc75 = int(len(selected_df)*0.75)
        selected_df = selected_df.sort_values('gyr').iloc[perc25:perc75, :]
    
    if(segs_per_comb_is_random):
        assignments = np.random.choice(np.arange(num_combinations), num_combinations*avg_segs_per_combination, replace=True)
        names, counts = np.unique(assignments, return_counts=True)
        num_segs_per_comb = {i:0 for i in range(num_combinations)}
        num_segs_per_comb.update({k:v for k,v in zip(names, counts)})
    else:
        num_segs_per_comb = {i:avg_segs_per_combination for i in range(num_combinations)}
    
    if(len(chrs) == 0):
        num_segs_per_comb = {i:0 for i in range(num_combinations)}

    combs_definitions = []
    for comb_id in range(num_combinations):
        if(selection_procedure=='random'):
            # randomly select segments among all the ones available
            comb_def = np.random.choice(selected_df.index, num_segs_per_comb[comb_id], replace=False)
        if(selection_procedure=='sparse_percentile'):
            # randomly select segments in such a way that they have sparse radii of gyration.
            # Do this by dividing the feature space in quantile regions and picking one segment per each region.
            comb_def = getRandomIntsAtPercentileRegions(selected_df.index.to_list(), selected_df['gyr'].values, num_segs_per_comb[comb_id])
        combs_definitions.append(comb_def)
    
    combs_dfs = []
    full_defs = []
    for comb_def in combs_definitions:
        if(verbose>1): print('len(comb_def): ', len(comb_def))
        full_def = []
        if(arrangement == 'lattice'):
            x_poss_count = y_poss_count = z_poss_count = int(np.ceil(np.cbrt(len(comb_def))))
            x_lattice_coords = np.linspace(0,x_poss_count-1,x_poss_count)
            y_lattice_coords = np.linspace(0,y_poss_count-1,y_poss_count)
            z_lattice_coords = np.linspace(0,z_poss_count-1,z_poss_count)
            segs_locsx,segs_locsy,segs_locsz = np.meshgrid(x_lattice_coords, y_lattice_coords, z_lattice_coords)
            segments_locations = np.stack([segs_locsz.flatten(order='C'), segs_locsx.flatten(order='C'), segs_locsy.flatten(order='C')])
            segments_locations = segments_locations[:,:len(comb_def)]
            segments_locations*= max_shift_amount
        elif(arrangement == 'tetrahedron'):
            if(len(comb_def) != 4):
                raise ValueError('Can\'t arrange in a tetrahedron a number different than 4 objects')
            segments_locations = np.array([
                [0,0,0],
                [0,1,1],
                [1,0,1],
                [1,1,0]
                ]).T*1/np.sqrt(2)*max_shift_amount
        elif(arrangement == 'random'):
            segments_locations = getPositionsGivenNumAndNNDist(len(comb_def), max_shift_amount).T
        else:
            raise ValueError('Invalid arrangement name')
        current_comb_dfs = []
        seg_count = -1
        for copy_id, seg_id in comb_def:
            seg_count+=1
            defin = {}
            defin['copy'] = copy_id
            defin['portion'] = seg_id
            full_def.append(defin)
            copy_df = chrs[copy_id]
            where_in_range = (copy_df['bp']>= seg_id*segment_length_bp) * (copy_df['bp'] < (seg_id+1)*segment_length_bp)
            where_in_range = where_in_range.astype('bool')
            selected_seg_df = copy_df.loc[where_in_range,:].copy()
            selected_seg_df['cluster-ID'] = seg_count
            current_comb_dfs.append(selected_seg_df)
        if(len(current_comb_dfs)>0):
            current_comb_joint_df = pd.concat(current_comb_dfs, axis=0).reset_index(drop=True)
            current_comb_center = current_comb_joint_df[['x','y','z']].median(axis=0).values
            lattice_size = segments_locations.max(axis=1) - segments_locations.min(axis=1)
            current_comb_origin = current_comb_center-lattice_size/2

            # Shifting clusters away from the center
            for i in np.unique(current_comb_joint_df['cluster-ID']):
                clus_center = getCoM(current_comb_joint_df.loc[current_comb_joint_df['cluster-ID'] == i,['x','y','z']].values)
                total_shift = (current_comb_origin-clus_center) + segments_locations[:,i]
                current_comb_joint_df.loc[current_comb_joint_df['cluster-ID'] == i,['x','y','z']] += total_shift
                full_def[i]['shift'] = np.array(total_shift)
        else:
            current_comb_joint_df = pd.DataFrame(columns=['x','y','z','cluster-ID'])
        combs_dfs.append(current_comb_joint_df)
        full_defs.append(full_def)

    if(return_definitions):
        return combs_dfs, full_defs
    else:
        return combs_dfs

def applyGravityStep(coords, which_fixed_coords, shift_amount=1/2, radius=100, min_dist=500):
    '''
    Get probes' positions after one step of attraction to other probes has been applied

    Arguments:
    * coords: the initial positions of the probes
    * which_fixed_coords: boolean index indicating which of coords should not move but just attract
    * shift_amount: multiplier for the amount of movement
    * radius: radius around each molecule for which the probes inside of it can affect the central one
    * min_dist: radius around each probes for which the probes inside of it cannot affect the central one. This represents the thickness of the molecule.

    Return:
    * np.array of same shape as coords, which represents the shifted coords
    '''
    # Find the molecules inside the radius and their distances from the central one
    from sklearn.neighbors import NearestNeighbors
    fixed_coords_inds = np.where(which_fixed_coords)[0]
    nn = NearestNeighbors(n_neighbors=1, n_jobs=32).fit(coords)
    dists, neighs = nn.radius_neighbors(radius=radius)

    shift_vecs = np.zeros(coords.shape)
    for i, (d,n) in enumerate(zip(dists,neighs)):
        if((not (i in fixed_coords_inds)) and (d.sum()>0)):
            # take the positions of the attracting molecules
            neigh_coords = coords[np.array(n).flatten()]
            # compute the normalized directions of attraction
            shift_dirs = (neigh_coords - coords[i])/d.reshape(-1,1)
            # magnitude of attraction is inversely proportional to distance squared
            magnitudes = -4*((min_dist/d)**12 - (min_dist/d)**6) # lennard jones
            # attraction from molecules inside the thickness is 0
            magnitudes[d<min_dist] = 0
            # shift applied to molecules is the sum of individual shifts
            # each shift is given by shift direction * shift magnitude
            shift_vecs[i] = (shift_dirs*magnitudes.reshape(-1,1)).sum(axis=0)
        else:
            # if the molecule is fixed it's shift is zero
            shift_vecs[i] = np.zeros(coords.shape[-1])
    shifted_coords = coords + shift_vecs*shift_amount
    return shifted_coords

def applyGravitySteps(coords, which_fixed_coords=None, iters_num=10, shift_amount=1/2, radius=100, verbose=False, return_iterations=False, min_dist=500):
    '''
    Get probes' positions after the attraction to other probes has been applied

    Arguments:
    * coords: the initial positions of the probes
    * which_fixed_coords: boolean index indicating which of coords should not move but just attract
    * iters_num: int, how many iterations need to be perfomed
    * shift_amount: multiplier for the amount of movement
    * radius: radius around each molecule for which the probes inside of it can affect the central one
    * min_dist: radius around each probes for which the probes inside of it cannot affect the central one. This represents the thickness of the molecule.
    * return iterations: bool, whether to return all the intermidiate positions or just the last one

    Return:
    * np.array of same shape as coords, which represents the shifted coords
    '''

    from tqdm import tqdm
    if(which_fixed_coords is None): which_fixed_coords = np.full(len(coords), False)
    shifted_coords = np.zeros((iters_num+1, coords.shape[0], coords.shape[-1]))
    # shifted_coords = coords.copy()
    shifted_coords[0] = coords.copy()
    for i in tqdm(range(1,iters_num+1), disable=not verbose):
        shifted_coords[i] = applyGravityStep(shifted_coords[i-1], which_fixed_coords, shift_amount=shift_amount, radius=radius, min_dist=min_dist)
    return shifted_coords if return_iterations else shifted_coords[-1]