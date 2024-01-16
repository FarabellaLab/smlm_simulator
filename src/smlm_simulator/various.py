import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from CIMA.detection import clusters
from CIMA.segments.SegmentGaussian import TransformBlurrer
from CIMA.maps import MapFeatures
from CIMA.maps import DensityProprieties
from CIMA.segments.SegmentInfoXYZ import SegmentXYZ
import seaborn as sns
import pyvista as pv

class TimeLabeledPrintOutNewline:
    
    def __init__(self):
        import sys
        self.old_out = sys.stdout
        self.nl = True
        
  
    def write(self, x):
        from datetime import datetime as dt 
        if x == '\n':
            self.old_out.write(x)
            self.nl = True
        elif self.nl:
            self.old_out.write('%s> %s' % (str(dt.now()), x))
            self.nl = False
        else:
            self.old_out.write(x)

    def flush(var):
        pass

class TimeLabeledPrintErrNewline:
    
    def __init__(self):
        import sys
        self.old_out = sys.stderr
        self.nl = True
        
  
    def write(self, x):
        from datetime import datetime as dt 
        if x == '\n':
            self.old_out.write(x)
            self.nl = True
        elif self.nl:
            self.old_out.write('%s> %s' % (str(dt.now()), x))
            self.nl = False
        else:
            self.old_out.write(x)

    def flush(var):
        pass

class TimeLabeledPrintOut:
    
    def __init__(self):
        import sys
        self.old_out = sys.stdout
  
    def write(self, x):
        from datetime import datetime as dt
        self.old_out.write('%s> %s' % (str(dt.now()), x))

    def flush(var):
        pass


class TimeLabeledPrintErr:
    
    def __init__(self):
        import sys
        self.old_err = sys.stderr
        
  
    def write(self, x):
        from datetime import datetime as dt
        self.old_err.write('%s> %s' % (str(dt.now()), x))

    def flush(var):
        pass

def overwriteOuts(newline=True):
    import sys
    if(newline):
        sys.stdout = TimeLabeledPrintOutNewline()
        sys.stderr = TimeLabeledPrintErrNewline()
    else:
        sys.stdout = TimeLabeledPrintOut()
        sys.stderr = TimeLabeledPrintErr()



def getClusterFeatures(cluster_coords, all_coords=None, selected_features=['volume','gyr','num']):
    assert len(selected_features) > 0
    d = {}
    if('num' in selected_features): d['num'] = len(cluster_coords)
    if('gyr' in selected_features): d['gyr'] = getGyration(cluster_coords)
    if('diameter' in selected_features): d['diameter'] = getDiameter(cluster_coords)
    if('volume' in selected_features): 
        timepoint = np.zeros((len(cluster_coords),1))
        df = pd.DataFrame(np.concatenate((cluster_coords, timepoint), axis=1), columns=['x','y','z','timepoint'])
        segment = SegmentXYZ(df)
        TB = TransformBlurrer()
        map = TB.SR_gaussian_blur(segment, 50., sigma_coeff=1.)
        d['volume'] = MapFeatures.GetVolume_abovecontour(map, factor=0.0)
    if(('num' in selected_features) and ('volume' in selected_features)):
        d['core_density'] = d['num']/d['volume']
    if(('num' in selected_features) and ('diameter' in selected_features)):
        d['overall_density'] = d['num']/radiusToArea(d['diameter']/2)
    if(('full_num' in selected_features) and ('diameter' in selected_features) and (not all_coords is None)):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors().fit(all_coords)
        center = np.median(cluster_coords, axis=0)
        neighs = nn.radius_neighbors(np.array([center]), radius = d['diameter']/2, return_distance=False)[0]
        d['full_num'] = len(neighs)

    return d

def getCoM(points):
    return points.mean(axis=0)

def getDiameter(points):
    from scipy.spatial.distance import pdist
    dists = pdist(points, 'euclidean')
    return dists.max()

def getGyration(points):
    com = getCoM(points)
    dists = np.linalg.norm(points-com,axis=1)
    return np.sqrt((dists**2).mean())

def getBoxCenter(points):
    return (points.max(axis=0)+points.min(axis=0))/2