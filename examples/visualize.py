
# Intro
# 
# Visualize a simulated experiment


from CIMA.utils.Visualization import *
from CIMA.parsers.ParserCSV import *
from pipeline.utils.various import *

# Select the simulated experiment you want to visualize
# file = '/home/ipiacere@iit.local/Desktop/tmp/generated_data/sim0.csv'
file = '/home/ipiacere@iit.local/synth1/seg_len_bp_1000000/varying_num/num_80/sim0.csv'

seg = CSVParser.read_CSV_file(file, content_type='free')

plotter = plotClustering3D(seg.Getcoord(), seg.atomList['clusterID'], show_noise=True)

# Export to an html file that can be visualized in the browser
exportPyVistaPlot(plotter)