
# Intro
# 
# Visualize a simulated experiment


from CIMA.utils.Visualization import *
from CIMA.parsers.ParserCSV import *
from pipeline.utils.various import *

file = '/home/ipiacere@iit.local/Desktop/tmp/generated_data/sim0.csv'

seg = CSVParser.read_CSV_file(file, content_type='free')

plotter = plotClustering3D(seg.Getcoord())

exportPyVistaPlot(plotter)