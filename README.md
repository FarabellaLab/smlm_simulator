# SMLM Simulator

A simulator for Single Molelecule Localization Microscopy data.

In examples/ you can find examples on how to use the simulator:
- generate_and_visualize: A notebook whene you simply generate a simulated experiment and visualize it
- segments_arrangements: A notebook where you generate different simulated experiments, arranging the segmnets in each in different configurations
- generate_scenarios: A python script that generates a dataset containing the 4 scenarios we use for assessing clustering algorithms
- visualize: A python script to visualize a simulated experiment

In src/ you can find the source code:
- simulation: contains the core functions involved in the simulation
- various: contains various utility functions

## Working scheme
Function simulate() generates a simulated experiment starting from predefined segment paths (which can be obtained e.g. with getChr21DLs) and following the specified parameters.

## Credits:
The disposition of probe localizations and false localizaions is based on https://github.com/JeremyPike/RSMLM (Pike, J. A., Khan, A. O., Pallini, C., Thomas, S. G., Mund, M., Ries, J., ... & Styles, I. B. (2020). Topological data analysis quantifies biological nano-structure from single molecule localization microscopy. Bioinformatics, 36(5), 1614-1621.)