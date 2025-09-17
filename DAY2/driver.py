# Python driver for 41525 FEM basic Matlab code
import os

# Changing to the directory of this file
os.chdir(os.path.dirname(__file__))
print ("Current working dir : %s" % os.getcwd())


# import FEA code module
from src.fea_euler import Fea

# Define input file (the standard FEM course Matlab input format)
input_file = 'TrussExercise2_2025.m'
# Perform FEA
fea = Fea(input_file)