import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
import math
import re
from scipy.sparse.linalg import spsolve

from src.plotsupports    import plotsupports
from src.plotloads       import plotloads

class Fea:
    def __init__(self, input_file):       
        # Read input in the standard Matlab format and convert to python variables
        with open(input_file, 'r') as file:
            inp = file.read()

            matches = re.findall(r'(?:\n|^)\s*(\w+)\s*=\s*\[\s*([\s\S]*?)\s*\]', inp, re.DOTALL)
            for name,content in matches:
                vals = content.strip().split('\n')
                vals = np.array([[float(val) for val in row.strip().split()] for row in vals])
                setattr(self, name, vals)     # this is for storing a variable in a class

            matches = re.findall(r'(?:\n|^)\s*(\w+)\s*=\s*(\d+)\s*(?:;|\n)', inp, re.DOTALL)
            for name,content in matches:
                setattr(self, name, float(content.strip()))

            for requiredvar in ("X", "IX", "mprop", "bound", "loads", "plotdof"):
                assert hasattr(self, requiredvar), \
                       f"Error in input file. Could not read variable {requiredvar}."
            X = self.X; IX = self.IX; mprop = self.mprop
            bound = self.bound; loads = self.loads; plotdof = self.plotdof


        # Calculate problem size
        neqn = X.shape[0] * X.shape[1]   # Number of equations
        ne = IX.shape[0]                 # Number of elements
        print(f'Number of DOF {neqn} Number of elements {ne}')

        # Initialize arrays
        Kmatr = sps.csc_matrix((neqn, neqn)) # Stiffness matrix
        P = np.zeros((neqn,1))           # Force vector
        D = np.zeros((neqn,1))           # Displacement vector
        R = np.zeros((neqn,1))           # Residual vector
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Calculate displacements
        P = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector

        Kmatr = buildstiff(X, IX, ne, mprop, Kmatr)  # Build global stiffness matrix
        
        Kmatr, P = enforce(Kmatr, P, bound)          # Enforce boundary conditions
        
        strain, stress = recover(mprop, X, IX, D, ne, strain, stress)  # Calculate element stress and strain

        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure


def buildload(X, IX, ne, P, loads, mprop):
    for i in range(loads.shape[0]):
        print(f'ERROR in fea/buildload: build load vector')
    return P

def buildstiff(X, IX, ne, mprop, K):
    for e in range(ne):
        # Define element parameters
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)

        # Assemble element stiffness matrix
        Ae = mprop[int(IX[e,2])-1,1]
        Ee = mprop[int(IX[e,2])-1,0]
        ke = (Ae*Ee/L**3) * np.array([[dx**2, dx*dy, -dx**2, -dx*dy],
                                      [dx*dy, dy**2, -dx*dy, -dy**2],
                                      [-dx**2, -dx*dy, dx**2, dx*dy],
                                      [-dx*dy, -dy**2, dx*dy, dy**2]])
        
        # Adding into global stiffeness matrix
        ind = [int(IX[e,0]*2)-2, int(IX[e,1]*2)-2] # Index of each sub-atrix
        for ig, il in enumerate(range(0,2,2)):
            for jg, jl in enumerate(range(0,2,2)):
                K[ind[ig],ind[jg]] += ke[il,jl]
                K[ind[ig]+1,ind[jg]] += ke[il+1,jl]
                K[ind[ig],ind+1[jg]] += ke[il,jl+1]
                K[ind[ig]+1,ind+1[jg]] += ke[il+1,jl+1]
    return K

def enforce(K, P, bound):
    for i in range(bound.shape[0]):
        print(f'ERROR in fea/enforce: enforce boundary conditions')
    return K, P

def recover(mprop, X, IX, D, ne, strain, stress):
    for e in range(ne):
        print(f'ERROR in fea/recover: calculate strain and stress')
    return strain, stress

def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
        # Plot the deformed and undeformed structure

        # Plot settings
        plt.figure(1)
        lw = 3.5        # Linewidth for plotting bars
        scale = 1.0     # Displacement scaling

        for e in range(ne):
            xx = X[IX[e, 0:2].astype(int)-1, 0]
            yy = X[IX[e, 0:2].astype(int)-1, 1]
            # Plot undeformed solution
            plt.plot(xx, yy, 'k:', linewidth=1)
            # Get displacements in x and y
            n1, n2 = IX[e, 0:2].astype(int)
            edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1])
            xx_def = xx + scale*D[edof[0:4:2]-2,0]
            yy_def = yy + scale*D[edof[1:4:2]-2,0]
            plt.plot(xx_def, yy_def, 'b', linewidth=lw)

        plt.legend(["Undeformed", "Deformed"], loc="upper right")

        # Plot supports and loads 
        Xnew, dsup = plotsupports(X, D, neqn, bound)
        plotloads(loads, Xnew, dsup)
        plt.axis('equal')
        plt.show(block=True)