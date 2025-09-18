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
            n_incr = self.n_incr

        # Calculate problem size
        neqn = X.shape[0] * X.shape[1]   # Number of equations
        ne = IX.shape[0]                 # Number of elements
        print(f'Number of DOF {neqn} Number of elements {ne}')

        # Initialize arrays
        P = np.zeros((neqn,1))           # Force vector
        R = np.zeros((neqn,1))           # Residual vector
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Calculate displacements
        P_final = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector
        
        disp, force, strain, stress, res = euler(X, IX, ne, mprop, bound, n_incr, neqn, P_final, strain, stress, R)

        # Plot results
        print(f'Residual Vector: {res}')
        plt.plot(disp[-2], force[-2], 'b-o')
        plt.ylabel('Load P [N]')
        plt.xlabel('Displacement D [m]')
        plt.title('Load-displacement diagram')
        plt.grid(True)
        plt.show(block=True)
        PlotStructure(X, IX, ne, neqn, bound, loads, disp[:, int(n_incr-1):int(n_incr)], stress)


def buildload(X, IX, ne, P, loads, mprop):
    '''Update load vector with loads from input file
    
    returns updated load vector
    '''
    for i in range(loads.shape[0]):
        node, ldof, force = int(loads[i,0]), int(loads[i,1]), loads[i,2]
        dof = (node - 1) * 2 + (ldof - 1)
        P[dof] += force
    return P


def buildstiff(X, IX, ne, mprop, neqn, D):
    '''Assemble global stiffness matrix
    Assembles the global stiffness matrix from the element stiffness matrices.
    Uses tangential modulus for nonlinear materials.

    returns global stiffness matrix
    '''
    K = sps.csc_matrix((neqn, neqn))
    for e in range(ne):
        # Define element parameters
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)
        Ae = mprop[int(IX[e,2])-1,1]
        
        # Local to global mapping (zero-indexing in python)
        n1 = int(IX[e,0])-1
        n2 = int(IX[e,1])-1
        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # Material properties
        c1 =  mprop[int(IX[e,2])-1,2]
        c2 =  mprop[int(IX[e,2])-1,3]
        c3 =  mprop[int(IX[e,2])-1,4]
        c4 =  mprop[int(IX[e,2])-1,5]

        # Find strain and stetch of the element
        du = D[edof[2],0] - D[edof[0],0]
        dv = D[edof[3],0] - D[edof[1],0]

        eps = (dx*du + dy*dv)/L        
        lam = 1 + c4 * eps

        # Element stiffness matrix
        Et = c4*(c1*(1+2*lam**(-3))+3*c2*lam**(-4)+3*c3*(-1+lam**2-2*lam**(-3)+2*lam**(-4)))
        ke = (Ae*Et/L**3) * np.array([[dx**2, dx*dy, -dx**2, -dx*dy],
                                      [dx*dy, dy**2, -dx*dy, -dy**2],
                                      [-dx**2, -dx*dy, dx**2, dx*dy],
                                      [-dx*dy, -dy**2, dx*dy, dy**2]])
        
        # Assemble into global stiffness matrix
        for i in range(4):
            for j in range(4):
                K[edof[i], edof[j]] += ke[i, j]
    return K

def enforce(K, P, bound):
    '''Enforce boundary conditions
    Enforces boundary conditions by modifying the global stiffness matrix.
    Uses the big number method.

    returns modified stiffness matrix and load vector
    '''
    # Very high stiffness for addition to diagonals
    alpha = 1e12 * np.max(K.diagonal())

    K_mod = K.copy()
    P_mod = P.copy()

    for i in range(bound.shape[0]):
        node, ldof, disp = int(bound[i,0]), int(bound[i,1]), bound[i,2]

        # Calculate global dof
        dof = (node - 1) * 2 + (ldof - 1)

        # Check wether boundry stiffness is given
        if disp:
            K_mod[dof, dof] += disp
        else:
            K_mod[dof, dof] += alpha

    return K_mod, P_mod 


def recover(mprop, X, IX, D, ne, strain, stress, R):
    '''Recover residuals, strains and stresses
    Calculate strains and stresses in each element.
    
    returns residual, strain and stress vectors'''

    for e in range(ne):

        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)

        # Check for zero length elements
        if L == 0:
            strain[e,0] = 0.0
            stress[e,0] = 0.0
            continue
        
        # Element node numbers and material index
        n1 = int(IX[e, 0]) - 1
        n2 = int(IX[e, 1]) - 1
        midx = int(IX[e, 2]) - 1
        
        E = mprop[midx, 0]  
        A = mprop[midx, 1]     
        # Local displacement vector
        de = np.array([D[2*n1, 0], D[2*n1 + 1, 0], D[2*n2, 0], D[2*n2 + 1, 0]])

        # Linear strain displacement matrix
        B_0 = (1/L**2) * np.array([[-dx], [-dy], [dx], [dy]])

        # Non-linear strain and stress
        eps = np.matmul(np.transpose(B_0), de)

        c1 =  mprop[int(IX[e,2])-1,2]
        c2 =  mprop[int(IX[e,2])-1,3]
        c3 =  mprop[int(IX[e,2])-1,4]
        c4 =  mprop[int(IX[e,2])-1,5]

        lam = 1 + c4*eps
        sig = c1 * (lam - lam**(-2)) \
            + c2 * (1 - lam**(-3)) \
            + c3 * (1 - 3*lam + lam**3 - 2*lam**(-3) + 3*lam**(-2))
        
        strain[e, 0] += eps
        stress[e, 0] += sig
        
        # Residuals
        R_int = B_0 * sig * A * L

        R[2*n1:2*n1+2, 0] += R_int[0:2, 0]
        R[2*n2:2*n2+2, 0] += R_int[2:4, 0]
        
    return strain, stress, R

def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
        '''Plot undeformed and deformed structure
        
        returns none
        '''
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

            # --- Color depending on stress ---
            if np.abs(stress[e, 0]) < 1e-6:
                color = 'g'
            elif stress[e, 0] > 0:
                color = 'b'  # Tension: blue
            else:
                color = 'r'  # Compression: red

            plt.plot(xx_def, yy_def, color, linewidth=lw)

        plt.legend(["Undeformed", "Deformed"], loc="upper right")

        # Plot supports and loads 
        Xnew, dsup = plotsupports(X, D, neqn, bound)
        plotloads(loads, Xnew, dsup)
        plt.axis('equal')
        plt.show(block=True)


def euler(X, IX, ne, mprop, bound, n_incr, neqn, P_final, strain, stress, R):
    
    # Initialize arrays
    dP = P_final/n_incr
    
    P = np.zeros((neqn, int(n_incr + 1)))
    D = np.zeros((neqn, int(n_incr + 1)))
    
    # Calculate displacements
    for n in range(1, int(n_incr + 1)):

        P[:, n:n+1] = P[:, n-1:n] + dP

        Kmatr = buildstiff(X, IX, ne, mprop, neqn, D[:, n-1:n])
        Kmatr, P[:, n:n+1] = enforce(Kmatr, P[:, n:n+1], bound)

        D[:, n:n+1] = D[:, n-1:n] + spsolve(Kmatr, P[:, n:n+1]).reshape(-1, 1)
    
    disp = D[:, int(n_incr-1):int(n_incr)]
    strain, stress, R = recover(mprop, X, IX, disp, ne, strain, stress, R)

    return D, P, strain, stress, R