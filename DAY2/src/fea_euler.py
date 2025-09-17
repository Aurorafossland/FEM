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
        Kmatr = sps.csc_matrix((neqn, neqn)) # Stiffness matrix
        P = np.zeros((neqn,1))           # Force vector
        D = np.zeros((neqn,1))           # Displacement vector
        R = np.zeros((neqn,1))           # Residual vector
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Calculate displacements
        P_final = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector
        dP = P_final/n_incr
        
        P = np.zeros((neqn, int(n_incr + 1)))
        D = np.zeros((neqn, int(n_incr + 1))) # Displacement vector in global coordinates
        
        for n in range(1, int(n_incr + 1)):

            P[:, n:n+1] = P[:, n-1:n] + dP

            Kmatr = buildstiff(X, IX, ne, mprop, neqn, D[:, n-1:n])  # Build global stiffness matrix
            Kmatr, P[:, n:n+1] = enforce(Kmatr, P[:, n:n+1], bound)          # Enforce boundary conditions

            D[:, n:n+1] = D[:, n-1:n] + spsolve(Kmatr, P[:, n:n+1]).reshape(-1, 1)   #Global displacementvector
        
        strain, stress = recover(mprop, X, IX, D[:, int(n_incr-1):int(n_incr)], ne, strain, stress)  # Calculate element stress and strain

        # Plot results
        print(f'Force Vector: {D[-2]}')
        plt.plot(D[-2], P[-2], 'b-o')
        plt.ylabel('Load P [N]')
        plt.xlabel('Displacement D [m]')
        plt.title('Load-displacement diagram')
        plt.grid(True)
        plt.show(block=True)
        PlotStructure(X, IX, ne, neqn, bound, loads, D[:, int(n_incr-1):int(n_incr)], stress)  # Plot structure


def buildload(X, IX, ne, P, loads, mprop):
    for i in range(loads.shape[0]):
        node, ldof, force = int(loads[i,0]), int(loads[i,1]), loads[i,2]
        dof = (node - 1) * 2 + (ldof - 1)
        P[dof] += force
        # print(f'ERROR in fea/buildload: build load vector')
    return P

def buildstiff(X, IX, ne, mprop, neqn, D):
    K = sps.csc_matrix((neqn, neqn))
    for e in range(ne):
        # Define element parameters
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)

        # Local to global mapping
        n1 = int(IX[e,0])-1
        n2 = int(IX[e,1])-1
        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # Assemble element stiffness matrix
        Ae = mprop[int(IX[e,2])-1,1]

        c1 =  mprop[int(IX[e,2])-1,2]
        c2 =  mprop[int(IX[e,2])-1,3]
        c3 =  mprop[int(IX[e,2])-1,4]
        c4 =  mprop[int(IX[e,2])-1,5]

        du = D[edof[2],0] - D[edof[0],0]
        dv = D[edof[3],0] - D[edof[1],0]

        eps = (dx*du + dy*dv)/L
        print(f'eps: {eps}')
        
        lam = 1 + c4 * eps

        Et = c4*(c1*(1+2*lam**(-3))+3*c2*lam**(-4)+3*c3*(-1+lam**2-2*lam**(-3)+2*lam**(-4)))
        ke = (Ae*Et/L**3) * np.array([[dx**2, dx*dy, -dx**2, -dx*dy],
                                      [dx*dy, dy**2, -dx*dy, -dy**2],
                                      [-dx**2, -dx*dy, dx**2, dx*dy],
                                      [-dx*dy, -dy**2, dx*dy, dy**2]])
        
        # Assemble into global stiffness matrix
        for i in range(4):
            for j in range(4):
                K[edof[i], edof[j]] += ke[i, j]

    np.set_printoptions(precision=3, suppress=True)
    print("Stiffness matrix K (dense):")
    print(K.toarray())
    return K

def enforce(K, P, bound):
    alpha=1e12 #infinity
    ndof_total = K.shape[0] #gets the number of degrees of freedom. shape will give a tuple with (i,j)

    K_mod = K.copy() #makes a copy of K
    P_mod = P.copy() #Makes a copy of P

   
    for i in range(bound.shape[0]): #Iterates through the BC-matrix
        node, ldof, disp = int(bound[i,0]), int(bound[i,1]), bound[i,2] #gets out the relevant values from the bound-matrix
        dof = (node - 1) * 2 + (ldof - 1) #will find the degrees of freedom on the global matrix

        
        K_mod[dof, dof] += alpha #will add infinity to the element
        #spør om man skal legge til for p-matrisen også eller kun for k-matrisen. 

    return K_mod, P_mod 

def recover(mprop, X, IX, D, ne, strain, stress):

    for e in range(ne):

        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)

        if L == 0:
            strain[e,0] = 0.0
            stress[e,0] = 0.0
            continue

        midx = int(IX[e, 2]) - 1     #index
        E    = mprop[midx, 0]       

        length_vector = np.array([-dx, -dy, dx, dy])

        n1 = int(IX[e, 0]) - 1   
        n2 = int(IX[e, 1]) - 1   

    
        de = np.array([
            D[2*n1, 0],     
            D[2*n1 + 1, 0], 
            D[2*n2, 0],    
            D[2*n2 + 1, 0]  
        ])

        B0_T = (1/L**2) * length_vector

        eps = float(B0_T @ de)   
        sig = E * eps            

        strain[e, 0] += eps
        stress[e, 0] += sig
    print(f'D: {D}')
    print(f' Strain: {strain}')
    print(f' Stress: {stress}')
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
