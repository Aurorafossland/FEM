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
        
        # --- ta vare på K og P før enforce ---
        # --- ta vare på K og P før enforce ---
        K_original = Kmatr.copy()
        P_original = P.copy()

        Kmatr, P = enforce(Kmatr, P, bound)
        D = spsolve(Kmatr, P).reshape(-1,1)

        # === Beregn og print reaksjonskrefter ===
        R = recover_reactions(K_original, P_original, D, bound)

        strain, stress = recover(mprop, X, IX, D, ne, strain, stress)
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)

def buildload(X, IX, ne, P, loads, mprop):
    for i in range(loads.shape[0]):
        node, ldof, force = int(loads[i,0]), int(loads[i,1]), loads[i,2]
        dof = (node - 1) * 2 + (ldof - 1)
        P[dof] += force
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
        # --- FIX: ind should be [2*n1, 2*n1+1, 2*n2, 2*n2+1] ---
        n1 = int(IX[e,0])-1
        n2 = int(IX[e,1])-1
        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]  # <-- minimal fix

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

        midx = int(IX[e, 2]) - 1     # material index
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

        strain[e, 0] = eps
        stress[e, 0] = sig

    
 
    node_C = 9
    dof_vC = 2*(node_C-1) + 1   # vertikal dof
    vC = float(D[dof_vC, 0])    # i meter
    print(f"\nNode C (node {node_C}) vertical displacement: {vC:.6f} m ({vC*1e3:.2f} mm)")

    # --- NY DEL: sikkerhetssjekk ---
    yield_limit = {1: 335e6, 2: 250e6}  # Pa
    material_name = {1: "Steel (mild)", 2: "Aluminium"}
    sig = stress.reshape(-1)
    props = IX[:, 2].astype(int)

    print("\n" + "="*60)
    print("MAX STRESS PER MATERIAL (abs), compared to yield")
    print("="*60)

    overall_safe = True
    for m in [1, 2]:
        mask = (props == m)
        if not np.any(mask):
            continue
        idx_local = np.argmax(np.abs(sig[mask]))
        elem_indices = np.where(mask)[0]
        e_star = int(elem_indices[idx_local])
        sigma_star = float(sig[e_star])
        sigma_abs = abs(sigma_star)
        ys = yield_limit[m]

        n1, n2 = int(IX[e_star, 0]), int(IX[e_star, 1])
        state = "tension" if sigma_star >= 0 else "compression"

        print(f"{material_name[m]}:")
        print(f"  -> Element #{e_star+1} (nodes {n1}-{n2}) has max |σ| = {sigma_abs/1e6:.1f} MPa ({state})")
        print(f"  -> Yield limit = {ys/1e6:.1f} MPa --> {'SAFE ' if sigma_abs <= ys else 'NOT SAFE '}")

        overall_safe &= (sigma_abs <= ys)

    print("-"*60)
    print(f"STRUCTURE SAFETY: {'SAFE ' if overall_safe else 'NOT SAFE '}")
    print("-"*60)

    # --- gammel utskrift beholdes ---
    print(f'D: {D}')
    print(strain)
    print(stress)
    return strain, stress


def recover_reactions(K_original, P_original, D, bound):
    
    R = K_original.dot(D) - P_original

    print("\n=== Reaction-forces ===")
    for i in range(bound.shape[0]):
        node, ldof, disp = int(bound[i,0]), int(bound[i,1]), bound[i,2]
        dof = (node - 1)*2 + (ldof - 1)
        label = 'Fx' if ldof == 1 else 'Fy'
        print(f"Node {node} {label}: {R[dof,0]:.3f} N")
    return R


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
