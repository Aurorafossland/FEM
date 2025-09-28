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
            V_cond = self.V_cond; max_iopt = int(self.max_iopt)


        # Calculate problem size
        neqn = X.shape[0] * X.shape[1]   # Number of equations
        ne = IX.shape[0]                 # Number of elements
        print(f'Number of DOF {neqn} Number of elements {ne}')

        # Initialize arrays
        P = np.zeros((neqn,1))           # Force vector
        R = np.zeros((neqn,1))           # Residual vector
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Initialize parameters
        p = 2
        eta = 0.5

        # Calculate displacements
        P_final = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector

        rho, D, f = topology(X, IX, ne, neqn, P, loads, bound, mprop, V_cond, p, max_iopt, eta)
        
        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, rho)

def buildload(X, IX, ne, P, loads, mprop):
    '''Update load vector with loads from input file
    
    returns updated load vector
    '''
    for i in range(loads.shape[0]):
        node, ldof, force = int(loads[i,0]), int(loads[i,1]), loads[i,2]
        dof = (node - 1) * 2 + (ldof - 1)
        P[dof] += force
    return P


def calculate_residual(ne, X, IX, mprop, D, P):
    Bbars = []  #saves Bbar
    ndof = D.shape[0] 
    R_int = np.zeros((ndof, 1)) #Sets up a global inner force-vector. 

    #So far we have the first part in the resuidal-term before further summation
    for e in range(ne):
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)
        #same as prev code 

        midx = int(IX[e, 2]) - 1 #this gets out the materialnumber for element e, and we need to subtract 1 to index correctly in python
        E    = mprop[midx, 0]
        A = mprop[midx, 1]
         #reads materialdata from mprop 

        length_vector = np.array([-dx, -dy, dx, dy])
        B0 = (1/L**2) * length_vector   
        #same av prev code

        n1 = int(IX[e, 0]) - 1
        n2 = int(IX[e, 1]) - 1

        d = np.array([
            D[2*n1, 0],
            D[2*n1 + 1, 0],
            D[2*n2, 0],
            D[2*n2 + 1, 0]
        ])
        #this is the same as prev code

        M = np.array([
            [ 1,  0, -1,  0],
            [ 0,  1,  0, -1],
            [-1,  0,  1,  0],
            [ 0, -1,  0,  1]
        ]) #from lecture, given 

        quad_term = (d.T @ (M @ d)) / (2 * L**2) #term to calculate the eps_g

        Bbar_T = B0.T + (d.T @ M) / (L**2) #calculate Bbar at the given element
        eps_G = float(B0.T @ d + quad_term)    #calculate the eps_G by the term given in lecture day 3
        N = A * E * eps_G        #axial force with hookes law              

        f_int_e = (Bbar_T * N * L).reshape(-1,1) #this is the first part of the redisual-term for the goven elemnt

        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1] #dof
        for i in range(4): #this will be the summation-part of the redisual-term. the local matrix is the size of four
            R_int[edof[i], 0] += f_int_e[i, 0]

    residual = R_int - P #this is the final redisual-vector. this will have the size dofx1

    print(f'redisual: {residual}')
    return residual


def buildstiff(X, IX, ne, mprop, neqn, rho, p):
    '''Assemble global stiffness matrix
    Assembles the global stiffness matrix from the element stiffness matrices.
    Adds up the stress stiffness, initial linear stiffness and displacement stiffness.

    returns global stiffness matrix
    '''
    K = sps.lil_matrix((neqn, neqn))

    for e in range(ne):
        # Define element parameters
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)
        Ae = mprop[int(IX[e,2])-1,1]
        Ee = mprop[int(IX[e,2])-1,0]
        
        # Local to global mapping (zero-indexing in python)
        n1 = int(IX[e,0])-1
        n2 = int(IX[e,1])-1
        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # Local displacement vector and linear strain displacement matrix
        B_0 = (1/L**2) * np.array([[-dx], [-dy], [dx], [dy]])

        #Alle disse gir 4x4 matriser
        k_0 = Ae * Ee * L * (B_0 @ B_0.T)
        K_g = k_0 * rho[e]**p  #Initial linear stiffness

        #Assemble into global stiffness matrix
        for i in range(4):
            for j in range(4):
                K[edof[i], edof[j]] += K_g[i, j]


        ###Aurora del slutt:) Har nå endret slik at tidligere K blir overkjørt.
    return K


def enforce(K, P, bound):
    '''Enforce boundary conditions
    Enforces boundary conditions by modifying the global stiffness matrix.
    Uses the big number method.

    returns modified stiffness matrix and load vector
    '''
    # Very high stiffness for addition to diagonals
    alpha = 1e9 * np.max(K.diagonal())

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

    return K_mod.tocsc(), P_mod 


def recover(mprop, X, IX, D, ne, p, rho):
    '''Recover residuals, strains and stresses
    Calculate strains and stresses in each element.
    
    returns residual, strain and stress vectors'''


    df = np.zeros((ne, 1))
    dg = np.zeros((ne, 1))
    Vf = np.zeros((ne, 1)) # f/rho**p

    neqn = D.shape[0]
    R_out = np.zeros((neqn, 1))

    for e in range(ne):

        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)
        
        # Element node numbers and material index
        n1 = int(IX[e, 0]) - 1
        n2 = int(IX[e, 1]) - 1
        midx = int(IX[e, 2]) - 1
        
        Ae = mprop[midx, 1]
        Ee = mprop[midx, 0]     
        # Local displacement vector
        de = np.array([[D[2*n1, 0]], [D[2*n1 + 1, 0]], [D[2*n2, 0]], [D[2*n2 + 1, 0]]])

        # Linear strain displacement matrix
        B_0 = (1/L**2) * np.array([[-dx], [-dy], [dx], [dy]])

        # Linear stiffness matrix
        k_0 = Ae * Ee * L * (B_0 @ B_0.T)


        # Sensitivity objective function
        df[e] = -p * (rho[e]**(p-1)) * de.T @ k_0 @ de
        Vf[e] = de.T @ k_0 @ de

        # Sensitivity constraint function
        dg[e] = rho[e] * L * Ae
        
    return df.reshape(-1), dg.reshape(-1), Vf.reshape(-1)

def find_displacement(Kmatr, P, node_num):

    D = spsolve(Kmatr, P).reshape(-1, 1)
    

    dof_x = 2 * (node_num - 1)
    dof_y = dof_x + 1
    
    ux = float(D[dof_x, 0])  #horisontal
    uy = float(D[dof_y, 0])  #vertical

    print(f"Node {node_num}: ux = {ux:.6e} m, uy = {uy:.6e} m")
    return D, ux, uy


def securety_check(stress, IX):
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

    return 


def recover_reactions(K_original, P_original, D, bound):
    
    R = K_original.dot(D) - P_original

    print("\n=== Reaction-forces ===")
    for i in range(bound.shape[0]):
        node, ldof, disp = int(bound[i,0]), int(bound[i,1]), bound[i,2]
        dof = (node - 1)*2 + (ldof - 1)
        label = 'Fx' if ldof == 1 else 'Fy'
        print(f"Node {node} {label}: {R[dof,0]:.3f} N")
    return R


def PlotStructure(X, IX, ne, neqn, bound, loads, D, rho):
    # Plot the deformed and undeformed structure

    plt.figure(1)
    lw_min = 0.5   # Minimum linewidth
    lw_max = 10.0   # Maximum linewidth
    scale = 1.0    # Displacement scaling

    # Normalize rho for linewidth scaling
    rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho) + 1e-12)
    linewidths = lw_min + (lw_max - lw_min) * rho_norm

    for e in range(ne):
        if rho[e] > 0.1 * np.max(rho):  # Only plot elements with significant density
            xx = X[IX[e, 0:2].astype(int)-1, 0]
            yy = X[IX[e, 0:2].astype(int)-1, 1]
            # Plot undeformed solution
            plt.plot(xx, yy, 'k:', linewidth=0.5)
            # Plot with linewidth based on rho
            plt.plot(xx, yy, color='b', linewidth=linewidths[e])

    plt.legend(["Undeformed", "Density-weighted"], loc="upper right")

    # Plot supports and loads 
    Xnew, dsup = plotsupports(X, np.zeros_like(D), neqn, bound)
    plotloads(loads, Xnew, dsup)
    plt.axis('equal')
    plt.show(block=True)


def topology(X, IX, ne, neqn, P, loads, bound, mprop, V_cond, p, max_iopt, eta=1.0):

    V_tot = 0
    P = buildload(X, IX, ne, P, loads, mprop)

    # Calculate total volume

    V = np.zeros(len(IX))

    for e in range(IX.shape[0]):
        dx = X[int(IX[e,1])-1,0] - X[int(IX[e,0])-1,0]
        dy = X[int(IX[e,1])-1,1] - X[int(IX[e,0])-1,1]
        L = np.sqrt(dx**2 + dy**2)
        Ae = mprop[int(IX[e,2])-1,1]
        Ve = Ae * L
        V[e] = Ve
    
    V_tot = np.sum(V)
    rho = np.empty(IX.shape[0])
    rho.fill(V_cond / V_tot)

    f = np.zeros((max_iopt, 1))

    for iopt in range(1, int(max_iopt + 1)):
        rho_old = rho.copy()

        K = buildstiff(X, IX, ne, mprop, neqn, rho, p)
        K , _ = enforce(K, P, bound)
        D = spsolve(K, P).reshape(-1, 1)

        df, dg, Vf = recover(mprop, X, IX, D, ne, p, rho)

        def rho_func(lam):
            B = -df/(lam*dg)
            return np.clip(rho_old * B**eta, 1e-6, 1.0)

        def volume_constraint(lam):
            return np.sum(rho_func(lam) * V) - V_cond

        lam = bisection(volume_constraint, 1e-10, 1e10)

        rho = rho_func(lam).reshape(-1)

        if np.linalg.norm(rho_old - rho) < 1e-10 * np.linalg.norm(rho):
            print(f"Converged after {iopt} iterations.")
            break
        
        f[iopt-1] = Vf.T @ rho**p

        print(iopt)
        
        # plot structure?

    return rho, D, f

def bisection(func, lam1, lam2, tol=1e-5, max_iter=100):
    '''Bisection method for finding root of a function'''
    f1 = func(lam1)
    f2 = func(lam2)

    if f1 * f2 > 0:
        raise ValueError("Function has same sign at the interval endpoints.")

    for i in range(max_iter):
        lam_mid = (lam1 + lam2) / 2
        f_mid = func(lam_mid)

        if (lam2 - lam1)/(lam2 + lam1) < tol:
            return lam_mid

        if f1 * f_mid < 0:
            lam2 = lam_mid
            f2 = f_mid
        else:
            lam1 = lam_mid
            f1 = f_mid

    raise ValueError("Maximum iterations reached without convergence.")
        


def plotForce(IX, d, mprop):
    '''Plot force-displacement curve for Von Mises truss problem
    
    returns force vector'''

    a = 0.4
    l0 = np.sqrt(1.5**2 + a**2)
    E = mprop[0, 0]
    A = mprop[0, 1]
    
    P = 2 * E * A * (a/l0)**3 * ((d/a) - 3/2 * (d/a)**2 + 1/2 * (d/a)**3)
    
    return P