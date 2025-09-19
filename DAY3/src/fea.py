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
        
        disp_nr, force_nr, strain_nr, stress_nr, res_nr = newton_raphson(X, IX, ne, mprop, bound, n_incr, neqn, P_final, strain, stress, R)

        D = np.linspace(0, 0.9, 100)
        force_curve = plotForce(IX, D, mprop)

        # Plot results
        print(f'Residual Vector: {res_nr}')
        plt.plot(disp_nr[-3], force_nr[-3], 'g--o', label='Newton-Raphson', markersize=8)
        plt.plot(D, force_curve, 'k-', label='Exact', linewidth=2)
        plt.ylabel('Load P [N]')
        plt.xlabel('Displacement D [m]')
        plt.title('Load-displacement diagram')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
        PlotStructure(X, IX, ne, neqn, bound, loads, disp_nr[:, int(n_incr-1):int(n_incr)], stress)

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


def buildstiff(X, IX, ne, mprop, neqn, D):   ##La til de to siste inputparametrene for å kunne kalle på eps_g. om det ikke stemmer må vi bare definere det manuelt. 
    '''Assemble global stiffness matrix
    Assembles the global stiffness matrix from the element stiffness matrices.
    Adds up the stress stiffness, initial linear stiffness and displacement stiffness.

    returns global stiffness matrix
    '''
    
    M = np.array([
            [ 1,  0, -1,  0],
            [ 0,  1,  0, -1],
            [-1,  0,  1,  0],
            [ 0, -1,  0,  1]
        ]) #from lecture, given 

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
        d = np.array([[D[2*n1, 0]], [D[2*n1 + 1, 0]], [D[2*n2, 0]], [D[2*n2 + 1, 0]]])
        B_0 = (1/L**2) * np.array([[-dx], [-dy], [dx], [dy]])

        ###Aurora sin del, regne ut K day 3: --> Oppdaterte og ryddet litt
        ##Notat til meg selv, sjekk ut B0T og BdT osv og sjekk at T-termene er riktige :)
        B_d = ((M @ d) / L**2).reshape(-1, 1)

        eps_g = float(B_0.T @ d + 1/2 * B_d.T @ d)
        N_g = Ae * Ee * eps_g

        #Alle disse gir 4x4 matriser
        k_theta = (1/L**2) * N_g * L * M          #stress stiffness (M er 4x4)
        k_0 = Ae * Ee * L * (B_0 @ B_0.T)           #initial linear stiffness (outer product)
        k_d = Ae * Ee * L * (B_0 @ B_d.T + B_d @ B_0.T + B_d @ B_d.T)  #displacement stiffness

        K_g = k_theta + k_d + k_0

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


def recover(mprop, X, IX, D, ne):
    '''Recover residuals, strains and stresses
    Calculate strains and stresses in each element.
    
    returns residual, strain and stress vectors'''

    strain = np.zeros((ne, 1))
    stress = np.zeros((ne, 1))

    neqn = D.shape[0]
    R_int = np.zeros((neqn, 1))

    M = np.array([
        [ 1,  0, -1,  0],
        [ 0,  1,  0, -1],
        [-1,  0,  1,  0],
        [ 0, -1,  0,  1]
    ])

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
        
        A = mprop[midx, 1]  
        E = mprop[midx, 0]   

        # Local displacement vector
        d = np.array([[D[2*n1, 0]], [D[2*n1 + 1, 0]], [D[2*n2, 0]], [D[2*n2 + 1, 0]]])

        # Linear strain displacement matrix
        B_0 = (1/L**2) * np.array([[-dx], [-dy], [dx], [dy]])

        # Non-linear part of strain displacement matrix
        B_d = ((M @ d) / L**2).reshape(-1, 1)

        # Total strain displacement matrix
        Bbar_T = B_0.T + (d.T @ M) / (L**2)

        # Calculate strain and stress
        eps_G = float(B_0.T @ d + 1/2 * B_d.T @ d)
        sig_G = E * eps_G
        Ne = A * sig_G

        strain[e, 0] = eps_G
        stress[e, 0] = sig_G

        # Internal force vector
        f_int_e = (Bbar_T * Ne * L).reshape(-1,1)

        # Add up to global residual vector
        edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            R_int[edof[i], 0] += f_int_e[i, 0]
        
    return strain, stress, R_int


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


def Newton_Raphson(ne, D, K, X, IX, P, mprop, loads, n_incr, neqn, bound):
    limit = 1e-10
    max_iter = 10
    P_final = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector
    dP = P_final/n_incr
    P = np.zeros(neqn, int(n_incr + 1))
    D = np.zeros(neqn, int(n_incr + 1)) # Displacement vector in global coordinates
    R = np.zeros(neqn, int(n_incr +1))

    for n in range(1, int(n_incr + 1)):
        P[:, n:n+1] = P[:, n-1:n] + dP
        Dn = np.zeros(neqn, int(max_iter + 1))
        for i in range(max_iter+1): 
            res = calculate_residual(ne, X, IX, mprop, Dn[:, i-1:i], P[:, n:n+1])
            #må legge inn BC
            if np.linalg.norm(res) < limit: 
                break

            #calculate K with equation from lecture
            K , _ = enforce(K, P, bound)
            dD = -spsolve(K, res)
            Dn[:, i:i+1] += Dn[:, i-1:i] + dD 
        D[:, n:n+1] = Dn[:, i:i+1]

    return D, P

def newton_raphson(X, IX, ne, mprop, bound, n_incr, neqn, P_final, strain, stress, R):
    '''Newton-Raphson method
    Solves the nonlinear problem using the Newton-Raphson method.

    Returns displacement, force, strain, stress and residual vectors.
    '''    
    # Initialize arrays
    dP = P_final/n_incr
    
    P = np.zeros((neqn, int(n_incr + 1)))
    D = np.zeros((neqn, int(n_incr + 1)))

    # Parameterrs for equilibrium iteration
    limit = 1e-3
    max_iter = 70
    alpha = 1
    
    # Calculate displacements
    for n in range(1, int(n_incr + 1)):

        # Initialize load step
        P[:, n:n+1] = P[:, n-1:n] + dP
        Dn = np.zeros((neqn, int(max_iter + 1)))

        for i in range(1, int(max_iter + 1)): 

            # Calculate residual and check convergence
            _, __, R_int = recover(mprop, X, IX, Dn[:, i-1:i], ne)
            R_i = R_int - P[:, n:n+1]

            if np.linalg.norm(R_i) < limit: 
                break

            # Calculate tangent stiffness matrix and solve system
            K = buildstiff(X, IX, ne, mprop, neqn, Dn[:, i-1:i])

            K , _ = enforce(K, P, bound)
            dD = spsolve(K, R_i).reshape(-1, 1)

            Dn[:, i:i+1] = Dn[:, i-1:i] - dD * alpha
        
        # Update displacements
        D[:, n:n+1] = Dn[:, i:i+1]
    
    # Final recovery of strains and stresses
    disp = D[:, int(n_incr):int(n_incr+1)]
    strain, stress, R = recover(mprop, X, IX, disp, ne)

    return D, P, strain, stress, R


def plotForce(IX, d, mprop):
    '''Plot force-displacement curve for Von Mises truss problem
    
    returns force vector'''

    a = 0.4
    l0 = np.sqrt(1.5**2 + a**2)
    E = mprop[0, 0]
    A = mprop[0, 1]
    
    P = 2 * E * A * (a/l0)**3 * ((d/a) - 3/2 * (d/a)**2 + 1/2 * (d/a)**3)
    
    return P