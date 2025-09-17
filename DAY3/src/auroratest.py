import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
import math
import re
from scipy.sparse.linalg import spsolve


#Dette er kun en test for å sjekke at koden virker for BCs

# Hardkodet 12x12 stivhetsmatrise (symmetrisk, positiv-definit)
K = np.array([
    [20, -5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [-5, 15, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -3, 18, -4,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, -4, 16, -2,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, -2, 14, -5,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, -5, 17, -3,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0, -3, 19, -4,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0, -4, 16, -2,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0, -2, 15, -5,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, -5, 18, -3,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -3, 16, -4],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -4, 20]
], dtype=float)

# Hardkodet lastvektor (12x1)
P = np.array([10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)



bound = np.array([
    [1, 1, 0.0],
    [1, 2, 0.0],
    [4, 1, 0.0]
], dtype=float)



def enforce(K, P, bound):
    alpha=1e12 #infinity
    ndof_total = K.shape[0] #gets the number of degrees of freedom. shape will give a tuple with (i,j)

    K_mod = K.copy() #makes a copy of K
    P_mod = P.copy() #Makes a copy of P

   
    for i in range(bound.shape[0]): #Iterates through the BC-matrix
        node, ldof, disp = int(bound[i,0]), int(bound[i,1]), bound[i,2] #gets out the relevant values from the bound-matrix
        dof = (node - 1) * 2 + (ldof - 1) #will find the degrees of freedom on the global matrix

        
        K_mod[dof, dof] += alpha #will add infinity to the element
        P_mod[dof] += alpha * disp #will add infinity to the element 

        #spør om man skal legge til for p-matrisen også eller kun for k-matrisen. 

    return K_mod, P_mod 



print(enforce(K, P, bound))