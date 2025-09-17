import numpy as np
import matplotlib.pyplot as plt


def plotForce(): 
    l0 = 2


    c1 = 1
    c2 = 50
    c3 = 0.1
    c4 = 100
    E = 3E+6
    A = 2
    dl = np.linspace(0, 0.20, 200)  
    eps = dl/l0
    
    lam = 1 + c4*eps
    stress = c1 * (lam - lam**(-2)) \
       + c2 * (1 - lam**(-3)) \
       + c3 * (1 - 3*lam + lam**3 - 2*lam**(-3) + 3*lam**(-2))

    force = stress * A
    
    plt.plot(dl, force)
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.grid(True)
    plt.show()
    return 


plotForce()