try:
    import numpy as np
    from numpy.linalg import inv
except:
    raise Exception("Importing Math Packages Failed")

#recursion function calculates the first cell self-energy for a given hamiltonian, and hopping matrix.
def recursion(h,V,E):
    it = 0
    
    # Define first cell as energetically identical to the rest
    epsilon = h.copy()
    epsilon_s = h.copy()

    # A small non-zero eta is used
    eta = 0.001
    # z is defined by the identity matrix
    z = np.identity(len(h))*(E + 1j*eta)


    #Define hopping parameters
    b = V
    a = np.conj(V).transpose()
    #Define the bulk Green's function for an infinitly separated system
    g_b = inv(z - epsilon)
    
    # Recursion takes place until the hopping parameters are sufficiently small
    while (abs(a).max() > 1.e-5):
        agb = np.dot(np.dot(a,g_b),b)
        #Parameters are recalculated for the reduced system of G(2n,0) by equations (76) - (79).
        epsilon_s = epsilon_s + agb
        epsilon = epsilon + agb + np.dot(np.dot(b,g_b),a)
        a = np.dot(np.dot(a,g_b),a)
        b = np.dot(np.dot(b,g_b),b)
        # Bulk Green's function is updated
        g_b = inv(z - epsilon)
        it += 1
    
    #After recursion the self-energy is outputted
    Sigma_R = epsilon_s - h
    Sigma_L = epsilon - h - Sigma_R 
    
    
    return Sigma_L, Sigma_R