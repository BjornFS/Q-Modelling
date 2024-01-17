try:
    import numpy as np
except:
    raise Exception("Import of Numpy Module Failed")

global a, cut, vpppi
a = 1.46 # C-C distance
cut = a + 0.3
Vpppi = -2.7 # eV


def hamiltonian(xyz):
    #### Tight binding hamiltonian for a set of atomic coordinates in units of Vpppi
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    return np.where((dist < (a + 0.15)) & (dist > 0.1), Vpppi, 0)   

def hamiltonian2(xyz1, xyz2):
    #### Tight binding hamiltonian for a set of atomic coordinates in units of Vpppi
    dist = np.linalg.norm(xyz1[None, :, :] - xyz2[:, None, :], axis=2)
    return np.where((dist < (a + 0.15)) & (dist > 0.1), Vpppi, 0)   


def hamdd(xyz):
    #### Distance-dependent tight binding hamiltonian for a set of atomic coordinates in units of Vpppi
    N = len(xyz)
    hamdd = np.zeros([N,N])
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    for i in np.arange(N):
        for j in np.arange(N):
            if (i != j) & (dist[i,j] < cut):
                hamdd[i,j] = Vpppi*(a/dist[i,j])**2  # if using distance dependence!
    return hamdd


def SplitHam(H,nL,nR):
        no = len(H)
        nd = no - nL - nR
        if(nd < 1):
            print("Setup error: number of device sites = ", nd)
            print("Use [nL|nL|nd|nR|nR] setup")
            return
    
        hL = H[nL:2*nL,nL:2*nL]
        VL = H[0:nL,nL:2*nL] # left-to-right hop
        hD = H[nL:nL+nd,nL:nL+nd]
        hR = H[-nR:,-nR:]
        VR = H[-2*nR:-nR,-nR:] # Left-to-right hop
        return hL,VL,hR,VR,hD

