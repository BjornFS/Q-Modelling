try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except:
    raise Exception("Import of Numpy Module Failed")


def BandStructure2D(Hamiltonian, HopX, HopY):
    """
    """
    size = len(Hamiltonian)
    # make grid of 100 equi-distant points
    resolution = 100
    k_x_values = np.linspace(-np.pi, np.pi, resolution)
    k_y_values = np.linspace(-np.pi, np.pi, resolution)
    kx, ky = np.meshgrid(k_x_values, k_y_values)
    k_values = np.vstack((kx.ravel(), ky.ravel())).T

    eigenvals = np.zeros((resolution, resolution, size)) 

    # iterate over mesh to calculate H_k as function of k point in both x and y direction
    for i, (kx_val, ky_val) in enumerate(k_values):
        # Construct the Hamiltonian matrix H_k
        H_k = Hamiltonian + HopX * np.exp(-1j * kx_val) + np.conj(HopX.T) * np.exp(1j * kx_val) \
              + HopY * np.exp(-1j * ky_val) + np.conj(HopY.T) * np.exp(1j * ky_val)
        # Calculate the eigenvalues
        e_vals, e_vecs = np.linalg.eigh(H_k)

        # Store the eigenvalues
        ix,iy = np.unravel_index(i, (resolution, resolution))
        eigenvals[ix, iy, :] = e_vals

    # You can now plot the band structure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for band in range(size):
        # Reshape the eigenvalues for band 'band' to a 2D array
        data = eigenvals[:, :, band].reshape((resolution, resolution))
        # Plot the surface
        ax.plot_surface(kx, ky, data, cmap='viridis')

    # Set labels
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Energy')

    # Show the plot
    plt.show()

def BandStructure1D(Hamiltonian, Hopping):
    """
    """
    size = len(Hamiltonian)
    # make grid of 100 equi-distant points
    resolution = 100

    k_values = np.linspace(-np.pi, np.pi, resolution) # Example range, adjust as needed
    
    # Prepare to store the eigenvalues
    eigenvalues = np.zeros((len(k_values), size)) # Assuming 4 bands

    # Solve the eigenvalue problem for each k
    for i, k in enumerate(k_values):
        # Construct the matrix for this value of k
        matrix = Hamiltonian + Hopping * np.exp(-1j * k) + np.conj(Hopping).T * np.exp(1j * k)

        # Calculate eigenvalues
        vals = np.linalg.eigvalsh(matrix)
        
        # Store the first 4 eigenvalues
        eigenvalues[i, :] = vals[:size]

    # Plot the band structure
    for b in range(size):
        plt.plot(k_values, eigenvalues[:, b], label=f'Band {b+1}')

    plt.xlabel('k')
    plt.ylabel(r'$\epsilon_b(k)$')
    plt.title('Band Structure')
    plt.legend()
    plt.show()