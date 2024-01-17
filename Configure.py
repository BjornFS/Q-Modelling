try:
    import numpy as np
except:
    raise Exception("Import of Numpy Module Failed")

try: 
    import ase
    from ase import io
    from ase.visualize import view
    import ase.build
    from ase.data import pubchem
    from ase.visualize.plot import plot_atoms
    from ase import Atoms
    from ase.build import attach
except:
    raise Exception("Import of ASE Module Failed")

def Atoms(X : int, Y : int, preset = None):
    """Function to create new carbon structure

    Params:
        - `preset` : name of existing file, if a pre-existing configuration exists
        - `Y` : the number of chains in the Y-direction
        - `X` : the number of `Y`'s plotted in the X-direction
    
    Returns:
        - `atoms` : sorted atoms
    """

    if preset:
        atoms =  ase.io.read(preset)
    else:
        atoms = ase.build.graphene_nanoribbon(Y, X, type='zigzag', saturated=False, vacuum=3.5)
        atoms.rotate(90,[1,0,0])
        atoms.rotate(90,[0,0,1])

    xyz = atoms.get_positions()

    # Sort the atoms based on x, then y, and lastly z
    sorted_indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))

    # Rearrange the atoms based on the sorted indices
    atoms = atoms[sorted_indices]

    xyz = atoms.get_positions()

    return atoms, xyz


