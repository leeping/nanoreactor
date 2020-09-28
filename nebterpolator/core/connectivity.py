"""
Routines to etermine the bond/angle/dihedral connectivity in a molecular graph.
"""

##############################################################################
# Imports
##############################################################################

import numpy as np
from scipy.spatial.distance import squareform, pdist
import networkx as nx
from itertools import combinations

##############################################################################
# GLOBALS
##############################################################################

# these are covalent radii taken from the crystalographic data in nm
# Dalton Trans., 2008, 2832-2838, DOI: 10.1039/B801115J
# http://pubs.rsc.org/en/Content/ArticleLanding/2008/DT/b801115j
# COVALENT_RADII = {'C': 0.0762, 'N': 0.0706, 'O': 0.0661, 'H': 0.031,
#                   'S': 0.105}

COVALENT_RADII = {'H': 0.031, 'He': 0.028, 
                  'Li': 0.128, 'Be': 0.096, 'B': 0.084, 'C': 0.076, 'N': 0.071, 'O': 0.066, 'F': 0.057, 'Ne': 0.058, 
                  'Na': 0.166, 'Mg': 0.141, 'Al': 0.121, 'Si': 0.111, 'P': 0.107, 'S': 0.105, 'Cl': 0.102, 'Ar': 0.106, 
                  'K': 0.203, 'Ca': 0.176, 'Sc': 0.170, 'Ti': 0.160, 'V': 0.153, 'Cr': 0.139, 'Mn': 0.161, 'Fe': 0.152, 'Co': 0.150, 
                  'Ni': 0.124, 'Cu': 0.132, 'Zn': 0.122, 'Ga': 0.122, 'Ge': 0.120, 'As': 0.119, 'Se': 0.120, 'Br': 0.120, 'Kr': 0.116, 
                  'Rb': 0.220, 'Sr': 0.195, 'Y': 0.190, 'Zr': 0.175, 'Nb': 0.164, 'Mo': 0.154, 'Tc': 0.147, 'Ru': 0.146, 'Rh': 0.142, 
                  'Pd': 0.139, 'Ag': 0.145, 'Cd': 0.144, 'In': 0.142, 'Sn': 0.139, 'Sb': 0.139, 'Te': 0.138, 'I': 0.139, 'Xe': 0.140, 
                  'Cs': 0.244, 'Ba': 0.215, 'La': 0.207, 'Ce': 0.204, 'Pr': 0.203, 'Nd': 0.201, 'Pm': 0.199, 'Sm': 0.198, 
                  'Eu': 0.198, 'Gd': 0.196, 'Tb': 0.194, 'Dy': 0.192, 'Ho': 0.192, 'Er': 0.189, 'Tm': 0.190, 'Yb': 0.187, 
                  'Lu': 0.187, 'Hf': 0.175, 'Ta': 0.170, 'W': 0.162, 'Re': 0.151, 'Os': 0.144, 'Ir': 0.141, 'Pt': 0.136, 
                  'Au': 0.136, 'Hg': 0.132, 'Tl': 0.145, 'Pb': 0.146, 'Bi': 0.148, 'Po': 0.140, 'At': 0.150, 'Rn': 0.150, 
                  'Fr': 0.260, 'Ra': 0.221, 'Ac': 0.215, 'Th': 0.206, 'Pa': 0.200, 'U': 0.196, 'Np': 0.190, 'Pu': 0.187, 
                  'Am': 0.180, 'Cm': 0.169}

__all__ = ['bond_connectivity', 'angle_connectivity', 'dihedral_connectivity']

##############################################################################
# Functions
##############################################################################


def bond_connectivity(xyz, atom_names, enhance=1.3):
    """Get a list of all the bonds in a conformation

    Regular bonds are assigned to all pairs of atoms where
    the interatomic distance is less than or equal to 'enhance' times the
    sum of their respective covalent radii.

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation. The coodinates
        are expected to be in units of nanometers.
    atom_names : array_like of strings, length=n_atoms
        A list of the names of each of the atoms, which will be used for
        grabbing the covalent radii.

    Returns
    -------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    xyz = np.asarray(xyz)
    if not xyz.ndim == 2:
        raise TypeError('xyz has ndim=%d. Should be 2' % xyz.ndim)

    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    if len(atom_names) != n_atoms:
        raise ValueError(('atom_names must have the same number of atoms '
                          'as xyz'))

    # TODO: This logic only works for elements that are a single letter
    # If we need to deal with other elements, we can easily generalize it.
    proper_atom_names = np.zeros(n_atoms, dtype=str)
    for i in range(n_atoms):
        # name of the element that is atom[i]
        # take the first character of the AtomNames string,
        # after stripping off any digits
        proper_atom_names[i] = atom_names[i].strip('123456789 ')[:2]
        if not proper_atom_names[i] in list(COVALENT_RADII.keys()):
            raise ValueError("I don't know about this atom_name: %s" %
                             atom_names[i])

    distance_mtx = squareform(pdist(xyz))
    connectivity = []

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Regular bonds are assigned to all pairs of atoms where
            # the interatomic distance is less than or equal to 1.3 times the
            # sum of their respective covalent radii.
            d = distance_mtx[i, j]
            if d < enhance * (COVALENT_RADII[proper_atom_names[i]] +
                              COVALENT_RADII[proper_atom_names[j]]):
                connectivity.append((i, j))

    return np.array(connectivity)


def angle_connectivity(ibonds):
    """Given the bonds, get the indices of the atoms defining all the bond
    angles

    A 'bond angle' is defined as any set of 3 atoms, `i`, `j`, `k` such that
    atom `i` is bonded to `j` and `j` is bonded to `k`

    Parameters
    ----------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    iangles : np.ndarray, shape[n_angles, 3], dtype=int
        n_angles x 3 array of indices, where each row is the index of three
        atoms m,n,o such that n is bonded to both m and o.
    """

    graph = nx.from_edgelist(ibonds)
    iangles = []

    for i in graph.nodes():
        for (m, n) in combinations(graph.neighbors(i), 2):
            # so now the there is a bond angle m-i-n
            iangles.append((m, i, n))

    return np.array(iangles)


def dihedral_connectivity(ibonds):
    """Given the bonds, get the indices of the atoms defining all the dihedral
    angles

    A 'dihedral angle' is defined as any set of 4 atoms, `i`, `j`, `k`, `l`
    such that atom `i` is bonded to `j`, `j` is bonded to `k`, and `k` is
    bonded to `l`.

    Parameters
    ----------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    idihedrals : np.ndarray, shape[n_dihedrals, 4], dtype=int
        All sets of 4 atoms `i`, `j`, `k`, `l` such that such that atom `i` is
        bonded to `j`, `j` is bonded to `k`, and `k` is bonded to `l`.
    """
    graph = nx.from_edgelist(ibonds)
    idihedrals = []

    # TODO: CHECK FOR DIHEDRAL ANGLES THAT ARE 180 and recover
    # conf : msmbuilder.Trajectory
    #    An msmbuilder trajectory, only the first frame will be used. This
    #    is used purely to make the check for angle(ABC) != 180.

    for a in graph.nodes():
        for b in graph.neighbors(a):
            for c in [c for c in graph.neighbors(b) if c not in [a, b]]:
                for d in [d for d in graph.neighbors(c) if d not in [a, b, c]]:
                    quadruplet = (a, b, c, d) if a < d else (d, c, b, a)
                    if quadruplet not in idihedrals:
                        idihedrals.append(quadruplet)

    return np.array(idihedrals)
