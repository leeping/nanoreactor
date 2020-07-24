"""
Numpy routines to calculate the analytical derivatives of internal coordinates
(bonds, angles, dihedrals) with respect to cartesian coordinates
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function
import numpy as np

##############################################################################
# GLOBALS
##############################################################################

VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
__all__ = ['bond_derivs', 'angle_derivs', 'dihedral_derivs']

##############################################################################
# Functions
##############################################################################


def bond_derivs(xyz, ibonds):
    """
    Derivatives of the bond lengths with respect to cartesian coordinates

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation.
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    bond_derivs : np.ndarray, shape=[n_bonds, n_atoms, 3]
        The derivates are a 3d array, where bond_derivs[i,j,k] gives the
        derivative of the `i`th bond length (the bond between atoms
        ibonds[i,0] and ibonds[i,1]) with respect to the `j`th atom's
        `k`th cartesian coordinate.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    xyz = np.asarray(xyz)
    ibonds = np.asarray(ibonds)

    if not xyz.ndim == 2:
        raise TypeError('xyz has ndim=%d. Should be 2' % xyz.ndim)
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_bonds, two = ibonds.shape
    if two != 2:
        raise TypeError('ibonds must have 2 columns.')

    unique_bond_indices = np.unique(ibonds)
    if not np.all((0 <= unique_bond_indices) &
                  (unique_bond_indices < n_atoms)):
        raise ValueError(('The bond indices must be between 0 and '
                          'n_atoms-1 inclusive. They are zero indexed'))

    derivatives = np.zeros((n_bonds, n_atoms, 3))
    for b, (m, n) in enumerate(ibonds):
        u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])

        derivatives[b, m, :] = u
        derivatives[b, n, :] = -u

    return derivatives


def angle_derivs(xyz, iangles):
    """
    Derivatives of the bond angles with respect to cartesian coordinates

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation.
    iangles : np.ndarray, shape=[n_angles, 3], dtype=int
        Each row in `iangles` is a triplet of indicies `i`, `j`, `k`
        indicating that atoms `i` - `j` - `k` form an angle of interest.

    Returns
    -------
    angle_derivs : np.ndarray, shape=[n_angles, n_atoms, 3]
        The derivates are a 3d array, where angle_derivs[i,j,k] gives the
        derivative of the `i`th bond angle (the angle between atoms
        iangles[i,0] and iangles[i,1] and iangles[i,2]) with respect to the
        `j`th atom's `k`th cartesian coordinate.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    xyz = np.asarray(xyz)
    iangles = np.asarray(iangles)

    if not xyz.ndim == 2:
        raise TypeError('xyz has ndim=%d. Should be 2' % xyz.ndim)
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_angles, three = iangles.shape
    if three != 3:
        raise TypeError('iangles must have 3 columns.')

    unique_angle_indices = np.unique(iangles)
    if not np.all((0 <= unique_angle_indices) &
                  (unique_angle_indices < n_atoms)):
        raise ValueError(('The angle indices must be between 0 and '
                          'n_atoms-1 inclusive. They are zero indexed'))

    derivatives = np.zeros((n_angles, n_atoms, 3))

    for a, (m, o, n) in enumerate(iangles):
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm

        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)

        w = w_prime / np.linalg.norm(w_prime)
        
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm

        derivatives[a, m, :] = term1
        derivatives[a, n, :] = term2
        derivatives[a, o, :] = -(term1 + term2)

    return derivatives


def dihedral_derivs(xyz, idihedrals):
    """
    Derivatives of the dihedral angles with respect to cartesian coordinates

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation.
    idihedrals : np.ndarray, shape=[n_dihedrals, 4], dtype=int
        Each row in `idihedrals` is a quartet of indicies `i`, `j`, `k`, `l`,
        indicating that atoms `i` - `j` - `k` - `l` form a dihedral of
        interest.

    Returns
    -------
    dihedral_derivs : np.ndarray, shape=[n_dihedrals, n_atoms, 3]
        The derivates are a 3d array, where dihedral_derivs[i,j,k] gives the
        derivative of the `i`th dihedral angle (the dihedral between atoms
        idihedrals[i,0] and idihedrals[i,1], idihedrals[i,2] and
        idihedrals[i,3]) with respect to the `j`th atom's `k`th cartesian
        coordinate.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    xyz = np.asarray(xyz)
    idihedrals = np.asarray(idihedrals)

    if not xyz.ndim == 2:
        raise TypeError('xyz has ndim=%d. Should be 2' % xyz.ndim)
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_dihedrals, four = idihedrals.shape
    if four != 4:
        raise TypeError('idihedrals must have 4 columns.')

    unique_dihedral_indices = np.unique(idihedrals)
    if not np.all((0 <= unique_dihedral_indices) &
                  (unique_dihedral_indices < n_atoms)):
        raise ValueError(('The dihedral indices must be between 0 and '
                          'n_atoms-1 inclusive. They are zero indexed'))

    derivatives = np.zeros((n_dihedrals, n_atoms, 3))

    for d, (m, o, p, n) in enumerate(idihedrals):
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])

        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)

        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm

        term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))

        derivatives[d, m, :] = term1
        derivatives[d, n, :] = -term2
        derivatives[d, o, :] = -term1 + term3 - term4
        derivatives[d, p, :] = term2 - term3 + term4

    return derivatives


def main():
    xyz = np.random.randn(10, 3)
    iangles = [[0, 1, 2], [2, 3, 4]]

    print(angle_derivs(xyz, iangles).shape)

if __name__ == '__main__':
    main()
