"""
Numpy routines to calculate bond lengths, bond angles, and dihedral
angles from cartesian coordinates.

These routines require that you explicitly specify the indices of atoms
involved in each of the pairs/triplets/quartets.
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function
import numpy as np

##############################################################################
# GLOBALS
##############################################################################

__all__ = ['bonds', 'angles', 'dihedrals']

##############################################################################
# Functions
##############################################################################


def bonds(xyzlist, ibonds):
    """Calculate a set of bond lengths, for each frame in a trajectory

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates of each frame in the trajectory, collected
        in a numpy array
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    bonds : np.ndarray, shape=[n_frames, n_bonds]
        The collected distances, such that distances[i,j] is the distance
        between the jth pair of atoms (whose indices are ibonds[j,0] and
        ibonds[j,1]) in the `i`th frame of the trajectory
    """
    xyzlist = np.asarray(xyzlist)
    ibonds = np.asarray(ibonds, dtype=int)

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist has ndim=%d. Should be 3' % xyzlist.ndim)
    if not ibonds.ndim == 2:
        raise TypeError('ibonds has ndim=%d. Should be 2' % ibonds.ndim)

    n_frames, n_atoms, three = xyzlist.shape
    if three != 3:
        raise TypeError('xyzlist must be of length 3 in the last dimension.')
    n_bonds, two = ibonds.shape
    if two != 2:
        raise TypeError('ibonds must have 2 columns.')

    unique_bond_indices = np.unique(ibonds)
    if not np.all((0 <= unique_bond_indices) &
                  (unique_bond_indices < n_atoms)):
        raise ValueError(('The bond indices must be between 0 and n_atoms-1'
                          ' inclusive. They are zero indexed'))

    diff = xyzlist[:, ibonds[:, 0], :] - xyzlist[:, ibonds[:, 1], :]

    return np.sqrt(np.sum(diff**2, axis=2))


def angles(xyzlist, iangles):
    """Calculate a set of bond lengths, for each frame in a trajectory

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates of each frame in the trajectory, collected
        in a numpy array
    iangles : np.ndarray, shape=[n_angles, 3], dtype=int
        Each row in `iangles` is a triplet of indicies `i`, `j`, `k`
        indicating that atoms `i` - `j` - `k` form an angle of interest.

    Returns
    -------
    angles : np.ndarray, shape=[n_frames, n_angles]
        The collected angles, such that angles[i,j] is the angle between
        the jth triplet of atoms (whose indices are iangles[j,0], iangles[j,1]
        and iangles[j,2]) in the `i`th frame of the trajectory
    """
    xyzlist = np.asarray(xyzlist)
    iangles = np.asarray(iangles, dtype=int)

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist has ndim=%d. Should be 3' % xyzlist.ndim)
    if not iangles.ndim == 2:
        raise TypeError('iangles has ndim=%d. Should be 2' % iangles.ndim)

    n_frames, n_atoms, three = xyzlist.shape
    if three != 3:
        raise TypeError('xyzlist must be of length 3 in the last dimension.')
    n_angles, three = iangles.shape
    if three != 3:
        raise TypeError('iangles must have 3 columns.')

    unique_angle_indices = np.unique(iangles)
    if not np.all((0 <= unique_angle_indices) &
                  (unique_angle_indices < n_atoms)):
        raise ValueError(('The angle indices must be between 0 and n_atoms-1'
                          ' inclusive. They are zero indexed'))

    # vector from first atom to central atom
    vector1 = xyzlist[:, iangles[:, 0], :] - xyzlist[:, iangles[:, 1], :]

    # vector from last atom to central atom
    vector2 = xyzlist[:, iangles[:, 2], :] - xyzlist[:, iangles[:, 1], :]

    # norm of the two vectors
    norm1 = np.sqrt(np.sum(vector1**2, axis=2))
    norm2 = np.sqrt(np.sum(vector2**2, axis=2))

    dot = np.sum(np.multiply(vector1, vector2), axis=2)
    return np.arccos(dot / (norm1 * norm2))


def dihedrals(xyzlist, idihedrals, anchor=None):
    """Calculate a set of bond lengths, for each frame in a trajectory

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates of each frame in the trajectory, collected
        in a numpy array
    idihedrals : np.ndarray, shape=[n_dihedrals, 4], dtype=int
        Each row in `idihedrals` is a quartet of indicies `i`, `j`, `k`, `l`,
        indicating that atoms `i` - `j` - `k` - `l` form a dihedral of
        interest.
    anchor (optional) : np.ndarray, shape=[n_frames, n_dihedrals]
        Ensures that the dihedral angles being returned are within 2*pi of the anchor values.

    Returns
    -------
    dihedrals : np.ndarray, shape=[n_frames, n_dihedrals]
        The collected dihedrals, such that dihedrals[i,j] is the dihedral
        between the jth quartet of atoms (whose indices are idihedrals[j,0],
        idihedrals[j,1], idihedrals[j,2], and idihedrals[j,3]) in the `i`th
        frame of the trajectory
    """
    xyzlist = np.asarray(xyzlist)
    idihedrals = np.asarray(idihedrals, dtype=int)

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist has ndim=%d. Should be 3' % xyzlist.ndim)
    if not idihedrals.ndim == 2:
        raise TypeError('idihedrals has ndim=%d. Should be 2' %
                        idihedrals.ndim)

    n_frames, n_atoms, three = xyzlist.shape
    if three != 3:
        raise TypeError('xyzlist must be of length 3 in the last dimension.')
    n_angles, four = idihedrals.shape
    if four != 4:
        raise TypeError('idihedrals must have 4 columns.')

    unique_dihedral_indices = np.unique(idihedrals)
    if not np.all((0 <= unique_dihedral_indices) &
                  (unique_dihedral_indices < n_atoms)):
        raise ValueError(('The dihedrals indices must be between 0 and '
                          'n_atoms-1 inclusive. They are zero indexed'))

    vec1 = xyzlist[:, idihedrals[:, 1], :] - xyzlist[:, idihedrals[:, 0], :]
    vec2 = xyzlist[:, idihedrals[:, 2], :] - xyzlist[:, idihedrals[:, 1], :]
    vec3 = xyzlist[:, idihedrals[:, 3], :] - xyzlist[:, idihedrals[:, 2], :]

    cross1 = np.cross(vec2, vec3)
    cross2 = np.cross(vec1, vec2)

    arg1 = np.sum(np.multiply(vec1, cross1), axis=2) * \
        np.sqrt(np.sum(vec2**2, axis=2))
    arg2 = np.sum(np.multiply(cross1, cross2), axis=2)

    answer = np.arctan2(arg1, arg2)

    if anchor != None:
        if anchor.shape != answer[0].shape:
            raise TypeError('anchor must have same shape as first element of answer')
        for i in range(answer.shape[0]):
            for j in range(answer.shape[1]):
                if answer[i, j] - anchor[j] > np.pi:
                    answer[i, j] -= 2*np.pi
                elif answer[i, j] - anchor[j] < -np.pi:
                    answer[i, j] += 2*np.pi

    return answer


def main():
    xyzlist = [[[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]]
    ibonds = [[0, 1], [2, 3]]
    print(bonds(xyzlist, ibonds))

    iangles = [[0, 1, 2], [1, 2, 3]]
    print(angles(xyzlist, iangles))

    idihedrals = [[0, 1, 2, 3]]
    print(dihedrals(xyzlist, idihedrals))

if __name__ == '__main__':
    main()
