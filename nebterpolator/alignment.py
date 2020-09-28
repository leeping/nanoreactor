"""Numpy RMSD code
"""

##############################################################################
# Imports
##############################################################################

from __future__ import print_function
import numpy as np

##############################################################################
# Functions
##############################################################################


def kabsch(query, target, operator=True):
    """Compute the RMSD between two structures with he Kabsch algorithm

    Parameters
    ----------
    query : np.ndarray, ndim=2, shape=[n_atoms, 3]
        The set of query points
    target : np.ndarray, ndim=2, shape=[n_atoms, 3]
        The set of reference points to align to
    operator : bool
        Return the alignment operator, which is a callable wrapper for the
        rotation and translation matrix. To align the query points to
        the target, you'd apply the operator to the query, i.e. `op(query)`.

    Returns
    -------
    rmsd : float
        The root-mean-square deviation after alignment
    operator : AlignOperator, optional
        If operator = True, the alignment operator (rot and trans matrix)
        will be returned too.
    """
    if not query.ndim == 2:
        raise ValueError('query must be 2d')
    if not target.ndim == 2:
        raise ValueError('target must be 2d')
    n_atoms, three = query.shape
    if not three == 3:
        raise ValueError('query second dimension must be 3')
    n_atoms, three = target.shape
    if not three == 3:
        raise ValueError('target second dimension must be 3')
    if not query.shape[0] == target.shape[0]:
        raise ValueError('query and target must have same number of atoms')

    # centroids
    m_query = np.mean(query, axis=0)
    m_target = np.mean(target, axis=0)

    # centered
    c_query = query - m_query
    c_target = target - m_target

    error_0 = np.sum(c_query**2) + np.sum(c_target**2)

    A = np.dot(c_query.T, c_target)
    u, s, v = np.linalg.svd(A)

    #d = np.diag([1, 1, np.sign(np.linalg.det(A))])
    #print v.shape

    # LPW: I encountered some mirror-imaging if this line was not included.
    if np.sign(np.linalg.det(A)) == -1:
        v[2] *= -1.0

    rmsd = np.sqrt(np.abs(error_0 - (2.0 * np.sum(s))) / n_atoms)

    if operator:
        rotation_matrix = np.dot(v.T, u.T).T
        translation_matrix = m_query - np.dot(m_target, rotation_matrix)
        return rmsd, AlignOperator(rotation_matrix, translation_matrix)

    return rmsd


class AlignOperator(object):
    """Operator that applys a rotation and translation to an array

    Attributes
    ----------
    rot : np.ndarray, ndim=2
        The rotation matrix
    trans : np.ndarray, ndim=2
        Translation operator
    """
    def __init__(self, rot, trans):
        """Create a callable AlignOperator

        Parameters
        ----------
        rot : np.ndarray, ndim=2
            Rotation matrix
        trans : np.ndarray, ndim=2
            Translation operator
        """
        self.rot = rot
        self.trans = trans

    def __call__(self, matrix):
        return np.dot(matrix, self.rot) + self.trans


def align_trajectory(xyzlist, which='progressive'):
    """Align all of the frames, either to the 0th frame or to one behind it

    Uses a simple version of the Kabsch algorithm, as described on
    wikipedia: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The input cartesian coordinates.
    which : int, or 'progressive'

    Returns
    -------
    c_xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The outut cartesian coordinates, after alignment.
    """
    if xyzlist.ndim != 3:
        raise TypeError('xyzlist must be 3d')

    # center the conformations
    c_xyzlist = xyzlist - np.mean(xyzlist, axis=1)[:, np.newaxis, :]

    if isinstance(which, str):
        if which == 'progressive':
            progressive = True
        else:
            raise ValueError('I didn\'t understand your argument %s' % which)
    else:
        target = c_xyzlist[which, :, :]
        progressive = False

    for i in range(1, len(xyzlist)):
        if progressive:
            target = c_xyzlist[i-i]
        rmsd, operator = kabsch(c_xyzlist[i], target, operator=True)
        c_xyzlist[i] = operator(c_xyzlist[i])

    return c_xyzlist


if __name__ == '__main__':
    "Some test code"

    N = 40
    query = np.arange(N)[:, np.newaxis] * np.random.randn(N, 3)
    target = np.arange(N)[:, np.newaxis] * np.random.randn(N, 3)

    dist, op = kabsch(query, target)
    print('my rmsd        ', dist)

    from msmbuilder.metrics import RMSD
    _rmsdcalc = RMSD()
    t0 = RMSD.TheoData(query[np.newaxis, :, :])
    t1 = RMSD.TheoData(target[np.newaxis, :, :])
    print('msmbuilder rmsd', _rmsdcalc.one_to_all(t0, t1, 0)[0])

    print(np.sqrt(np.sum(np.square(target - op(query))) / N))
