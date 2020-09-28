"""Higher-level methods for manipulating trajectories
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function
# library imports
import numpy as np
import itertools
from scipy.interpolate import UnivariateSpline

# local imports
from . import core
from .alignment import align_trajectory
from .smoothing import buttersworth_smooth, angular_smooth, window_smooth
from .inversion import least_squares_cartesian

##############################################################################
# Globals
##############################################################################

__all__ = ['smooth_internal', 'smooth_cartesian', 'union_connectivity']
DEBUG = True


##############################################################################
# Functions
##############################################################################


def smooth_internal(xyzlist, atom_names, width, allpairs=False, w_morse=0.0, rep=False, anchor=-1, window='hanning', **kwargs):
    """Smooth a trajectory by transforming to redundant, internal coordinates,
    running a 1d timeseries smoothing algorithm on each DOF, and then
    reconstructing a set of consistent cartesian coordinates.

    TODO: write this function as a iterator that yields s_xyz, so that
    they can be saved to disk (async) immediately when they're produced.

    Parameters
    ----------
    xyzlist : np.ndarray
        Cartesian coordinates
    atom_names : array_like of strings
        The names of the atoms. Required for determing connectivity.
    width : float
        Width for the smoothing kernels
    allpairs : bool
        Use all interatomic distances (not just the bonds)
    w_morse: float
        Weight of the Morse potential in the smoothing
    window: string, default='hanning'
        Type of window to perform the averaging

    Other Parameters
    ----------------
    bond_width : float
        Override width just for the bond terms
    angle_width : float
        Override width just for the angle terms
    dihedral_width : float
        Override width just for the dihedral terms
    xyzlist_guess : 
        Cartesian coordinates to use as a guess during the
        reconstruction from internal
    xyzlist_match : 
        Cartesian coordinates to use as a guess during the
        reconstruction from internal

    Returns
    -------
    smoothed_xyzlist : np.ndarray
    """
    bond_width = kwargs.pop('bond_width', width)
    angle_width = kwargs.pop('angle_width', width)
    dihedral_width = kwargs.pop('dihedral_width', width)
    xyzlist_guess = kwargs.pop('xyzlist_guess', xyzlist)
    xyzlist_match = kwargs.pop('xyzlist_match', None)
    for key in list(kwargs.keys()):
        raise KeyError('Unrecognized key, %s' % key)

    ibonds, iangles, idihedrals = None, None, None
    s_bonds, s_angles, s_dihedrals = None, None, None
    ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names, allpairs=allpairs)
    # get the internal coordinates in each frame
    bonds = core.bonds(xyzlist, ibonds)
    angles = core.angles(xyzlist, iangles)
    dihedrals = core.dihedrals(xyzlist, idihedrals)

    # run the smoothing
    s_bonds = np.zeros_like(bonds)
    s_angles = np.zeros_like(angles)
    s_dihedrals = np.zeros_like(dihedrals)
    for i in range(bonds.shape[1]):
        #s_bonds[:, i] = buttersworth_smooth(bonds[:, i], width=bond_width)
        s_bonds[:, i] = window_smooth(bonds[:, i], window_len=bond_width, window=window)
    for i in range(angles.shape[1]):
        #s_angles[:, i] = buttersworth_smooth(angles[:, i], width=angle_width)
        s_angles[:, i] = window_smooth(angles[:, i], window_len=angle_width, window=window)
    # filter the dihedrals with the angular smoother, that filters
    # the sin and cos components separately
    for i in range(dihedrals.shape[1]):
        #s_dihedrals[:, i] = angular_smooth(dihedrals[:, i],
        #    smoothing_func=buttersworth_smooth, width=dihedral_width)
        s_dihedrals[:, i] = angular_smooth(dihedrals[:, i],
                                           smoothing_func=window_smooth, 
                                           window_len=dihedral_width, window=window)

    # compute the inversion for each frame
    s_xyzlist = np.zeros_like(xyzlist_guess)
    errors = np.zeros(len(xyzlist_guess))
    # Thresholds for error and jump
    w_xrefs = 0.0
    for i, xyz_guess in enumerate(xyzlist_guess):
        w_xref = 0.0
        passed = False
        corrected = False
        while not passed:
            passed = False
            if i > 0:
                xref = s_xyzlist[i-1]
            else:
                xref = None
                passed = True
            ramp = 0.1
            if i >= (1.0-ramp)*len(xyzlist_guess):
                w_morse_ = w_morse * float(len(xyzlist_guess)-i-1)/(ramp*len(xyzlist_guess))
            elif i <= ramp*len(xyzlist_guess):
                w_morse_ = w_morse * float(i)/(ramp*len(xyzlist_guess))
            else:
                w_morse_ = w_morse
            r = least_squares_cartesian(s_bonds[i], ibonds, s_angles[i], iangles,
                                        s_dihedrals[i], idihedrals, xyz_guess, xref=xref, w_xref=w_xref, 
                                        elem=atom_names, w_morse=w_morse_, rep=rep)
            s_xyzlist[i], errors[i] = r
    
            if i > 0:
                aligned0 = align_trajectory(np.array([xyzlist[i],xyzlist[i-1]]), 0)
                aligned1 = align_trajectory(np.array([s_xyzlist[i],s_xyzlist[i-1]]), 0)
                maxd0 = np.max(np.abs(aligned0[1] - aligned0[0]))
                maxd1 = np.max(np.abs(aligned1[1] - aligned1[0]))
                if maxd0 > 1e-5:
                    jmp = maxd1 / maxd0
                else:
                    jmp = 0.0
            else:
                maxd1 = 0.0
                jmp = 0.0
            if (not passed) and (anchor < 0 or jmp < anchor):
                passed = True
                if w_xref >= 1.99:
                    w_xrefs = w_xref - 1.0
                elif w_xref < 0.1:
                    w_xrefs = 0.0
                else:
                    w_xrefs = w_xref / 1.5
            elif not passed:
                if w_xref == 0.0:
                    if w_xrefs > 0.0:
                        w_xref = w_xrefs
                    else:
                        w_xref = 2.0**10 / 3.0**10
                else:
                    if w_xref >= 0.99:
                        w_xref += 1.0
                    else:
                        w_xref *= 1.5
                if w_xref > 30:
                    print("\nanchor %f, giving up" % (w_xref))
                    # Set it back to a reasonable (but still high) number
                    w_xrefs = 20.0
                    passed = True
                else:
                    print("jump %f max(dx) %f, trying anchor = %f\r" % (jmp, maxd1, w_xref), end=' ')
                corrected = True
        if xyzlist_match != None:
            aligned_ij = align_trajectory(np.array([s_xyzlist[i], xyzlist_match[i]]), 0)
            maxd_ij = np.max(np.abs(aligned_ij[1] - aligned_ij[0]))
            if maxd_ij < 1e-3:
                print("% .4f" % maxd_ij, "\x1b[92mMatch\x1b[0m")
                s_xyzlist[i:] = xyzlist_match[i:]
                break
        # Print out a message if we had to correct it.
        if corrected:
            print('\rxyz: error %f max(dx) %f jump %s anchor %f' % (errors[i], maxd1, jmp, w_xref))
        if (i%10) == 0:
            print("\rWorking on frame %i / %i" % (i, len(xyzlist_guess)), end=' ')
            print()
        if i > 0:
            print('max(dx) %f (new) %f (old) %s%f\x1b[0m (ratio)' % (maxd1, maxd0, "\x1b[91m" if jmp > 3 else "", jmp))

    #return_value = (interweave(s_xyzlist), interweave(errors))
    return_value = s_xyzlist, errors

    return return_value

def equal_velocity(xyzlist):
    """ Equalize the velocity of a trajectory by performing linear interpolation
    along the time series of maximum atomic displacement. """

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist must be 3d')
    n_frames, n_atoms, n_dims = xyzlist.shape

    aligned = align_trajectory(xyzlist, 'progressive')
    smoothed = np.empty_like(xyzlist)
    ArcMol = np.array([np.max([np.linalg.norm(xyzlist[i+1,j,:]-xyzlist[i,j,:]) 
                               for j in range(n_atoms)]) for i in range(n_frames-1)])
    ArcMolCumul = np.insert(np.cumsum(ArcMol), 0, 0.0)
    ArcMolEqual = np.linspace(0, max(ArcMolCumul), len(ArcMolCumul))
    for a in range(n_atoms):
        for i in range(3):
            smoothed[:,a,i] = np.interp(ArcMolEqual, ArcMolCumul, aligned[:, a, i])
    return align_trajectory(smoothed, 'progressive')

def smooth_cartesian(xyzlist, strength=None, weights=None):
    """Smooth cartesian coordinates with spline on each coordinate

    Parameters
    ----------
    xyzlist : np.ndarray, ndim=3, shape=[n_frames, n_atoms, n_dims]
        The cartesian coordinates
    strength : float, optional
        Positive smoothing factor used to choose the number of knots.
    errors : np.ndarray, ndim=1, shape=[n_frames]
        The weight to assign to each point
    Notes
    -----
    The underlying spline engine is in scipy. It's documentation is here at
    the url below.
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    """

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist must be 3d')
    n_frames, n_atoms, n_dims = xyzlist.shape
    if weights is None:
        weights = np.ones(n_frames, dtype=np.float)
    if not len(weights) == n_frames:
        raise ValueError('errors must be of length n_frames')
    weights /= sum(weights)

    aligned = align_trajectory(xyzlist, 'progressive')
    smoothed = np.empty_like(xyzlist)

    t = np.arange(n_frames)

    for i in range(n_atoms):
        for j in range(n_dims):
            y = aligned[:, i, j]
            #smoothed[:, i, j] = UnivariateSpline(x=t, y=y,
            #                        s=strength, w=weights)(t)
            smoothed[:, i, j] = window_smooth(aligned[:, i, j], window_len=int(strength)*2+1, window='flat')

    return align_trajectory(smoothed, which='progressive')


def union_connectivity(xyzlist, atom_names, allpairs=False):
    """Get the union of all possible proper bonds/angles/dihedrals
    that appear in any frame of a trajectory

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The cartesian coordinates of each frame in a trajectory. The units
        should be in nanometers.
    atom_names : array_like of strings, length=n_atoms
        A list of the names of each of the atoms, which will be used for
        grabbing the covalent radii.
    """
    if not xyzlist.ndim == 3:
        raise ValueError('I need an xyzlist that is 3d')

    # collect a set of all of the bonds/angles/dihedrals that appear in any
    # frame. These will be sets of tuples -- 2-tuples for bonds, 3-tuples for
    # angles, and 4-tuples for dihedrals. We need to use tuples instead of
    # numpy arrays so that they're immutable, which lets the set data
    # structure make sure they're unique
    set_bonds = set()
    set_angles = set()
    set_dihedrals = set()

    for xyz in xyzlist:
        bonds = core.bond_connectivity(xyz, atom_names)
        angles = core.angle_connectivity(bonds)
        dihedrals = core.dihedral_connectivity(bonds)

        set_bonds.update(set([tuple(e) for e in bonds]))
        set_angles.update(set([tuple(e) for e in angles]))
        set_dihedrals.update(set([tuple(e) for e in dihedrals]))

    # Try to make sure we have at least one angle and one dihedral in the system.

    enhance = 1.3
    while (len(set_angles) == 0):
        longbonds = core.bond_connectivity(xyz, atom_names, enhance=enhance)
        angles = core.angle_connectivity(longbonds)
        set_angles.update(set([tuple(e) for e in angles]))
        enhance += 0.05

    enhance = 1.3
    while (len(set_dihedrals) == 0):
        longbonds = core.bond_connectivity(xyz, atom_names, enhance=enhance)
        dihedrals = core.dihedral_connectivity(longbonds)
        set_dihedrals.update(set([tuple(e) for e in dihedrals]))
        enhance += 0.05

    # sort the sets and convert them to the numpy arrays that we
    # prefer. The sorting is not strictly necessary, but it seems good
    # form.

    ibonds = np.array(sorted(set_bonds, key=lambda e: sum(e)))
    iangles = np.array(sorted(set_angles, key=lambda e: sum(e)))
    idihedrals = np.array(sorted(set_dihedrals, key=lambda e: sum(e)))

    # get ALL of the possible bonds
    if allpairs:
        ibonds = np.array(list(itertools.combinations(list(range(xyzlist.shape[1])), 2)))

    return ibonds, iangles, idihedrals
