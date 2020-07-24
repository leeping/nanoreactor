'''
Wrappers to C functions for computing the geometry at each frame of
a trajectory
'''
import numpy as np
import _contact_wrap

def atom_distances(xyzlist, atom_contacts):
    '''
    For each frame in xyzlist, compute the (euclidean) distance between
    pairs of atoms whos indices are given in contacts.
    
    xyzlist should be a traj_length x num_atoms x num_dims array
    of type float64
    
    contacts should be a num_contacts x 2 array where each row
    gives the indices of 2 atoms whos distance you care to monitor.
    
    Returns: traj_length x num_contacts array of euclidean distances
    
    Note:
    For nice wrappers around this, see the prepare_trajectory method
    of various metrics in metrics.py
    '''
    
    # check shapes
    traj_length, num_atoms, num_dims = xyzlist.shape
    if not num_dims == 3:
        raise ValueError("xyzlist must be an n x m x 3 array")
    try: 
        num_contacts, width = atom_contacts.shape
        assert width is 2
    except (AttributeError, ValueError, AssertionError):
        raise ValueError('contacts must be an n x 2 array')
        
    if not np.all(np.unique(atom_contacts) < num_atoms):
        raise ValueError('Atom contacts goes larger than num_atoms')
    
    # check type
    if xyzlist.dtype != np.float64:
        xyzlist = np.float64(xyzlist)
    if atom_contacts.dtype != np.int32:
        atom_contacts = np.int32(atom_contacts)
    
    # make sure contiguous
    if not xyzlist.flags.contiguous:
        warnings.warn("xyzlist is not contiguous: copying", RuntimeWarning)
        xyzlist = np.copy(xyzlist)
    if not atom_contacts.flags.contiguous:
        warnings.warn("contacts is not contiguous: copying", RuntimeWarning)
        contacts = np.copy(contacts)
    
    results = np.zeros((traj_length, num_contacts), dtype=np.float64)
    
    _contact_wrap.atomic_contact_wrap(xyzlist, atom_contacts, results)
    
    return results

def atom_displacements(xyzlist, atom_contacts):
    '''
    For each frame in xyzlist, compute the (euclidean) distance between
    pairs of atoms whos indices are given in contacts.
    
    xyzlist should be a traj_length x num_atoms x num_dims array
    of type float64
    
    contacts should be a num_contacts x 2 array where each row
    gives the indices of 2 atoms whos distance you care to monitor.
    
    Returns: traj_length x num_contacts array of euclidean distances
    
    Note:
    For nice wrappers around this, see the prepare_trajectory method
    of various metrics in metrics.py
    '''
    
    # check shapes
    traj_length, num_atoms, num_dims = xyzlist.shape
    if not num_dims == 3:
        raise ValueError("xyzlist must be an n x m x 3 array")
    try: 
        num_contacts, width = atom_contacts.shape
        assert width is 2
    except (AttributeError, ValueError, AssertionError):
        raise ValueError('contacts must be an n x 2 array')
        
    if not np.all(np.unique(atom_contacts) < num_atoms):
        raise ValueError('Atom contacts goes larger than num_atoms')
    
    # check type
    if xyzlist.dtype != np.float64:
        xyzlist = np.float64(xyzlist)
    if atom_contacts.dtype != np.int32:
        atom_contacts = np.int32(atom_contacts)
    
    # make sure contiguous
    if not xyzlist.flags.contiguous:
        warnings.warn("xyzlist is not contiguous: copying", RuntimeWarning)
        xyzlist = np.copy(xyzlist)
    if not atom_contacts.flags.contiguous:
        warnings.warn("contacts is not contiguous: copying", RuntimeWarning)
        contacts = np.copy(contacts)
    
    results_dr = np.zeros((traj_length, num_contacts), dtype=np.float64)
    results_dx = np.zeros((traj_length, num_contacts, 3), dtype=np.float64)
    
    _contact_wrap.atomic_displacement_wrap(xyzlist, atom_contacts, results_dr, results_dx)
    
    return results_dr, results_dx

def residue_distances(xyzlist, residue_membership, residue_contacts):
    '''
    For each frame in xyzlist, and for each pair of residues in the
    array contact, compute the distance between the closest pair of
    atoms such that one of them belongs to each residue.
    
    xyzlist should be a traj_length x num_atoms x num_dims array
    of type float64
    
    residue_membership should be a list of lists where
    residue_membership[i] gives the list of atomindices
    that belong to residue i. residue_membership should NOT
    be a numpy 2D array unless you really mean that all of
    the residues have the same number of atoms
    
    residue_contacts should be a 2D numpy array of shape num_contacts x 2 where
    each row gives the indices of the two RESIDUES who you are interested
    in monitoring for a contact.
    
    Returns: a 2D array of traj_lenth x num_contacts where out[i,j] contains
    the distance between the pair of atoms, one from residue_membership[residue_contacts[j,0]]
    and one from residue_membership[residue_contacts[j,1]] that are closest.
    '''
    
    traj_length, num_atoms, num_dims = xyzlist.shape
    if not num_dims == 3:
        raise ValueError("xyzlist must be n x m x 3")
    try: 
        num_contacts, width = residue_contacts.shape
        assert width is 2
    except (AttributeError, ValueError, AssertionError):
        raise ValueError('residue_contacts must be an n x 2 array')
        
    # check type
    if xyzlist.dtype != np.float64:
        xyzlist = np.float64(xyzlist)
    if residue_contacts.dtype != np.int32:
        residue_contacts = np.int32(residue_contacts)
        
    # check contiguous
    if not xyzlist.flags.contiguous:
        warnings.warn("xyzlist is not contiguous: copying", RuntimeWarning)
        xyzlist = np.copy(xyzlist)
    if not residue_contacts.flags.contiguous:
        warnings.warn("contacts is not contiguous: copying", RuntimeWarning)
        residue_contacts = np.copy(residue_contacts)
        
    num_residues = len(residue_membership)
    residue_widths = np.array([len(r) for r in residue_membership], dtype=np.int32)
    max_residue_width = max(residue_widths)
    residue_membership_array = -1 * np.ones((num_residues, max_residue_width), dtype=np.int32)
    for i in xrange(num_residues):
        residue_membership_array[i, 0:residue_widths[i]] = np.array(residue_membership[i], dtype=np.int32)
    
    results = np.zeros((traj_length, num_contacts), np.float64)
        
    _contact_wrap.closest_contact_wrap(xyzlist, residue_membership_array, residue_widths, residue_contacts, results)
    
    return results


