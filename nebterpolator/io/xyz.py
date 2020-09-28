"""Higher-level methods for manipulating trajectories
"""

##############################################################################
# Imports
##############################################################################

# library imports
import itertools
import numpy as np
from datetime import datetime

##############################################################################
# Classes
##############################################################################


class XYZFormatError(Exception):
    pass


class XYZFile(object):
    def __init__(self, filename, mode='r'):
        """Open a .xyz format file

        Parameters
        ----------
        filename : str
            The path on the filesystem to open
        mode : str
            The mode in which to open the file
        """
        self._handle = open(filename, mode)
        self._closed = False

    def read_frame(self):
        """Read a single molecule/frame from an xyz file

        Returns
        -------
        xyz : np.ndarray, shape=[n_atoms, 3], dtype=float
            The cartesian coordinates
        atom_names : list of strings
            A list of the names of the atoms
        """
        try:
            n_atoms = self._handle.readline()
            if n_atoms == '':
                raise EOFError('The end of the file was reached')
            n_atoms = int(n_atoms)
        except ValueError:
            raise XYZFormatError('The number of atoms wasn\'t parsed '
                                 'correctly.')

        comment = self._handle.readline()
        atom_names = [None for i in range(n_atoms)]
        xyz = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            line = self._handle.readline().split()
            if len(line) != 4:
                raise XYZFormatError('line was not 4 elemnts: %s' % str(line))

            atom_names[i] = line[0]
            try:
                xyz[i] = np.array([float(e) for e in line[1:]], dtype=float)
            except:
                raise XYZFormatError('The coordinates were not correctly '
                                     'parsed.')

        return xyz, atom_names

    def read_trajectory(self):
        """Read all of the frames from a xyzfile

        Returns
        -------
        xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
            The cartesian coordinates
        atom_names : list of strings
            A list of the names of the atoms
        """
        xyz, atom_names = self.read_frame()

        np_atom_names = np.array(atom_names)
        xyzlist = [xyz]

        for i in itertools.count(1):
            try:
                xyz, tmp_atom_names = self.read_frame()
                xyzlist.append(xyz)
                if not np.all(np_atom_names == np.array(tmp_atom_names)):
                    raise XYZFormatError('Frame %d does not contain the same'
                                         'atoms as the other frames' % i)
            except EOFError:
                break

        return np.array(xyzlist), atom_names

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self._handle.close()
            self._closed = True

    def write_frame(self, xyz, atom_names, comment=None):
        """Write a single frame to an xyz format file

        Parameters
        ----------
        xyzlist : np.ndarray, shape=[n_atoms, 3], dtype=float
            The cartesian coordinates of the frame
        atom_names : list of strings
            A list of the names of the atoms
        """
        n_atoms = len(atom_names)

        if comment is None:
            cts = datetime.today().strftime('%c')
            comment = 'Frame written by nebterpolator, %s\n' % cts

        xyz = np.asarray(xyz, dtype=float)
        if not xyz.ndim == 2:
            raise TypeError('xyz is not 2d')
        if not xyz.shape[0] == n_atoms:
            raise TypeError('number of columns in xyz doesn\'t match number '
                            'of atoms in atom_names')
        if not xyz.shape[1] == 3:
            raise TypeError('xyz does not have 3 columns')

        self._handle.write('%s\n' % n_atoms)
        self._handle.write(comment)

        for i in range(n_atoms):
            line = '%-4s %14.10f %14.10f %14.10f\n' % \
                (atom_names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2])
            self._handle.write(line)

    def write_trajectory(self, xyzlist, atom_names):
        """Write a trajectory to an xyz file

        Parameters
        ----------
        xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
            The cartesian coordinates of each frame in the trajectory
        atom_names : list of strings
            A list of the names of the atoms
        """

        cts = datetime.today().strftime('%c')
        comment = 'Frame written by nebterpolator, %s\n' % cts

        for xyz in xyzlist:
            self.write_frame(xyz, atom_names, comment)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
