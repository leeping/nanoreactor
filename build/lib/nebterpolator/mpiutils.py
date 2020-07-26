"""Utilities for working with MPI
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function
import sys
import inspect
import numpy as np
from mpi4py import MPI

##############################################################################
# Globals
##############################################################################

__all__ = ['mpi_root', 'mpi_rank', 'SelectiveExecution', 'group', 'interweave']

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()

##############################################################################
# Code
##############################################################################

def group(iterable, n_groups):
    """Group a list into a list of lists

    This function can be used to split up a big list items into a set to be
    processed by each worker in an MPI scheme.

    Parameters
    ----------
    iterable : array_like
        A list of elements
    n_groups : int
        The number of groups you'd like to make

    Returns
    -------
    groups : list
        A list of lists, where each element in `groups` is a list of
        approximately `len(iterable) / n_groups` elements from `iterable`

    See Also
    --------
    interweave : inverse of this operation
    """
    return [iterable[i::n_groups] for i in range(n_groups)]


def interweave(list_of_arrays):
    """Interweave a list of numpy arrays into a single array

    This function does the opposite of `group`, taking care to to put the elemets
    back in the correct place

    Parameters
    ----------
    list_of_arrays : list of np.ndarray
        A list of numpy arrays, formed (perhaps) by splitting a single large
        array with `group`.

    Returns
    -------
    out_array : np.ndarray

    See Also
    --------
    group : inverse of this operation
    """

    first_dimension = sum(len(e) for e in list_of_arrays)
    output_shape = (first_dimension,) + list_of_arrays[0].shape[1:]

    output = np.empty(output_shape, dtype=list_of_arrays[0].dtype)
    for i in range(SIZE):
        output[i::SIZE] = list_of_arrays[i]
    return output


def mpi_rank(comm=None):
    """Get the rank of the curent MPI node

    Parameters
    ----------
    comm : mpi communicator, optional
        The MPI communicator. By default, we use the COMM_WORLD

    Returns
    -------
    rank : int
        The rank of the current mpi process. The return value is 0
        (the root node) if MPI is not running
    """
    if comm is None:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except ImportError:
            rank = 0
    else:
        rank = comm.Get_rank()

    return rank


class _ExitContext(Exception):
    """Special exception used to skip execution of a with-statement block."""
    pass


class SelectiveExecution(object):
    """Contect manager that executes body only under a certain
    condition.

    http://stackoverflow.com/questions/12594148
    """

    def __init__(self, skip=False):
        """Create the context manager

        Parameters
        ----------
        skip : bool
            If true, the body will be skipped
        """
        self.skip = skip

    def __enter__(self):
        if self.skip:
            # Do some magic
            sys.settrace(lambda *args, **keys: None)
            frame = inspect.currentframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise _ExitContext()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is _ExitContext:
            return True
        else:
            return False


class mpi_root(SelectiveExecution):
    """Context manager that selectively executes its body on the root node"""
    def __init__(self, comm=None):
        """Open a context managert that only executes its body on the root
        node.

        Paramters
        ---------
        comm : mpi communicator, optional
            The MPI communicator. By default, we use the COMM_WORLD
        """
        skip_if = (mpi_rank(comm) != 0)
        super(mpi_root, self).__init__(skip_if)


##############################################################################
# Test code
##############################################################################


def main():
    "Example code"
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    a = None
    with mpi_root():
        print('Initializing a only on rank=%s' % rank)
        a = [1, 2, 3, 4]
    a = comm.bcast(a)

    print('RANK %s, a=%s' % (rank, a))


if __name__ == '__main__':
    main()
