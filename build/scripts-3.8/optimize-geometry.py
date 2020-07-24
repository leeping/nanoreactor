#!/home/hpark21/miniconda3/envs/python3/bin/python

"""
Run a Q-Chem geometry optimization.  Save frames where the energy is
monotonically decreasing and save charges / spins to disk.
"""

from nanoreactor import contact
from nanoreactor.qchem import QChem, tarexit
from nanoreactor.molecule import Molecule
from nanoreactor.nifty import _exec, click, monotonic_decreasing
import traceback
import argparse
import time
import os, sys
import itertools
import numpy as np
from collections import OrderedDict

tarexit.tarfnm = 'optimize.tar.bz2'
tarexit.include = ['*']

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('initial', type=str, help='initial coordiate file for geometry optimization (required)')
    parser.add_argument('--charge', type=int, help='Net charge (required)')
    parser.add_argument('--cycles', type=int, default=100, help='Number of optimization cycles')
    parser.add_argument('--mult', type=int, help='Spin multiplicity (required)')
    parser.add_argument('--method', type=str, help='Electronic structure method (required)')
    parser.add_argument('--basis', type=str, help='Basis set (required)')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def QCOpt(initial, charge, mult, method, basis, cycles=100, gtol=600, dxtol=2400, etol=200, cart=False):
    """
    Run a Q-Chem geometry optimization.  Default tolerances for
    gradient, displacement and energy are higher than usual because we
    don't want significant nonbonded conformational changes in the
    pathway.
    
    Parameters
    ----------
    initial : str
        Initial XYZ file for optimization
    charge : int
        Net charge
    mult : int
        Spin multiplicity
    method : str
        Electronic structure method
    basis : str
        Basis set
    cycles : int
        Number of optimization cycles
    gtol : int
        Gradient tolerance for Q-Chem
    dxtol : int
        Displacement tolerance for Q-Chem
    etol : int
        Energy tolerance for Q-Chem
    cart : bool
        Perform the optimization in Cartesian coordinates
    """
    # Create Q-Chem object.
    QC = QChem(initial, charge=charge, mult=mult, method=method, basis=basis, qcin='optimize.in')
    # Run a stability analysis first to ensure we're in the ground state.
    QC.make_stable()
    # Set geometry optimization options.
    QC.jobtype = 'opt'
    QC.remextra = OrderedDict()
    if cart: QC.remextra['geom_opt_coords'] = 0
    QC.remextra['thresh'] = 14
    QC.remextra['geom_opt_tol_gradient'] = gtol
    QC.remextra['geom_opt_tol_displacement'] = dxtol
    QC.remextra['geom_opt_tol_energy'] = etol
    QC.remextra['geom_opt_max_cycles'] = cycles
    # Run Q-Chem.
    QC.calculate()
    # Create Molecule object from running Q-Chem.
    M = QC.load_qcout()
    return M

def QCOptIC(*args, **kwargs):
    """ 
    Try to run a Q-Chem geometry optimization; if it fails for some
    reason, then try Cartesian coordinates.
    """
    OptOut = QCOpt(*args, cart=False, **kwargs)
    if OptOut.qcerr == 'OPTIMIZE fatal error':
        print "Internal cartesian coordinates failed - trying Cartesian"
        OptOut = QCOpt(*args, cart=True, **kwargs)
    # Return success on maximum cycles reached.
    if len(OptOut.qcerr) == 0 or OptOut.qcerr == 'Maximum optimization cycles reached':
        return OptOut
    else:
        print "Geometry optimization failed! (%s)" % OptOut.qcerr
        tarexit()

def main():
    # Get user input.
    args = parse_user_input()
    # Start timer.
    click()
    # Run Q-Chem geometry optimization.
    M = QCOptIC(args.initial, args.charge, args.mult, args.method, args.basis, cycles=args.cycles)
    # Select frames where the energy is monotonically decreasing.
    M = M[monotonic_decreasing(M.qm_energies)]
    # Write optimization results to file.
    M.write('optimize.xyz')
    # Write Mulliken charge and spin populations to file.
    QS = M.get_populations()
    QS.write('optimize.pop', ftype='xyz')
    # Archive and exit.
    tarexit()

if __name__ == "__main__":
    main()
