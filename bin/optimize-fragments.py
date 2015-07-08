#!/usr/bin/env python

"""
Run a Q-Chem geometry optimization.  Save frames where the energy is
monotonically decreasing and save charges / spins to disk.
"""

from nanoreactor import contact
from nanoreactor.nanoreactor import commadash
from nanoreactor.qchem import QChem, tarexit
from nanoreactor.molecule import *
from nanoreactor.nifty import _exec, click, monotonic_decreasing
import networkx as nx
import traceback
import argparse
import time
import os, sys
import itertools
import numpy as np
import tarfile
from collections import OrderedDict

tarexit.tarfnm = 'fragmentopt.tar.bz2'
tarexit.include = ['*']

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=100, help='Number of optimization cycles')
    parser.add_argument('--method', type=str, help='Electronic structure method (required)')
    parser.add_argument('--basis', type=str, help='Basis set (required)')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def QCOpt(initial, charge, mult, method, basis, cycles=100, gtol=600, dxtol=2400, etol=200, cart=False, qcin='optimize.in'):
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
    QC = QChem(initial, charge=charge, mult=mult, method=method, basis=basis, qcin=qcin)
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
    
    subxyz = []
    subna = []
    subchg = []
    submult = []
    subvalid = []
    subefstart = []
    subeffinal = []
    SumFrags = []
    fragfiles = []
    
    # Pick out the files in the tarfile that have the structure "example*.sub_*.xyz"
    idhome = os.path.abspath('..')
    tar = tarfile.open(os.path.join(idhome, 'fragmentid.tar.bz2'), 'r')
    for tarinfo in tar:
        splitlist = tarinfo.name.split(".")
        if len(splitlist) > 2:
            if splitlist[2] == "xyz":
                subxyz.append(tarinfo.name)
                fragfiles.append(tarinfo)
    # Extract these files from the tarfile
    tar.extractall(members=fragfiles)
    tar.close()
    
    # Load files as molecules
    for frag in subxyz:
        M = Molecule(frag)
        M.read_comm_charge_mult()
        subchg.append(M.charge)
        submult.append(M.mult)
        subna.append(M.na)
        efstart = re.search('(?<=Molecular formula )\w+', M.comms[0]).group(0)
        subefstart.append(efstart)

    print "Running individual optimizations."
    # Run individual geometry optimizations.
    FragE = 0.0
    for i in range(len(subxyz)):
        if subna[i] > 1:
            M = QCOptIC(subxyz[i], subchg[i], submult[i], args.method, args.basis, cycles=args.cycles,
                        qcin='%s.opt.in' % os.path.splitext(subxyz[i])[0])
            # Select frames where the energy is monotonically decreasing.
            M = M[monotonic_decreasing(M.qm_energies)]
        else:
            # Special case of a single atom
            QCSP = QChem(subxyz[i], charge=subchg[i], mult=submult[i], method=args.method, 
                         basis=args.basis, qcin='%s.sp.in' % os.path.splitext(subxyz[i])[0])
            QCSP.make_stable()
            QCSP.jobtype = 'sp'
            QCSP.calculate()
            M = QCSP.load_qcout()
        # Write optimization results to file.
        M.write('%s.opt.xyz' % os.path.splitext(subxyz[i])[0])
        # Write Mulliken charge and spin populations to file.
        QS = M.get_populations()
        QS.write('%s.opt.pop' % os.path.splitext(subxyz[i])[0], ftype='xyz')
        # Print out new molecular formulas.
        M = M[-1]
        # This time we should be able to use covalent radii.
        M.build_topology(force_bonds=True)
        optformula = ' '.join([m.ef() for m in M.molecules])
        subeffinal.append(optformula)
        print "%s.opt.xyz : formula %-12s charge %i mult %i energy % 18.10f" % (os.path.splitext(subxyz[i])[0], optformula, subchg[i], submult[i], M.qm_energies[-1])
        FragE += M.qm_energies[-1]
        SumFrags.append(M[-1])
    for fragment in SumFrags : fragment.write('fragmentopt.xyz', append=True)
    if subefstart != subeffinal: print "Fragments changed during optimization, calculation invalid" 
    print "Final energy of optimized frags: % 18.10f" % FragE
    nrg = open('fragmentopt.nrg', 'w')
    for subef in subeffinal : nrg.write(subef + " ")
    nrg.write("\nTotal energy: % 18.10f" % FragE)
    nrg.close()
    # Archive and exit.
    tarexit()

if __name__ == "__main__":
    main()
