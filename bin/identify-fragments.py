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
from collections import OrderedDict

tarexit.tarfnm = 'fragmentid.tar.bz2'
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

def main():
    # Get user input.
    args = parse_user_input()
    # Start timer.
    click()
    # Run single point calculation to determine bond order matrix.
    QCSP = QChem(args.initial, charge=args.charge, mult=args.mult, method=args.method, basis=args.basis)
    QCSP.make_stable()
    QCSP.jobtype = 'sp'
    QCSP.remextra = OrderedDict([('scf_final_print', '1')])
    QCSP.calculate()
    # Rebuild bond list using the bond order matrix.
    M = QCSP.load_qcout()
    InitE = M.qm_energies[0]
    M.bonds = []
    # Determined from an incredibly small # of data points.
    # In example_1.xyz, an O-H distance of 1.89 Angstrom had a BO of 0.094 (so it should be > 0.1)
    # In example_1.xyz, a C-H distance of 1.39 Angstrom had a BO of 0.305 (so it should be < 0.3)
    bo_thresh = 0.2
    numbonds = 0
    bondfactor = 0.0
    print "Atoms  QM-BO  Distance"
    for i in range(M.na):
        for j in range(i+1,M.na):
            # Print out the ones that are "almost" bonds.
            if M.qm_bondorder[i,j] > 0.1:
                print "%s%i%s%s%i % .3f % .3f" % (M.elem[i], i, '-' if M.qm_bondorder[i,j] > bo_thresh else ' ', M.elem[j], j, M.qm_bondorder[i,j], np.linalg.norm(M.xyzs[-1][i]-M.xyzs[-1][j]))
            if M.qm_bondorder[i,j] > bo_thresh:
                M.bonds.append((i,j))
                numbonds += 1
                bondfactor += M.qm_bondorder[i,j]
    bondfactor /= numbonds
    M.build_topology()
    subxyz = []
    subef = []
    subna = []
    subchg = []
    submult = []
    subvalid = []
    SumFrags = []
    # Divide calculation into subsystems.
    for subg in M.molecules:
        matoms = subg.nodes()
        frag = M.atom_select(matoms)
        # Determine output file name.
        fout = os.path.splitext(args.initial)[0]+'.sub_%i' % len(subxyz)+'.xyz'
        # Assume we are getting the Mulliken charges from the last frame, it's safer.
        # Write the output .xyz file.
        Chg = sum(M.qm_mulliken_charges[-1][matoms])
        SpnZ = sum(M.qm_mulliken_spins[-1][matoms])
        Spn2 = sum([i**2 for i in M.qm_mulliken_spins[-1][matoms]])
        frag.comms = ["Molecular formula %s atoms %s from %s charge %+.3f sz %+.3f sz^2 %.3f" % 
                      (subg.ef(), commadash(subg.nodes()), args.initial, Chg, SpnZ, Spn2)]
        frag.write(fout)
        subxyz.append(fout)
        subef.append(subg.ef())
        subna.append(frag.na)
        # Determine integer charge and multiplicity.
        ichg, chgpass = extract_int(np.array([Chg]), 0.3, 1.0, label="charge")
        ispn, spnpass = extract_int(np.array([abs(SpnZ)]), 0.3, 1.0, label="spin-z")
        nproton = sum([Elements.index(i) for i in frag.elem])
        nelectron = nproton + ichg
        # If calculation is valid, append to the list of xyz/chg/mult to be calculated.
        if (not chgpass or not spnpass):
            print "Cannot determine charge and spin for fragment %s\n" % subg.ef()
            subchg.append(None)
            submult.append(None)
            subvalid.append(False)
        elif ((nelectron-ispn)/2)*2 != (nelectron-ispn):
            print "Inconsistent charge and spin-z for fragment %s\n" % subg.ef()
            subchg.append(None)
            submult.append(None)
            subvalid.append(False)
        else:
            subchg.append(ichg)
            submult.append(ispn+1)
            subvalid.append(True)
    print "%i/%i subcalculations are valid" % (len(subvalid), sum(subvalid))
    for i in range(len(subxyz)):
        print "%s formula %-12s charge %i mult %i %s" % (subxyz[i], subef[i], subchg[i], submult[i], "valid" if subvalid[i] else "invalid")
    fragid = open('fragmentid.txt', 'w')
    for formula in subef : fragid.write(formula+" ")
    fragid.write("\nBondfactor: " + str(bondfactor))
    if not all(subvalid): fragid.write("\n invalid calculation")
    fragid.close()
    # Archive and exit
    tarexit()

if __name__ == "__main__":
    main()
