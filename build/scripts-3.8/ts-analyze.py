#!/home/hpark21/miniconda3/envs/python3/bin/python

"""
ts-bondorder.py

Calculate the Mayer bond order matrix and vibrational frequencies for
a transition state.  This is intended to provide some more transition
state info for Sofia's methods.

Requires: ts.xyz (transition state coordinates)

TS vibrational frequencies and modes are saved to irc-transition.vib (vibrational data) .
TS bond order matrix is saved to irc-transition.bnd .
"""

from nanoreactor.molecule import Molecule
from nanoreactor.qchem import QChem, tarexit
from nanoreactor.nifty import _exec
from collections import OrderedDict
import os, sys, re, shutil
import numpy as np
import argparse

tarexit.tarfnm = 'ts-analyze.tar.bz2'
tarexit.include = ['*.xyz', 'irc*', 'qc*', '*.log', '*.err']
tarexit.save = ['*.bnd', '*.vib', '*.log']

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('ts', type=str, help='Coordinate file for transition state (required)')
    parser.add_argument('--charge', type=int, help='Net charge (required)')
    parser.add_argument('--mult', type=int, help='Spin multiplicity (required)')
    parser.add_argument('--method', type=str, help='Electronic structure method (required)')
    parser.add_argument('--basis', type=str, help='Basis set (required)')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def main():
    # Get user input.
    args = parse_user_input()
    _exec("rm -f ts-analyze.tar.bz2 irc_transition.xyz irc_transition.vib", print_command=False)
    # Standardize base name to "irc_transition".
    shutil.copy2(args.ts, 'irc_transition.xyz')
    # Run Q-Chem calculations..
    QCT = QChem("irc_transition.xyz", charge=args.charge, mult=args.mult, method=args.method, basis=args.basis)
    # Ensure stability.
    QCT.make_stable()
    # Single point calculation for bond order matrix.
    QCT.jobtype = 'sp'
    QCT.remextra = OrderedDict([('scf_final_print', '1')])
    QCT.calculate()
    np.savetxt("irc_transition.bnd", QCT.load_qcout().qm_bondorder, fmt="%8.3f")
    # Frequency calculation.
    QCT.jobtype = 'freq'
    QCT.calculate()
    QCT.write_vdata('irc_transition.vib')
    tarexit()

if __name__ == "__main__":
    main()
