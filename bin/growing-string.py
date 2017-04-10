#!/usr/bin/env python

"""
growing-string.py

Starting from an initial .xyz file with an initial pathway
perform a growing string calculation.
"""

from nanoreactor.molecule import Molecule
from nanoreactor.qchem import tarexit, prepare_template, qcrem_default
from nanoreactor.nifty import _exec
import os, sys, re, shutil
import argparse

tarexit.tarfnm = 'growing-string.tar.bz2'
tarexit.include = ['*.xyz', 'gs*', '*.num', '*.nsb', 'GRAD*', '*.txt', '*.log']

# Adjustable Settings
# Not used at this time (see qcrem_default in qchem.py)
# qcoptions="""$rem
# jobtype             force
# exchange            {method}
# basis               {basis}
# symmetry            off
# sym_ignore          true
# unrestricted        true
# scf_guess           core
# scf_guess_mix       5
# max_scf_cycles      50
# $end
# """

inpfile="""# This text matches exactly with baron's old code
#
# General information required for all string jobs
#
------------- General Info ---------------------
{xyz} # name of initial (old) string. A file with series of xyzs
1,{npts}    # begining and end points for the new string
{images}    # number of points in final string
QCHEM       # software :(QCHEM, JAGUAR, G03, TURBOMOL)
{npts}      # number of points in new string at start of sim., make it 4 or more
0           # how many fixed atoms, these are to be listed last in xyz file
------------------------------------------------

# Needed only if QCHEM is the software used
# 
------------- QCHEM Info ---------------------
qcoptions.in                                                        # file containing typical input file, coords not important
{path}                                                              # full path for main scratch dir; end with "/"
qcdir                                                               # name of run , name of scratch dir of this run
----------------------------------------------

# If you need to change any defaults add them here
# The order of things dont matter here
# If not required to change default, then delete from below
------------ Default Info -------------------
MAX_ITERS               {cyc}    # maximum number of iterations
PERP_GRAD_TOL           0.002 # perperndicular gradient for convergence
END_PERP_GRAD_TOL       0.040 # if grad < TOL then grow end of string
MAX_DX                  0.800 # max MWC  distance a node can move? default 0.80
ALPHA_TOL               0.05  # max MWC  distance a node can move? default 0.05
EPSILON_CUTOFF          1.00  # nodes with (grad < EPS*AVG_GRAD) will be frozen
CHARGE              {chg}     # Charge on structure
SPIN                {mult}     # spin: singlet or triplet?
MASS_WEIGHTED         YES     # mass weighted coords
#MASS_WEIGHT_FILE   mass.txt  # file containing mass-weights, if needed
----------------------------------------------
"""

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('initial', type=str, help='Initial pathway for growing string calculation')
    parser.add_argument('--charge', type=int, help='Net charge (required)')
    parser.add_argument('--mult', type=int, help='Spin multiplicity (required)')
    parser.add_argument('--method', type=str, help='Electronic structure method (required)')
    parser.add_argument('--epsilon', type=float, help='Dielectric constant for polarizable continuum (optional)')
    parser.add_argument('--basis', type=str, help='Basis set (required)')
    parser.add_argument('--stab', action='store_true', help='Perform stability analysis before gradient calculations')
    parser.add_argument('--images', type=int, help='Number of images in the string')
    parser.add_argument('--cycles', type=int, default=50, help='Number of growing string cycles')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def main():
    # Get user input.
    args = parse_user_input()
    M = Molecule(args.initial)

    # Write growing string input files.
    # Note that cycles are "throttled" to 100, because larger numbers will hit queue time limits and we lose all the work :(
    # with open("qcoptions.in",'w') as f: print >> f, qcoptions.format(method=args.method, basis=args.basis)
    prepare_template(qcrem_default, "qcoptions.in", args.charge, args.mult, args.method, args.basis, epsilon=args.epsilon, molecule=M)
    with open("inpfile",'w') as f: print >> f, inpfile.format(xyz=args.initial, path=os.getcwd()+'/', 
                                                              chg=args.charge, mult=args.mult, cyc=min(100, args.cycles),
                                                              npts=len(M), images=args.images)

    # Bootleg file-based interface for triggering stability analysis.
    if args.stab: _exec("touch Do_Stability_Analysis", print_command=False)

    # Execute the growing string calculation.
    try:
        _exec("gstring.exe", print_command=True, print_to_screen=True)
        _exec("cp $(ls str[0-9]*.xyz -v | tail -1) final-string.xyz", print_command=True, persist=True)
        Images = None
        for qcout in os.popen("ls gs[0-9][0-9].out -v").readlines():
            if Images == None:
                Images = Molecule(qcout.strip(), build_topology=False).without('qcrems')
            else:
                Images += Molecule(qcout.strip(), build_topology=False).without('qcrems')
        Images.get_populations().write('final-string.pop', ftype="xyz")
    except:
        tarexit.include=['*']
        tarexit()

    # Archive files and exit.
    tarexit()

if __name__ == "__main__":
    main()
