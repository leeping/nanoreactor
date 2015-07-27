#!/usr/bin/env python

"""
freezing-string.py

Starting from an initial .xyz file with two structures (reactants and products), 
perform the following sequence of Q-Chem calculations:

1) Optimize the endpoints
2) Freezing string method (to obtain a connecting path)
3) Transition state optimization
4) Intrinsic reaction coordinate

Freezing string results are saved to file as stringfile.txt and Vfile.txt (Q-Chem output format.)
IRC results are saved to file as irc.xyz (coordinates), irc.pop (charge and spin populations), 
irc.nrg (arc length vs. energy) and irc_spaced.xyz (equally spaced coordinates).
"""

from nanoreactor.molecule import Molecule, arc
from nanoreactor.qchem import QChem, QChemTS, QChemIRC, SpaceIRC, tarexit
from nanoreactor.nifty import _exec
from collections import OrderedDict
import os, sys, re, shutil
import numpy as np
import argparse

tarexit.tarfnm = 'freezing-string.tar.bz2'
tarexit.include = ['*.xyz', 'irc*', 'qc*', '*.txt', '*.log', '*.err'] 
Ha_to_kcalmol = 627.5096080305927

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('initial', type=str, help='Two-frame coordinate file with reactant and product structures (required)')
    parser.add_argument('--charge', type=int, help='Net charge (required)')
    parser.add_argument('--mult', type=int, help='Spin multiplicity (required)')
    parser.add_argument('--methods', type=str, nargs='+', default=['b3lyp'], help='Which electronic structure method to use. ' 
                        'Provide 2 names if you want the final TS refinement + IRC to use a different method.')
    parser.add_argument('--bases', type=str, nargs='+', default=['6-31g(d)', '6-31+g(d,p)'], help='Which basis set to use. '
                        'Provide 2 names if you want the final TS refinement + IRC to use a different basis set.')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def main():
    # Get user input.
    args = parse_user_input()
    if len(args.methods) > 2:
        logger.error("Unsure what to do with >2 electronic structure methods")
        raise RuntimeError
    # Delete the result from previous jobs
    _exec("rm -rf fs_result.tar.bz2", print_command=False)
    # The initial path determines whether the IRC should be forwards or backwards.
    S = Molecule(args.initial)
    # Optimize reactant and product geometries, and write them to an input file for freezing string.
    print "Optimizing endpoints..."
    S[0].write("strrct.xyz")
    QcRct = QChem("strrct.xyz", charge=args.charge, mult=args.mult, method=args.methods[0], basis=args.bases[0], qcin="qcopt_rct.in")
    QcRct.opt()
    S[-1].write("strprd.xyz")
    QcPrd = QChem("strprd.xyz", charge=args.charge, mult=args.mult, method=args.methods[0], basis=args.bases[0], qcin="qcopt_prd.in")
    QcPrd.opt()
    RP = QcRct.M[-1] + QcPrd.M[-1]
    RP.write("strrp.xyz")
    # Create the freezing string calculation, and calculate the transition state.
    QCFS = QChem("strrp.xyz", charge=args.charge, mult=args.mult, method=args.methods[0], basis=args.bases[0], qcin="qcfsm.in")
    print "Freezing string.."; QCFS.fsm()
    QCFS.M.write('strts.xyz')

    # Perform transition state search.
    # First run the TS-calculation with the same basis set as the growing string.
    QCTS1 = QChemTS('strts.xyz', charge=args.charge, mult=args.mult, method=args.methods[0], 
                    basis=args.bases[0], finalize=(len(args.methods)==1), qcin='qcts1.in', vout='irc_transition.vib')
    QCTS1.write('ts1.xyz')
    if len(args.methods) == 2:
        print ' --== \x1b[1;92mUpgrading\x1b[0m ==--'
        QCTS2 = QChemTS("ts1.xyz", charge=args.charge, mult=args.mult, method=args.methods[1], 
                        basis=args.bases[1], finalize=True, qcin='qcts2.in', vout='irc_transition.vib')
        QCTS2.write('ts2.xyz')
        qcdir = QCTS2.qcdir
        shutil.copy2('ts2.xyz', 'ts.xyz')
    else:
        qcdir = QCTS1.qcdir
        shutil.copy2('ts1.xyz', 'ts.xyz')
    # Intrinsic reaction coordinate calculation.
    print "Intrinsic reaction coordinate.."
    # Process and save IRC results.
    M_IRC, E_IRC = QChemIRC("ts.xyz", charge=args.charge, mult=args.mult, method=args.methods[-1], basis=args.bases[-1], qcdir=qcdir, xyz0=args.initial)
    M_IRC.write("irc.xyz")
    M_IRC.get_populations().write('irc.pop', ftype='xyz')
    # Save the IRC energy as a function of arc length.
    ArcMol = arc(M_IRC, RMSD=True)
    ArcMolCumul = np.insert(np.cumsum(ArcMol), 0, 0.0)
    np.savetxt("irc.nrg", np.hstack((ArcMolCumul.reshape(-1, 1), E_IRC.reshape(-1,1))), fmt="% 14.6f", header="Arclength(Ang) Energy(kcal/mol)")
    # Create IRC with equally spaced structures.
    M_IRC_EV = SpaceIRC(M_IRC, E_IRC, RMSD=True)
    M_IRC_EV.write("irc_spaced.xyz")
    # Run two final single point calculations with SCF_FINAL_PRINT set to 1
    M_IRC[0].write("irc_reactant.xyz")
    M_IRC[-1].write("irc_product.xyz")
    QCR = QChem("irc_reactant.xyz", charge=args.charge, mult=args.mult, method=args.methods[-1], basis=args.bases[-1])
    QCP = QChem("irc_product.xyz", charge=args.charge, mult=args.mult, method=args.methods[-1], basis=args.bases[-1])
    shutil.copy2("ts.xyz", "irc_transition.xyz")
    QCT = QChem("irc_transition.xyz", charge=args.charge, mult=args.mult, method=args.methods[-1], basis=args.bases[-1])
    def FinalSCF(SP):
        SP.make_stable()
        SP.jobtype = 'sp'
        SP.remextra = OrderedDict([('scf_final_print', '1')])
        SP.calculate()
    FinalSCF(QCR)
    FinalSCF(QCP)
    FinalSCF(QCT)
    np.savetxt("irc_reactant.bnd", QCR.load_qcout().qm_bondorder, fmt="%8.3f")
    np.savetxt("irc_product.bnd", QCP.load_qcout().qm_bondorder, fmt="%8.3f")
    np.savetxt("irc_transition.bnd", QCT.load_qcout().qm_bondorder, fmt="%8.3f")
    # Calculate ZPEs, entropy, enthalpy for Delta-G calcs
    QCR.remextra = OrderedDict()
    QCP.remextra = OrderedDict()
    QCT.remextra = OrderedDict()
    QCR.freq()
    R = QCR.load_qcout()
    QCP.freq()
    P = QCP.load_qcout()
    QCT.freq()
    T = QCT.load_qcout()
    nrg = open('deltaG.nrg', 'w')
    deltaH = P.qm_energies[0]*Ha_to_kcalmol + P.qm_zpe[0] - R.qm_energies[0]*Ha_to_kcalmol - R.qm_zpe[0]
    deltaG = deltaH - P.qm_entropy[0]*0.29815 + P.qm_enthalpy[0] + R.qm_entropy[0]*0.29815 - R.qm_enthalpy[0]
    Ha = T.qm_energies[0]*Ha_to_kcalmol + T.qm_zpe[0] - R.qm_energies[0]*Ha_to_kcalmol - R.qm_zpe[0]
    Ga = Ha - T.qm_entropy[0]*0.29815 + T.qm_enthalpy[0] + R.qm_entropy[0]*0.29815 - R.qm_enthalpy[0]
    nrg.write("=> Delta-H(0K) = %.4f kcal/mol\n" % deltaH)
    nrg.write("=> Delta-G(STP) = %.4f kcal/mol\n" % deltaG)
    nrg.write("=> Activation enthalpy H_a (0K) = %.4f kcal/mol\n" % Ha)
    nrg.write("=> Activation Gibbs energy G_a (STP) = %.4f kcal/mol\n" % Ga)
    nrg.close()
    print "\x1b[1;92mIRC Success!\x1b[0m"
    tarexit()

if __name__ == "__main__":
    main()
