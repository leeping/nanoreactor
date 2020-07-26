#!/home/hpark21/miniconda3/envs/python3/bin/python

"""
transition-state.py

Starting from an initial .xyz file with one structure (TS estimate), 
perform the following sequence of Q-Chem calculations:

1) Transition state optimization
2) Intrinsic reaction coordinate

TS results are saved to ts.xyz (coordinates) and Q-Chem output files named qcts*.
IRC results are saved to file as irc.xyz (coordinates), irc.pop (charge and spin populations), 
irc.nrg (arc length vs. energy) and irc_spaced.xyz (equally spaced coordinates).
"""
from __future__ import print_function
from nanoreactor.molecule import Molecule, arc
from nanoreactor.qchem import QChem, QChemTS, QChemIRC, SpaceIRC, tarexit
from nanoreactor.nifty import _exec
from collections import OrderedDict
import os, sys, re, shutil
import numpy as np
import argparse

tarexit.tarfnm = 'transition-state.tar.bz2'
tarexit.include = ['*.xyz', 'irc*', 'qc*', '*.log', '*.err']
Ha_to_kcalmol = 627.5096080305927

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('tsest', type=str, help='Coordinate file for transition state guess (required)')
    parser.add_argument('--initpath', type=str, default=None, help='Initial pathway for determining direction of IRC (optional)')
    parser.add_argument('--charge', type=int, help='Net charge (required)')
    parser.add_argument('--mult', type=int, help='Spin multiplicity (required)')
    parser.add_argument('--methods', type=str, nargs='+', default=['b3lyp'], help='Which electronic structure method to use. ' 
                        'Provide 2 names if you want the final TS refinement + IRC to use a different method.')
    parser.add_argument('--bases', type=str, nargs='+', default=['6-31g(d)', '6-31+g(d,p)'], help='Which basis set to use. '
                        'Provide 2 names if you want the final TS refinement + IRC to use a different basis set.')
    parser.add_argument('--fragpath', type=str, default=None, help='Path to fragment directory')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def main():
    # Get user input.
    args = parse_user_input()
    if len(args.methods) > 2:
        logger.error("Unsure what to do with >2 electronic structure methods")
        raise RuntimeError
    # Delete the result from previous jobs
    _exec("rm -rf ts_result.tar.bz2", print_command=False)
    # Perform transition state search.
    # First run the TS-calculation with the smaller basis set
    QCTS1 = QChemTS(args.tsest, charge=args.charge, mult=args.mult, method=args.methods[0], 
                    basis=args.bases[0], finalize=(len(args.methods)==1), qcin='qcts1.in', vout='irc_transition.vib')
    QCTS1.write('ts1.xyz')
    if len(args.methods) == 2:
        print(' --== \x1b[1;92mUpgrading\x1b[0m ==--')
        QCTS2 = QChemTS("ts1.xyz", charge=args.charge, mult=args.mult, method=args.methods[1], 
                        basis=args.bases[1], finalize=True, qcin='qcts2.in', vout='irc_transition.vib')
        QCTS2.write('ts2.xyz')
        qcdir = QCTS2.qcdir
        shutil.copy2('ts2.xyz', 'ts.xyz')
    else:
        qcdir = QCTS1.qcdir
        shutil.copy2('ts1.xyz', 'ts.xyz')
    # Intrinsic reaction coordinate calculation.
    print("Intrinsic reaction coordinate..")
    # Process and save IRC results.
    M_IRC, E_IRC = QChemIRC("ts.xyz", charge=args.charge, mult=args.mult, method=args.methods[-1], basis=args.bases[-1], qcdir=qcdir, xyz0=args.initpath)
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
    # Should be able to take out freq calcs of R and P once we get can confidently get rid of this section.
    QCR.remextra = OrderedDict()
    QCP.remextra = OrderedDict()
    QCT.remextra = OrderedDict()
    QCR.freq()
    R = QCR.load_qcout()
    QCP.freq()
    P = QCP.load_qcout()
    QCT.freq()
    T = QCT.load_qcout()
    nrgfile = open('deltaG.nrg', 'w')
    deltaH = P.qm_energies[0]*Ha_to_kcalmol + P.qm_zpe[0] - R.qm_energies[0]*Ha_to_kcalmol - R.qm_zpe[0]
    deltaG = deltaH - P.qm_entropy[0]*0.29815 + P.qm_enthalpy[0] + R.qm_entropy[0]*0.29815 - R.qm_enthalpy[0]
    Ha = T.qm_energies[0]*Ha_to_kcalmol + T.qm_zpe[0] - R.qm_energies[0]*Ha_to_kcalmol - R.qm_zpe[0]
    Ga = Ha - T.qm_entropy[0]*0.29815 + T.qm_enthalpy[0] + R.qm_entropy[0]*0.29815 - R.qm_enthalpy[0]
    nrgfile.write("=> ** The following data is referenced to reactant and product complexes **\n")
    nrgfile.write("=> ** WARNING! Data may not be accurate in cases with >1 molecule        **\n")
    nrgfile.write("=> ** Activation enthalpy H_a (0K) =         %.4f kcal/mol               **\n" % Ha)
    nrgfile.write("=> ** Activation Gibbs energy G_a (STP) =    %.4f kcal/mol               **\n" % Ga)
    nrgfile.write("=> ** Delta-H(0K) =                          %.4f kcal/mol               **\n" % deltaH)
    nrgfile.write("=> ** Delta-G(STP) =                         %.4f kcal/mol               **\n" % deltaG)
    
    # Calculate Delta-G's based on fragment information from reactant and product
    # Get data from fragment nrg files 
    if args.fragpath == None:
        fragpath = os.path.abspath('../../../../../fragments')
    else:
        fragpath = args.fragpath
    formulas = []
    nrg = []
    zpe = []
    entr = []
    enth = []
    validity = []
    for frm in os.listdir(fragpath):
        optdir = os.path.join(fragpath, frm, "opt")
        if os.path.exists(os.path.join(optdir, 'fragmentopt.nrg')):
            fragnrgfile = open(os.path.join(optdir, 'fragmentopt.nrg'))
            formulas.append(fragnrgfile.readline().strip())
            nrg.append(float(fragnrgfile.readline().split()[3]))
            zpe.append(float(fragnrgfile.readline().split()[2]))
            entr.append(float(fragnrgfile.readline().split()[3]))
            enth.append(float(fragnrgfile.readline().split()[3]))
            validity.append(fragnrgfile.readline().strip())
            fragnrgfile.close()
    #Compare list of molecules to choose right energy
    formulasR = []
    formulasP = []
    nrgR = 0.0
    nrgP = 0.0
    R.build_topology()
    for subg in R.molecules:
        formulasR.append(subg.ef())
    formulasR = sorted(formulasR)
    P.build_topology()
    for subg in P.molecules:
        formulasP.append(subg.ef())
    formulasP = sorted(formulasP)
    for i in range(len(formulas)):
        formlist = sorted(formulas[i].split())
        if formlist == formulasR and validity[i] != "invalid":
            nrgR = nrg[i]
            zpeR = zpe[i]
            entrR = entr[i]
            enthR = enth[i]
        if formlist == formulasP and validity[i] != "invalid":
            nrgP = nrg[i]
            zpeP = zpe[i]
            entrP = entr[i]
            enthP = enth[i]
    # Calculate energetics
    if nrgR != 0.0:
        Ha = T.qm_energies[0]*Ha_to_kcalmol + T.qm_zpe[0] - nrgR*Ha_to_kcalmol - zpeR
        Ga = Ha - T.qm_entropy[0]*0.29815 + T.qm_enthalpy[0] + entrR*0.29815 - enthR
        nrgfile.write("=> ## The following data is calculated referenced to isolated reactant and product molecules:\n")
        nrgfile.write("=> ## Activation enthalpy H_a (0K) =         %.4f kcal/mol               ##\n" % Ha)
        nrgfile.write("=> ## Activation Gibbs energy G_a (STP) =    %.4f kcal/mol               ##\n" % Ga)
    else:
        nrgfile.write("=> Reactant state could not be identified among fragment calculations\n")
        nrgfile.write("=> No energetics referenced to isolated molecules will be calculated for this pathway\n")
    if nrgR != 0.0 and nrgP != 0.0:
        deltaH = nrgP*Ha_to_kcalmol + zpeP - nrgR*Ha_to_kcalmol - zpeR
        deltaG = deltaH - entrP*0.29815 + enthP + entrR*0.29815 - enthR 
        nrgfile.write("=> ## Delta-H(0K) =                          %.4f kcal/mol               **\n" % deltaH)
        nrgfile.write("=> ## Delta-G(STP) =                         %.4f kcal/mol               **\n" % deltaG)
    elif nrgP == 0.0:
        nrgfile.write("=> Product state could not be identified among fragment calculations\n")
        nrgfile.write("=> No reaction energies referenced to isolated molecules will be calculated for this pathway\n")
    nrgfile.close()
    print("\x1b[1;92mIRC Success!\x1b[0m")
    tarexit()

if __name__ == "__main__":
    main()
