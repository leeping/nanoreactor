#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore")
from nanoreactor import contact
from nanoreactor.qchem import *
from nanoreactor.molecule import *
from nanoreactor.nanoreactor import *
from nanoreactor.nifty import _exec, printcool_dictionary, printcool
import argparse
from scipy import optimize
import time
import math
import os, sys, re, socket
import itertools
from copy import deepcopy
from collections import OrderedDict, Counter

method = sys.argv[2]
basis = sys.argv[3]

def tarexit():
    # Archive files and quit.
    _exec("rm -f *.btr", print_command=False)
    _exec("touch %s.err" % os.path.splitext(sys.argv[1])[0], print_command=False)
    _exec("tar cjf result.tar.bz2 *.xyz %s" % ' '.join([i for i in os.listdir('.') if (i.startswith('qc') and not os.path.isdir(i))]), print_command=False)
    _exec("touch result.tar.bz2", print_command=False)
    sys.exit()

def unitgrad(grad):
    flatgrad = grad.flatten()
    flatgrad /= np.linalg.norm(flatgrad)
    return flatgrad

def polyfix(series, order):
    # Returns a polynomial fit of a time series, with the beginning and ending values fixed.
    def polynomial(b): # Polynomial of a particular order. b is the order minus one.
        def f(x):
            a0 = series[0]
            a1 = series[-1] - series[0] - sum(b)
            a = [a0, a1] + list(b)
            ans = np.sum(np.array([a[i] * x ** i for i in range(len(a))]),axis=0)
            return ans
        return f
    # Starting guess of independent parameters, corresponding to linear transit.
    b0 = np.zeros(order-1, dtype=float)
    x = np.linspace(0, 1, len(series))
    def objective(b):
        fn = polynomial(b)
        df = series - fn(x)
        ans = np.dot(df,df)
        return ans
    bopt = optimize.fmin(objective, b0, disp=0)
    return polynomial(bopt)(x)

finalfns = []
nebfns = []

printcool("STEP 1: Determining the charge and spin")
   
M = Molecule(sys.argv[1])
print M.comms[0].split("atoms")[0]

chg, spn = extract_pop(M)
if chg == -999: tarexit()

# This is the string from the xyz comment line that describes which atoms went to which.
AtomGroups = [[tuple(uncommadash(j)) for j in eval(i)] for i in re.findall('\[.*\]', M.comms[0])[0].split("->")]
AtomsFirst = AtomGroups[0]
AtomsLast = AtomGroups[1]

printcool("STEP 2: Finding geometry-optimized endpoints for path using Q-Chem")

chkcyc = 20

def QCOpt(M, frame, chg, spn, basename, cart=False, check = False, gtol=1000, dxtol=5000, etol=300, cyc=50, nanoref = None):
    basexyz = basename+".xyz"
    M[frame].write(basexyz)
    QC = QChem(basexyz, charge=chg, mult=spn+1, method=method, basis=basis, clean=True)
    QC_ = None
    if (not check) or (not os.path.exists(QC.qcout)):
        if cart: QC.remextra['geom_opt_coords'] = 0
        QC.make_stable()
        QC.jobtype = 'opt'
        QC.remextra = OrderedDict()
        QC.remextra['thresh'] = 14
        if nanoref != None:
            QC.remextra['geom_opt_tol_gradient'] = 5000
            QC.remextra['geom_opt_tol_displacement'] = 10000
            QC.remextra['geom_opt_tol_energy'] = 500000
            QC.remextra['geom_opt_max_cycles'] = chkcyc
            # os.system("cp -a %s %s" % (QC.qcdsav, QC.qcdsav+".b4opt"))
            QC.calculate()
            QC_ = QC.load_qcout()
            QC_[-1].write(".tmp.xyz")
            nanochk = Nanoreactor(".tmp.xyz", enhance=1.2, printlvl=-1, boring=[])
            if NanoEqual(nanochk, nanoref):
                QC_.full = 0
                return QC_
            QC.M.xyzs = [QC_.xyzs[-1]]
            # # Restart the geometry optimization.
            # QC.M.xyzs = [M[frame]]
            # os.system("rm -rf %s %s" % (QC.qcdir, QC.qcdsav))
            # os.system("mv %s %s" % QC.qcdsav+".b4opt", QC.qcdsav)
        QC.remextra['geom_opt_tol_gradient'] = gtol
        QC.remextra['geom_opt_tol_displacement'] = dxtol
        QC.remextra['geom_opt_tol_energy'] = etol
        QC.remextra['geom_opt_max_cycles'] = cyc
        QC.calculate()
    elif check:
        print "Skipping Q-Chem calculation, output file exists"
    M1 = QC.load_qcout()
    if QC_ != None:
        # Ensure that the two objects can be added
        QC_.qcerr = M1.qcerr
        QC_.qcrems = M1.qcrems
        QC_ += M1
        QC_.full = 1
        return QC_
    else:
        M1.full = 1
        return M1

def QCOptIC(M, frame, chg, spn, basename, check=False, gtol=1000, dxtol=5000, etol=300, cyc=50, nanoref=None):
    # Try Cartesian optimization if internal coordinates don't work.
    try: 
        return QCOpt(M, frame, chg, spn, basename, cart=False, check=check, gtol=gtol, dxtol=dxtol, etol=etol, cyc=cyc, nanoref=nanoref)
    except: 
        try:
            traceback.print_exc()
            print "Cartesian..",
            return QCOpt(M, frame, chg, spn, basename, cart=True, check=check, gtol=gtol, dxtol=dxtol, etol=etol, cyc=cyc, nanoref=nanoref)
        except:
            traceback.print_exc()
            print "Geometry optimization failed"
            tarexit()

t0 = time.time()

def arc(Mol, begin=None, end=None, RMSD=True):
    # Get the arc-length for a trajectory segment.  
    # Uses RMSD or maximum displacement of any atom in the trajectory.
    Mol.align()
    if begin == None:
        begin = 0
    if end == None:
        end = len(Mol)
    if RMSD:
        #Rmsd = Mol.all_pairwise_rmsd()
        #Arc = [Rmsd[i,i+1] for i in range(Rmsd.shape[0]-1)]
        Arc = Mol.pathwise_rmsd()
    else:
        Arc = np.array([np.max([np.linalg.norm(Mol.xyzs[i+1][j]-Mol.xyzs[i][j]) for j in range(Mol.na)]) for i in range(begin, end-1)])
    return Arc

def ConstantVelocity(Mol, RMSD=True):
    ArcMol = arc(Mol, RMSD=RMSD)
    ArcMolCumul = np.insert(np.cumsum(ArcMol), 0, 0.0)
    ArcMolEqual = np.linspace(0, max(ArcMolCumul), len(ArcMolCumul))
    xyzold = np.array(Mol.xyzs)
    xyznew = np.zeros(xyzold.shape)
    for a in range(Mol.na):
        for i in range(3):
            xyznew[:,a,i] = np.interp(ArcMolEqual, ArcMolCumul, xyzold[:, a, i])
    Mol.xyzs = list(xyznew)

RxnBegin = 0
RxnEnd = len(M)-1

print "Optimizing reactant frame."
QCR = QCOptIC(M, RxnBegin, chg, spn, "qcopt_reactants", check=True)
QCR.write("qcopt_reactants.xyz")
QCR[-1].write("qcmin_reactants.xyz")
NanoQCR = Nanoreactor("qcmin_reactants.xyz", enhance=1.2, printlvl=-1, boring=[])
AtomsQCR = set([tuple(i['graph'].L()) for i in NanoQCR.TimeSeries.values()])

print "Optimizing product frame." 
QCP = QCOptIC(M, RxnEnd, chg, spn, "qcopt_products", check=True)
QCP.write("qcopt_products.xyz")
QCP[-1].write("qcmin_products.xyz")
NanoQCP = Nanoreactor("qcmin_products.xyz", enhance=1.2, printlvl=-1, boring=[])
AtomsQCP = set([tuple(i['graph'].L()) for i in NanoQCP.TimeSeries.values()])

preserves = "\x1b[92mpreserves\x1b[0m"
alters = "\x1b[91malters\x1b[0m"
print

print "Q-Chem optimization %s reactants and %s products" % ((preserves if set(AtomsQCR) == set(AtomsFirst) else alters), (preserves if set(AtomsQCP) == set(AtomsLast) else alters))

def CommonAtoms(Nano1, Nano2):
    Commons = []
    # GIDs are not guaranteed to be identical for two Nanoreactor objects
    # even if atom numbers and graphs are the same, because they are tied
    # to the isomer number
    CGids1 = []
    CGids2 = []
    for gid1, ts1 in Nano1.TimeSeries.items():
        for gid2, ts2 in Nano2.TimeSeries.items():
            if ts1['graph'] == ts2['graph'] and ts1['graph'].L() == ts2['graph'].L():
                Commons += ts1['graph'].L()
                CGids1.append(gid1)
                CGids2.append(gid2)
    return Commons, CGids1, CGids2

if NanoEqual(NanoQCR, NanoQCP):
    print "Q-Chem reactant and product optimizations converged to the same point; reaction may have been \x1b[93mspontaneous\x1b[0m"
    print "There is nothing more to be done."
    tarexit()

if set(NanoQCR.Isomers) == set(NanoQCP.Isomers):
    print "Q-Chem optimized reactant and product frames consist of the same molecules, possibly a rearrangement"
        
qctime = time.time() - t0
print "Q-Chem walltime: %.1f s" % (qctime)

def InterpolatePath(Path1, fnm, verbose=True, outfnm=None):
    """ Given a path in the form of a Molecule object, 
    equalize the velocities and perform internal coordinate interpolation. 
    If outfnm is provided, then the final smoothed path is saved to that file name.
    Otherwise, the "stage1" and "stage2" files will be written.
    """

    # First align the frames in the path.
    Path1.align()

    # Perform a linear interpolation so we get a constant movement path that isn't jerky.
    ArcPath1 = arc(Path1, RMSD=True)
    if verbose: print "Before velocity equalizing, movement is %.3f +- %.3f A, max %.3f, min %.3f" % (np.mean(ArcPath1), np.std(ArcPath1), np.max(ArcPath1), np.min(ArcPath1))
    MaxVel = np.max(ArcPath1)
    ConstantVelocity(Path1, RMSD=True)
    ArcPath1 = arc(Path1, RMSD=True)
    ArcPath1Cumul = np.insert(np.cumsum(ArcPath1), 0, 0.0)
    if verbose: print "After velocity equalizing,  movement is %.3f +- %.3f A, max %.3f, min %.3f" % (np.mean(ArcPath1), np.std(ArcPath1), np.max(ArcPath1), np.min(ArcPath1))
    
    # Write the path to a file
    path1fnm = "%s_stage1.xyz" % os.path.splitext(fnm)[0]
    if verbose: print "Writing velocity-equalized path to %s" % path1fnm
    Path1.write(path1fnm)
    
    path2fnm = "%s_stage2.xyz" % os.path.splitext(fnm)[0]
    neb0fnm = "%s_dc.xyz" % os.path.splitext(fnm)[0]

    # The smoothing strength is set to one-third of the trajectory length.
    # Values too large lead to unreliable results.  This is also fine tuned.
    smoothing_strength = min(61, (len(Path1) / 5)*2+1)
    
    # This requires the nebterpolator package to be in the current directory.
    if not os.path.exists(path2fnm):
        _exec("Nebterpolate.py %s %i %s" % (path1fnm, smoothing_strength, path2fnm), print_to_screen=verbose, print_command=verbose, persist=True)
    else:
        print "Interpolated path exists"

    if os.path.exists(path2fnm):
        if verbose: print "Interpolated path is saved to %s" % path2fnm
        Path2 = Molecule(path2fnm)
        ArcPath2 = arc(Path2, RMSD=True)
        ArcPath2Cumul = np.insert(np.cumsum(ArcPath2), 0, 0.0)
        if verbose: print "Internal interpolation modifies the path length from %.3f to %.3f Angstrom, ; max movement %.3f" % (ArcPath1Cumul[-1], ArcPath2Cumul[-1], np.max(ArcPath2))
        if np.max(ArcPath2) > 2.0*MaxVel:
            if verbose: print "\x1b[91mJump detected!\x1b[0m"
            Jump = True
        else:
            Jump = False
        ConstantVelocity(Path2, RMSD=True)
        ArcPath2 = arc(Path2, RMSD=True)
        if verbose: print "Velocity equalized: movement is %.3f +- %.3f A, max %.3f, min %.3f" % (np.mean(ArcPath2), np.std(ArcPath2), np.max(ArcPath2), np.min(ArcPath2))
        Path2.comms = Path1.comms[:]
        Path2.write(path2fnm)
        # Print out 11 frames for NEB.
        if Jump or (ArcPath2Cumul[-1] > ArcPath1Cumul[-1]):
            if verbose: print "Internal interpolation did \x1b[91mnot\x1b[0m improve the trajectory"
            Path1_ = Molecule(path1fnm)
            feleven = np.array([int(round(i)) for i in np.linspace(0, len(Path1_)-1, min(len(Path1_), 11))])
            Path1_[feleven].write(neb0fnm)
            BestPath = path1fnm
        else:
            if verbose: print "Internal interpolation has \x1b[92mimproved\x1b[0m the trajectory"
            feleven = np.array([int(round(i)) for i in np.linspace(0, len(Path2)-1, min(len(Path2), 11))])
            Path2[feleven].write(neb0fnm)
            BestPath = path2fnm
    else:
        if verbose: print "Internal interpolation did not generate a file, it may have failed"
        Path1_ = Molecule(path1fnm)
        feleven = np.array([int(round(i)) for i in np.linspace(0, len(Path1_)-1, min(len(Path1_), 11))])
        Path1_[feleven].write(neb0fnm)
        BestPath = path1fnm
    if verbose: print "NEB starting path saved to %s" % neb0fnm
    if outfnm != None:
        os.system("mv %s %s" % (BestPath, outfnm))
        os.system("rm -f %s" % path1fnm)
        os.system("rm -f %s" % path2fnm)
        finalfns.append(outfnm)
        nebfns.append(neb0fnm)
    else:
        finalfns.append(BestPath)
        nebfns.append(neb0fnm)
    return BestPath

printcool("STEP 3: Performing internal coordinate smoothing of the reaction path")

t0 = time.time()

# At this point, we are finished with processing the endpoints.
# There is a chance that the endpoints now contain spectators.  We will remove them.

satoms, sgid0, sgid1 = CommonAtoms(NanoQCR, NanoQCP)
if len(satoms) > 0:
    print "After optimizing endpoints, atoms", ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for j, i in NanoQCR.TimeSeries.items() if j in sgid0]), "no longer participate in the reaction"
ratoms = sorted(list(set(range(M.na)) - set(satoms)))

QCR = QCR.atom_select(ratoms)
QCR.write("qcopt_reactants.xyz")
QCR[-1].write("qcmin_reactants.xyz")
NanoQCR = Nanoreactor("qcmin_reactants.xyz", enhance=1.2, printlvl=-1, boring=[])

QCP = QCP.atom_select(ratoms)
QCP.write("qcopt_products.xyz")
QCP[-1].write("qcmin_products.xyz")
NanoQCP = Nanoreactor("qcmin_products.xyz", enhance=1.2, printlvl=-1, boring=[])

QCR1 = deepcopy(QCR)
QCP1 = deepcopy(QCP)
for i in QCR1.Data.keys():
    if 'qm' in i or 'qc' in i:
        del QCR1.Data[i]
        del QCP1.Data[i]

QCR1.comms = [M.comms[0] for i in range(len(QCR1))]
QCP1.comms = [M.comms[-1] for i in range(len(QCP1))]

# Construct the path from the Q-Chem optimizations and the nanoreactor trajectory slice.
RxnPath = QCR1[::-1] + M.atom_select(ratoms)[RxnBegin+1:RxnEnd] + QCP1

basefn = re.sub("_stage[0-9]","",os.path.splitext(os.path.split(sys.argv[1])[-1])[0])

newpathfnm = InterpolatePath(RxnPath, basefn)

print "Interpolation walltime: %.1f s" % (time.time() - t0)

def FindGroups(sl1, sl2):
    # Given two lists of atom lists, find the groups of sets in each list
    # that only contain each others' elements (i.e. if somehow we have
    # two parallel reactions in one.)
    sl1c = [set(s) for s in sl1]
    sl2c = [set(s) for s in sl2]
    while set([tuple(sorted(list(s))) for s in sl1c]) != set([tuple(sorted(list(s))) for s in sl2c]):
        for s1 in sl1c:
            for s2 in sl2c:
                if len(s1.intersection(s2)) > 0:
                    s1.update(s2)
                    s2.update(s1)
    result = sorted([list(t) for t in list(set([tuple(sorted(list(s))) for s in sl1c]))])
    return result

def GetIntermediates(QC0, NanoQC0, QC1, NanoQC1):
    specatoms, sgid0, sgid1 = CommonAtoms(NanoQC0, NanoQC1)
    activeall = sorted(list(set(range(M2.na)) - set(specatoms)))
    Act0 = [i['graph'].L() for i in NanoQC0.TimeSeries.values() if any([j in activeall for j in i['graph'].L()])]
    Act1 = [i['graph'].L() for i in NanoQC1.TimeSeries.values() if any([j in activeall for j in i['graph'].L()])]
    activegrps = FindGroups(Act0, Act1)
    infos = []
    if len(activegrps) > 1:
        print "\x1b[96m%i concurrent reactions\x1b[0m are happening, splitting them" % len(activegrps)
    for actives in activegrps:
        sgid0a = [k for k, v in NanoQC0.TimeSeries.items() if all([i not in actives for i in v['graph'].L()])]
        sgid1a = [k for k, v in NanoQC1.TimeSeries.items() if all([i not in actives for i in v['graph'].L()])]
        # print sgid0a, sgid1a
        print "reaction    :", ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for j, i in NanoQC0.TimeSeries.items() if j not in sgid0a]), 
        print "->", ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for j, i in NanoQC1.TimeSeries.items() if j not in sgid1a])
        if len(sgid1a) > 0:
            print "speculators :", ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for j, i in NanoQC1.TimeSeries.items() if j in sgid1a])
        
        # Extract charges and spins from the QM-optimized "endpoints".
        Chg0 = [np.sum(QC0.qm_mulliken_charges[i][actives]) for i in range(len(QC0))]
        SpnZ0 = [np.sum(QC0.qm_mulliken_spins[i][actives]) for i in range(len(QC0))]
        Spn20 = [np.sum(QC0.qm_mulliken_spins[i][actives]**2) for i in range(len(QC0))]
    
        Chg1 = [np.sum(QC1.qm_mulliken_charges[i][actives]) for i in range(len(QC1))]
        SpnZ1 = [np.sum(QC1.qm_mulliken_spins[i][actives]) for i in range(len(QC1))]
        Spn21 = [np.sum(QC1.qm_mulliken_spins[i][actives]**2) for i in range(len(QC1))]
    
        # For a reaction subset to qualify, the net charge/spin on the atoms 
        # must be close to an integer and not experience large fluctuations
        # across the whole trajectory
    
        chgi, chgpass = extract_int(np.array(Chg0+Chg1), 0.3, 1.0, label="charge", verbose=False)
        spnzi, spnzpass = extract_int(abs(np.array(SpnZ0+SpnZ1)), 0.3, 1.0, label="spin-z", verbose=False)
        spn2i, spn2pass = extract_int(np.array(Spn20+Spn21), 0.3, 1.0, label="spin^2", verbose=False)
        if chgpass:
            # print "charge fluctuations are small enough; continuing"
            # Try to calculate the correct spin.
            elemi = [j for i, j in enumerate(M2.elem) if i in actives]
            npi = sum([Elements.index(i) for i in elemi])
            nei = npi + chg
            # The number of electrons should be odd iff the spin is odd.
            if ((nei-spnzi)/2)*2 != (nei-spnzi):
                print "electrons (%i) is inconsistent with spin-z (%i); \x1b[93minvalid\x1b[0m" % (nei, spnzi)
            else:
                # print "electrons consistent with spin-z"
                # Build a comment line resembling the nanoreactor script.
                gid0 = set(NanoQC0.TimeSeries.keys()) - set(sgid0a)
                iso0 = [NanoQC0.TimeSeries[i]['iidx'] for i in gid0]
                rx0 = '+'.join([("%i" % iso0.count(i) if iso0.count(i) > 1 else "") + NanoQC0.Isomers[i].ef() for i in sorted(set(iso0))])
                tx0 = str([commadash(sorted([actives.index(i) for i in NanoQC0.TimeSeries[g]['graph'].L()])) for g in gid0]).replace(" ","")
                gid1 = set(NanoQC1.TimeSeries.keys()) - set(sgid1a)
                iso1 = [NanoQC1.TimeSeries[i]['iidx'] for i in gid1]
                rx1 = '+'.join([("%i" % iso1.count(i) if iso1.count(i) > 1 else "") + NanoQC1.Isomers[i].ef() for i in sorted(set(iso1))])
                tx1 = str([commadash(sorted([actives.index(i) for i in NanoQC1.TimeSeries[g]['graph'].L()])) for g in gid1]).replace(" ","")
                evector="%s -> %s" % (rx0, rx1)
                tvector="%s -> %s" % (tx0, tx1)
                QC0.comms = ["Reaction: formula %s atoms %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, Chg0[i], SpnZ0[i], Spn20[i]) for i in range(len(QC0))]
                QC1.comms = ["Reaction: formula %s atoms %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, Chg1[i], SpnZ1[i], Spn21[i]) for i in range(len(QC1))]
                infos.append((actives, chgi, spnzi, spn2i, evector, tvector))
        else:
            print "noninteger charge or fluctuations too large; \x1b[93minvalid\x1b[0m"
    return infos

t0 = time.time()

QCRF = None
RFFrame = 0
QCPI = None
PIFrame = 0
DoIntermediates = True

if DoIntermediates:
    # Attempt to find intermediates in "inter-frames"
    printcool("STEP 4: Exploring the smoothed path for intermediates")
    NanoQCS = []
    QCS = []
    # List of tuples: (starting frame, ending frame, active atoms, integer charge, integer spin, integer spin^2)
    IInfo = []
    savedfns = []
    NanoQCL = None
    QCL = None
    INum = 0
    M2 = Molecule(newpathfnm)
    NanoRef = None
    RefFrm = 0
    IReact = 0
    Optimized = 0
    LastFrames = []
    for iframe in [int(round(i)) for i in np.linspace(0, len(M2), min(len(M2)+1, max(11, float(len(M2))/10)))]:
        M2[iframe].write("tmp_intermediates.xyz")
        NanoQCI0 = Nanoreactor("tmp_intermediates.xyz", enhance=1.2, printlvl=-1, boring=[])
        if NanoEqual(NanoQCI0, NanoQCR):
            print "Skipping frame %i ..." % iframe,
            M2[iframe].write("qcmin_intermediates_%i.xyz" % iframe)
            QCI = deepcopy(QCR)
            QCI_ = Molecule("qcmin_intermediates_%i.xyz" % iframe)
            QCI.xyzs[-1] = QCI_.xyzs[-1]
            QCI.full = 0
        elif (NanoQCL != None) and NanoEqual(NanoQCI0, NanoQCL):
            print "Skipping frame %i ..." % iframe,
            M2[iframe].write("qcmin_intermediates_%i.xyz" % iframe)
            QCI = deepcopy(QCL)
            QCI_ = Molecule("qcmin_intermediates_%i.xyz" % iframe)
            QCI.xyzs[-1] = QCI_.xyzs[-1]
            QCI.full = 0
        elif os.path.exists("qcmin_intermediates_%i.xyz" % iframe):
            print "Reading frame %i ..." % iframe,
            if os.path.exists("qcopt_intermediates_%i.out" % iframe):
                QCI = Molecule("qcopt_intermediates_%i.out" % iframe, errok=['SCF failed to converge', 'Maximum optimization cycles reached'])
            else:
                QCI = deepcopy(QCR)
                QCI_ = Molecule("qcmin_intermediates_%i.xyz" % iframe)
                QCI.xyzs[-1] = QCI_.xyzs[-1]
            QCI.full = 0
        else:
            print "Optimizing frame %i ..." % iframe,
            QCI = QCOptIC(M2, iframe, chg, spn, "qcopt_intermediates_%i" % iframe, nanoref=NanoRef)
            QCI.write("qcopt_intermediates_%i.xyz" % iframe)
            QCI[-1].write("qcmin_intermediates_%i.xyz" % iframe)

        NanoQCI = Nanoreactor("qcmin_intermediates_%i.xyz" % iframe, enhance=1.2, printlvl=-1, boring=[])

        # This is not ideal, as we need to "make the decision" in QCOptIC but we want the printout to be here.
        gix = min(len(QCI)-1, chkcyc)
        QCI[gix].write(".tmp.xyz")
        NanoChk = Nanoreactor(".tmp.xyz", enhance=1.2, printlvl=-1, boring=[])
        if NanoRef != None:
            print "[opt-%i" % gix, "\x1b[91m!=\x1b[0m" if not NanoEqual(NanoRef, NanoChk) else "==", "%i]" % RefFrm,
        else:
            NanoRef = deepcopy(NanoQCI)
            RefFrm = iframe

        Found = False
        Finish = False
        if NanoEqual(NanoQCI, NanoQCR) and (IReact == 0):
            print "same as reactants"
            QCRF = deepcopy(QCI)
            RFFrame = iframe
        elif NanoEqual(NanoQCI, NanoQCP):
            print "Converted to \x1b[94mproducts (%i)\x1b[0m" % len(NanoQCS)
            QCPI = deepcopy(QCI)
            PIFrame = iframe
            if QCL != None and not NanoEqual(NanoQCL, NanoQCR):
                Found = True
            Finish = True
        elif len(QCS) > 0 and NanoEqual(NanoQCI, NanoQCL):
            print "same as current intermediate"
        elif QCL != None:
            Found = True
        else:
            print "Initial point appears to have converged to a different state."
        if Found:
            IReact = 1
            print "Examining"
            infos = GetIntermediates(QCL, NanoQCL, QCI, NanoQCI)
            for info in infos:
                Write = True
                if any([((NanoEqual(NanoQCL, NanoQCS[sn][0])) and (NanoEqual(NanoQCI, NanoQCS[sn][1]))) for sn in range(len(NanoQCS))]):
                    print "Found a \x1b[93mduplicate pathway\x1b[0m"
                    Write = False
                elif any([((NanoEqual(NanoQCL, NanoQCS[sn][1])) and (NanoEqual(NanoQCI, NanoQCS[sn][0]))) for sn in range(len(NanoQCS))]):
                    print "Found a \x1b[93mreverse pathway\x1b[0m"
                    Write = False
                elif ((any([NanoEqual(NanoQCL, NanoQCS[sn][0]) for sn in range(len(NanoQCS))]) and any([NanoEqual(NanoQCI, NanoQCS[sn][1]) for sn in range(len(NanoQCS))])) or 
                      (any([NanoEqual(NanoQCL, NanoQCS[sn][1]) for sn in range(len(NanoQCS))]) and any([NanoEqual(NanoQCI, NanoQCS[sn][0]) for sn in range(len(NanoQCS))]))):
                    print "Found a new pathway for \x1b[94mexisting intermediates (%i)\x1b[0m" % len(NanoQCS)
                elif (not Finish):
                        print "Found a \x1b[94mnew intermediate (%i)\x1b[0m" % len(NanoQCS)

                # Don't save the segment if it's the entire path
                if (NanoEqual(NanoQCL, NanoQCR) and NanoEqual(NanoQCI, NanoQCP)):
                    Write = False

                if Write:
                    # Append the list of saved nanoreactor objects
                    NanoQCS.append((deepcopy(NanoQCL), deepcopy(NanoQCI)))
                    QCS.append((deepcopy(QCL), deepcopy(QCI)))
                    IInfo.append((LastFrame, iframe, info[0], info[1], info[2], info[3], info[4], info[5]))
                    # Create objects to be saved to disk
                    if QCL.full:
                        QC0 = deepcopy(QCL)
                        NanoQC0 = deepcopy(NanoQCL)
                    else:
                        for last in LastFrames[::-1]:
                            # Now we optimize the previous frames and break when the optimized structure is equal to 
                            # the (known) previous structure.
                            print "reoptimizing frame %i .." % last,
                            QC0 = QCOptIC(M2, last, chg, spn, "qcopt_intermediates_%i" % last)
                            QC0.write("qcopt_intermediates_%i.xyz" % last)
                            QC0[-1].write("qcmin_intermediates_%i.xyz" % last)
                            NanoQC0 = Nanoreactor("qcmin_intermediates_%i.xyz" % last, enhance=1.2, printlvl=-1, boring=[])
                            if NanoEqual(NanoQC0, NanoRef):
                                print "converged to previous intermediate"
                                break
                            print
                    QC1 = deepcopy(QCI)
                    NanoQC1 = deepcopy(NanoQCI)
                    actives, chgi, spnzi, spn2i, evector, tvector = info
                    for i in QC0.Data.keys():
                        if 'qm' in i or 'qc' in i:
                            del QC0.Data[i]
                            del QC1.Data[i]
                    Mid = M2.atom_select(actives)[LastFrame+1:iframe]
                    Mid.comms = ["Reaction: formula %s atoms %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, chgi, spnzi, spn2i) for i in range(len(Mid))]
                    Segment = QC0.atom_select(actives)[::-1] + Mid + QC1.atom_select(actives)
                    if "_split" in basefn:
                        splitfn = basefn + "-%i" % INum + ".xyz"
                    else:
                        splitfn = basefn + "_split%i" % INum + ".xyz"
                    InterpolatePath(Segment, splitfn, outfnm=splitfn, verbose=False)
                    print "Wrote segment to %s, length %i(QM) + %i(MD) + %i(QM) = %i" % (splitfn, len(QC0), len(Mid), len(QC1), len(Segment))
                    INum += 1
            NanoRef = deepcopy(NanoQCI)
            RefFrm = iframe

        # Store the optimized molecule from the previous frame.
        NanoQCL = deepcopy(NanoQCI)
        QCL = deepcopy(QCI)
        LastFrame = iframe
        LastFrames.append(iframe)
        if Finish: break

print

print "Intermediate-finding walltime: %.1f s" % (time.time() - t0)

printcool("STEP 5: Minimizing length of the original path")

if QCRF == None: QCRF = deepcopy(QCR1)
if QCPI == None: QCPI = deepcopy(QCP1)

for i in QCRF.Data.keys():
    if any([j in i for j in ['qm', 'qc', 'charge', 'mult']]):
        del QCRF.Data[i]

for i in QCPI.Data.keys():
    if any([j in i for j in ['qm', 'qc', 'charge', 'mult']]):
        del QCPI.Data[i]

if RFFrame == PIFrame:
    RFFrame -= 1
    PIFrame += 1

QCRF.comms = [M2.comms[RFFrame+1] for i in range(len(QCRF))]
QCPI.comms = [M2.comms[PIFrame] for i in range(len(QCPI))]

RxnPath2 = QCRF[::-1] + M2[RFFrame+1:PIFrame] + QCPI
print "Trimming the original path, length %i(old) -> %i(new) = %i(QM) + %i(MD) + %i(QM)" % (len(RxnPath), len(RxnPath2), len(QCRF), PIFrame-RFFrame-1, len(QCPI))
basefn = re.sub("_stage[0-9]","",os.path.splitext(os.path.split(sys.argv[1])[-1])[0])
newpathfnm = InterpolatePath(RxnPath2, basefn)

print "Second Interpolation walltime: %.1f s" % (time.time() - t0)

# The trajectory should now be broken down into parts, now write them.
# if len(QCS) > 1:
#     for INum in range(len(QCS)):
#         QC0 = deepcopy(QCS[INum][0])
#         QC1 = deepcopy(QCS[INum][1])
#         NanoQC0 = deepcopy(NanoQCS[INum][0])
#         NanoQC1 = deepcopy(NanoQCS[INum][1])
#         LastFrame, iframe, actives, chgi, spnzi, spn2i, evector, tvector, hcreate = IInfo[INum]
#         for i in QC0.Data.keys():
#             if 'qm' in i or 'qc' in i:
#                 del QC0.Data[i]
#                 del QC1.Data[i]
#         Mid = M2.atom_select(actives)[LastFrame+1:iframe]
#         Mid.comms = ["Reaction: formula %s atoms %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, chgi, spnzi, spn2i) for i in range(len(Mid))]
#         Segment = QC0.atom_select(actives)[::-1] + Mid + QC1.atom_select(actives)
#         if "_split" in basefn:
#             splitfn = basefn + "-%i" % INum + ".xyz"
#         else:
#             splitfn = basefn + "_split%i" % INum + ".xyz"
#         InterpolatePath(Segment, splitfn, outfnm=splitfn, verbose=False)
#         print "Wrote segment to %s, length %i(QM) + %i(MD) + %i(QM) = %i" % (splitfn, len(QC0), len(Mid), len(QC1), len(Segment))
#         INum += 1

got = []
for i, j in zip(finalfns, nebfns):
    if (i, j) not in got:
        print "Output path has been saved to %s and decimated path to %s" % (i, j)
    got.append((i, j))

printcool("Path analysis is finished!%s" % (" Found %i total segments" % (INum) if INum > 1 else ""))
tarexit()

