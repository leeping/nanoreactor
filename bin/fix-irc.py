#!/usr/bin/env python

from nanoreactor.molecule import Molecule, arc
from nanoreactor.qchem import erroks, SpaceIRC
from nanoreactor.nifty import monotonic_decreasing, extract_tar, bak
from copy import deepcopy
import os, sys, shutil, re
import numpy as np
from collections import OrderedDict

"""
Recover from errors in IRC processing.  This is a temporary fix until
I can figure out what's wrong with determining the TS-frame in
qchem.py .
"""

if os.path.exists('transition-state.tar.bz2'):
    fnm = 'transition-state.tar.bz2'
elif os.path.exists('freezing-string.tar.bz2'):
    fnm = 'freezing-string.tar.bz2'
else:
    raise RuntimeError('No archive')

if not os.path.exists('fix-TS'):
    os.makedirs('fix-TS')
os.chdir('fix-TS')
os.system('tar xjf ../%s' % fnm)

icalc = 0
iTS = 0
M = None
began = False
while True:
    qcouts = [i.strip() for i in os.popen('ls qcirc.%02i*out -tr 2> /dev/null' % icalc).readlines()]
    if len(qcouts) == 0: break
    qcout = qcouts[-1]
    jobtype = qcout.split('.')[-2]
    # Load the Molecule object
    M_ = Molecule(qcout, errok = erroks[jobtype] + ['SCF failed to converge', 'Maximum optimization cycles reached'])
    # Delete extraneous keys that aren't the same
    del M_.Data['qctemplate']
    del M_.Data['qcrems']
    del M_.Data['qcerr']
    del M_.Data['topology']
    del M_.Data['molecules']
    # Add the TS frame onto the beginning...?
    M_.align()
    rmsd = M_.ref_rmsd(0)
    if np.min(rmsd[1:]) < 1e-10:
        print qcout, "returned to the beginning"
        fret = np.argmin(rmsd[1:])+1
        # # We shouldn't trust the second segment.
        if M != None:
            M_ = M_[:fret]
            if began:
                if iTS > 0:
                    M = M[:iTS] + M_
                else:
                    M += M_
            else:
                M = M_[::-1] + M
                iTS = fret
            print "Transition state is frame", iTS
        else:
            M = M_[:fret][::-1] + M_[fret:]
            iTS = fret
            print "Transition state is frame", iTS
        began = True
    else:
        # print qcout, "did not return to the beginning"
        if M == None: M = deepcopy(M_)
        else:
            shift = 0
            while True:
                JoinFwd = M[-1-shift] + M_
                RmsdFwd = JoinFwd.ref_rmsd(0)
                JoinBak = M[0+shift] + M_
                RmsdBak = JoinBak.ref_rmsd(0)
                print shift, RmsdFwd[1], RmsdBak[1]
                if RmsdFwd[1] < 1e-4:
                    print qcout, "will be joined onto the end"
                    if shift: M = M[:-shift]
                    M += M_
                    break
                elif RmsdBak[1] < 1e-4:
                    print qcout, "will be joined onto the front"
                    if shift: M = M[shift:]
                    M = M_[::-1] + M
                    iTS += len(M_)
                    iTS -= shift
                    print "Transition state is frame", iTS
                    break
                shift += 1
                if shift >= 10:
                    raise RuntimeError('No idea what to do with %s' % qcout)
    icalc += 1

def GetRMSD(m0, m1):
    m = Molecule()
    m.elem = m0.elem
    m.xyzs = m0.xyzs
    m.xyzs += m1.xyzs
    return m.ref_rmsd(0)[-1]

S = Molecule('irc.xyz', ftype='xyz')
RMSD1 = GetRMSD(S[0], M[0]) + GetRMSD(S[-1], M[-1])
RMSD2 = GetRMSD(S[0], M[-1]) + GetRMSD(S[-1], M[0])
print "IRC RMSD to initial path endpoints (fwd, bkwd) = %6.3f %6.3f" % (RMSD1, RMSD2)
fwd = (RMSD1 < RMSD2)
if not fwd: 
    M = M[::-1]
    iTS = len(M) - iTS

# Energies in kcal/mol.
E = M.qm_energies
E -= E[0]
E *= 627.51

# Write IRC energies to comments.
M.comms = ["Intrinsic Reaction Coordinate: Energy = % .4f kcal/mol" % i for i in E]
M.comms[iTS] += " (Transition State)"

# Eliminate geometry optimization frames that go up in energy.
selct = np.concatenate((monotonic_decreasing(E, iTS, 0, verbose=True)[::-1], monotonic_decreasing(E, iTS, len(M), verbose=True)[1:]))
M = M[selct]
E = E[selct]

# Save the IRC energy as a function of arc length.
ArcMol = arc(M, RMSD=True)
ArcMolCumul = np.insert(np.cumsum(ArcMol), 0, 0.0)
np.savetxt("irc.nrg", np.hstack((ArcMolCumul.reshape(-1, 1), E.reshape(-1,1))), fmt="% 14.6f", header="Arclength(Ang) Energy(kcal/mol)")

MOld = Molecule("irc.xyz")
ArcOld = arc(MOld, RMSD=True)
if len(M) > len(MOld):
    print "Repair was \x1b[1;92msuccessful\x1b[0m (RMSDMax: %.3f -> %.3f)" % (np.max(ArcOld), np.max(ArcMol))
    replace = True
else:
    print "Repair was unnecessary (RMSDMax: %.3f -> %.3f)" % (np.max(ArcOld), np.max(ArcMol))
    replace = False
    
M.align_center()
M.write("irc.xyz")
M.get_populations().write('irc.pop', ftype='xyz')

M_EV = SpaceIRC(M, E, RMSD=True)
M_EV.write("irc_spaced.xyz")

if replace: os.system('tar cjf %s * --remove-files' % fnm)

os.chdir('..')

if replace: 
    print "Replacing results with fixed version."
    bak(fnm)
    shutil.move('fix-TS/'+fnm, fnm)
    extract_tar(fnm, ['irc.nrg', 'irc.pop', 'irc.xyz', 'irc_spaced.xyz'], force=True)
    os.system('draw-reaction.py')
    
shutil.rmtree('fix-TS')

