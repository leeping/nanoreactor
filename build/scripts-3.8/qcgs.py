#!/home/hpark21/miniconda3/envs/python3/bin/python

""" 
qcgs.py

Python script for calling Q-Chem via growing string. The name of
this script is actually hardcoded into gstring.exe. 

This is not too complicated, I hope - this script, which pretends to
be the Q-Chem executable, creates a QChem() object and then calls
Q-Chem that way.

The motivation is that I want to prevent HF/KS unstable states from
entering the growing string calculation.  However, these states
are quite rare, and the stability analysis is expensive.  So my
compromise solution is to do a stability analysis every 10 string
iterations, or for any image that was unstable for the previous
string iteration.

Also, this script ensures that every image in the growing string calculation
uses the Q-Chem 
"""

import os, sys
from nanoreactor.qchem import QChem, tarexit
tarexit.tarfnm = "growing-string.tar.bz2"

QC = QChem(sys.argv[1], qcout=sys.argv[2], qcdir=sys.argv[3], qcsave=True, readsave=True)

# File that records how many calculations were done
# for this image.
calcfnm = os.path.splitext(sys.argv[1])[0]+".num"
calcnum = -1
if os.path.exists(calcfnm):
    calcnum = int(open(calcfnm).readlines()[0].strip())
calcnum += 1
with open(calcfnm,'w') as f: print >> f, calcnum

# File that records how many iterations ago we had to do a stability
# analysis.  "# of days since last accident" kind of thang.
stabfnm = "qchem.nsb"
stablast = int(open(stabfnm).readlines()[0].strip()) if os.path.exists(stabfnm) else 0

# Do stability analysis every 5 string iterations or if any calculation
# needed stability corrections in the previous 30 iterations.
QC.nstab = 1
if os.path.exists('Do_Stability_Analysis'):
    if calcnum%10 == 0 or stablast < 30:
        QC.make_stable()
QC.force()

# If unstable, then reset accident counter to zero.
with open(stabfnm, 'w') as f:
    if QC.nstab > 1:
        print >> f, 0
    else:
        print >> f, stablast+1
