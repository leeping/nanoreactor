#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore")
from nanoreactor import contact
from nanoreactor.molecule import *
from nanoreactor.nanoreactor import *
from nanoreactor.nifty import _exec, printcool_dictionary, printcool
from scipy import optimize
import time
import math
import os, sys, re
import itertools
import traceback
from copy import deepcopy
from collections import OrderedDict, Counter

# Delete the result from previous jobs
_exec("rm -rf neb_result.tar.bz2") 

# Adjustable Settings
tcin = """
basis           6-31g*
method          uhf
charge          {chg}
spinmult        {spn}
nstep           4000
threall         1.0e-12
run             ts
scf             diis+a
maxit           50
min_nebk        1
min_coordinates cartesian
ts_method       neb_frozen
min_image       11
timebomb        off
coordinates     initial.xyz
end
"""

# Determine the charge and spin from the xyz comment line.

M = Molecule(sys.argv[1])
print M.comms[0].split("atoms")[0]

# Read in the charge and spin on the whole system.
srch  = lambda s : np.array([float(re.search('(?<=%s )[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?' % s, c).group(0)) for c in M.comms])
Chgs  = srch('charge') # An array of the net charge.
SpnZs = srch('sz')    # An array of the net Z-spin.
Spn2s = srch('sz\^2') # An array of the sum of sz^2 by atom.

def extract_int(arr, avgthre, limthre, label="value", verbose=True):
    """ Get the representative integer value from an array.
    Sanity check: Make sure the value does not go through big excursions.
    The integer value is the rounded value of the average.
    thresh = A threshold to make sure we're not dealing with 
    fluctuations that are too large. """
    average = np.mean(arr)
    maximum = np.max(arr)
    minimum = np.min(arr)
    rounded = round(average)
    passed = True
    if abs(average - rounded) > avgthre:
        if verbose: print "Average %s (%f) deviates from integer %s (%i) by more than threshold of %f" % (label, average, label, rounded, avgthre)
        passed = False                                                                                        
    if abs(maximum - minimum) > limthre:
        if verbose: print "Maximum %s fluctuation (%f) is larger than threshold of %f" % (label, abs(maximum-minimum), limthre)
        passed = False
    return int(rounded), passed

chg, chgpass = extract_int(Chgs, 0.3, 1.0, label="charge")
spn, spnpass = extract_int(abs(SpnZs), 0.3, 1.0, label="spin-z")

# Try to calculate the correct spin.
nproton = sum([Elements.index(i) for i in M.elem])
nelectron = nproton + chg

if not spnpass:
    print "Going with the minimum spin consistent with charge."
    if nelectron%2 == 0:
        spn = 0
    else:
        spn = 1

# The number of electrons should be odd iff the spin is odd.
if ((nelectron-spn)/2)*2 != (nelectron-spn):
    print "\x1b[91mThe number of electrons (%i) is inconsistent with the spin-z (%i)\x1b[0m" % (nelectron, spn)
    print "Exiting."
    sys.exit()

print "Number of electrons:", nelectron
print "Net charge:", chg
print "Net spin:", spn

with open("neb.in",'w') as f: print >> f, tcin.format(chg=str(chg), spn=str(spn+1))

_exec("int neb.in", print_command=True, print_to_screen=True, persist=True)
_exec("mv neb.log neb.log1")
_exec("tar cvjf neb_result.tar.bz2 scr /dev/null --remove-files", persist=True)
_exec("mv neb.log1 neb.log")
