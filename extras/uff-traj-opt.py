#!/usr/bin/env python
from __future__ import print_function
from nanoreactor import Nanoreactor
from nanoreactor.molecule import *
from nanoreactor import contact
import itertools
import networkx as nx
import numpy as np
import os, sys

#======================================================================#
#|                                                                    |#
#|                UFF trajectory optimization script                  |#
#|                                                                    |#
#|                Lee-Ping Wang (leeping@stanford.edu)                |#
#|                  Last updated October 13, 2012                     |#
#|                                                                    |#
#|               [ IN PROGRESS, USE AT YOUR OWN RISK ]                |#
#|                                                                    |#
#|                            Purpose                                 |#
#|                                                                    |#
#|   Given a trajectory of a single molecule, optimize up to 100      |#
#|   frames along the trajectory (from 10% to 90%) using the UFF      |#
#|   force field and save the lowest energy structure.                |#
#|                                                                    |#
#|   Topologically disconnected structures are penalized              |#
#|   (i.e. favors single molecules.)                                  |#
#|                                                                    |#
#|   Useful for generating geometry-optimized reaction products from  |#
#|   LearnReactions.py; use extract_###.xyz as the input argument.    |#
#|                                                                    |#
#|                   Required: Turbomole 5.10                         |#
#|                                                                    |#
#|             Instructions:                                          |#
#|                                                                    |#
#|           Make sure Turbomole is properly set up                   |#
#|           i.e. your PATH and TURBODIR environment variables        |#
#|           Run ./uff-traj-opt.py input.xyz                          |#
#|                                                                    |#
#======================================================================#

definetxt="""

a test.coord
ff
m 300

*
no
*
*
n
q
"""

tmpfiles=["basis", "coord", "fapprox", "new.xyz", "test.coord", "test.xyz", "*uff*"]

assert 'turbomole' in os.environ['PATH'], 'Turbomole (5.10) needs to be installed in order for this script to work'

M = Molecule(sys.argv[1])
# Choose <=100 geometries from 10% to 90% of the trajectory length.
first = int(len(M) * 0.1)
last = int(len(M) * 0.9)
if last - first <= 100:
    frames = list(range(first, last))
else:
    frames = [int(i) for i in np.linspace(first, last, 100)]

os.system("rm -rf uff_temp")
os.system("mkdir -p uff_temp")
os.chdir("uff_temp")
Eopt = []
Tops = []
Mopt = None
with open("define.in","w") as d: print(definetxt, file=d)
print("Using Turbomole to optimize 100 geometries from frames:")
print(frames)
for fi, f in enumerate(frames):
    for t in tmpfiles:
        os.system("rm -rf %s" % t)
    if (fi+1) % 10 == 0:
        print("Working on frame %i" % (fi+1))
    M.write("test.xyz",select=f)
    os.system("x2t test.xyz > test.coord")
    E = []
    for line in os.popen("define < define.in 2> /dev/null").readlines():
        if 'UFF energy' in line:
            E.append(float(line.split()[3].replace('D','e')))
    os.system("rm control")
    os.system("t2x coord > new.xyz 2> /dev/null")
    ufftop = open("ufftopology").readlines()
    topstr=""
    for ln, line in enumerate(ufftop):
        if "connectivity (= bond)" in line:
            nb = int(ufftop[ln+1].strip())
            topstr += ' '.join([' '.join([str(int(float(ufftop[i].split()[j]))) for j in [0,1,3]]) for i in range(ln+2, ln+2+nb)])
        if "angle" in line:
            na = int(ufftop[ln+1].strip())
            topstr += ' '
            topstr += ' '.join([' '.join([str(int(float(ufftop[i].split()[j]))) for j in [0,1,2,3,5,6]]) for i in range(ln+2, ln+2+na)])
    Tops.append(topstr)
    Eopt.append(E[-1])
    if Mopt == None:
        Mopt = Molecule("new.xyz")
    else:
        Mnew = Molecule("new.xyz")
        Mopt += Mnew
Mopt.write("opt_traj.xyz")
Mreact = Nanoreactor("opt_traj.xyz", enhance=1.2)
os.chdir("..")
TopSet = list(set(Tops))
TopCount = np.array([Tops.count(Top) for Top in TopSet])
print(len(TopCount))
# Penalize uncommon and disconnected topologies
Eopt = np.array(Eopt) + 1e10*(np.array(Mreact.NumMols)-1) + 1e6*(np.array(Tops) != TopSet[np.argmax(TopCount)])
print("Energies:")
print(Eopt)
print("Minimum Energy:", min(Eopt))
print("Geometry with Minimum Energy:", np.argmin(Eopt))
fout = os.path.splitext(sys.argv[1])[0]+"_uff.xyz"
Mopt.write(fout,select=np.argmin(Eopt))
print("Optimized geometry written to", fout)
