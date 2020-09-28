#!/usr/bin/env python

#===========#
#| Imports |#
#===========#

from __future__ import print_function
from nebterpolator.io import XYZFile
from nebterpolator.path_operations import smooth_internal, smooth_cartesian, equal_velocity
import os, sys, argparse, shutil
import numpy as np
import traceback

#===========#
#| Globals |#
#===========#

# Recommended settings: --morse 1e-3 --repulsive --allpairs --anchor 2
parser = argparse.ArgumentParser()
parser.add_argument('--morse', type=float, default=0.0, help='Weight of Morse potential')
parser.add_argument('--repulsive', action='store_true', help='Purely repulsive Morse potential (i.e. no "one minus")')
parser.add_argument('--width', type=int, default=-1, help='Smoothing width, default is to use the trajectory length')
parser.add_argument('--window', type=str, default='hanning', help='Type of window for smoothing: choose flat, hanning, hamming, bartlett, blackman')
parser.add_argument('--anchor', type=float, default=-1, help='Anchor (1 is strongest), default is no anchor')
parser.add_argument('--allpairs', action='store_true', help='Use all interatomic distances, instead of just the bonds')
args, sys.argv= parser.parse_known_args(sys.argv)

input_filename = sys.argv[1]
if len(sys.argv) > 2:
    output_filename = sys.argv[2]
else:
    output_filename = os.path.splitext(sys.argv[1])[0]+"_out.xyz"
    print("Supplying default output filename", output_filename)
nm_in_angstrom = 0.1

#-----
# These two parameters are adjustable, and depend on the length of the traj
#-----
# Cutoff period for the internal coordinate smoother. motions with a shorter
# period than this (higher frequency) will get filtered out
smoothing_width = args.width
# Decide whether to perform a final smoothing step in Cartesian coordinates.
FinalSmooth = True
xyz_smoothing_strength = 2.0
if not FinalSmooth:
    xyz_smoothing_strength = 0.0

#==========#
#| Script |#
#==========#

xyzlist, atom_names = None, None
with XYZFile(input_filename) as f:
    xyzlist, atom_names = f.read_trajectory()
    # angstroms to nm
    xyzlist *= nm_in_angstrom
if xyzlist.shape[1] < 4:
    print("Interpolator cannot handle less than four atoms - copying input to output instead.")
    shutil.copy2(input_filename, output_filename)
    sys.exit()

# Equalize distances between frames (distance metric is max displacement.)
xyzlist = equal_velocity(xyzlist)

#-----
# Determine a reasonable smoothing width.  It must be an odd number
# and smaller than the trajectory length.
#-----
if smoothing_width == -1 or smoothing_width > len(xyzlist):
    smoothing_width = len(xyzlist)
    smoothing_width += smoothing_width%2
    smoothing_width -= 1
if smoothing_width%2 != 1:
    smoothing_width -= 1
    print("Smoothing width must be an odd number - changing to %i" % smoothing_width)

#-----
# Transform into redundant internal coordinates, apply a Fourier based
# smoothing, and then transform back to Cartesian.
# 
# The internal -> cartesian bit is the hard step, since there's no
# guarantee that a set of cartesian coordinates even exist that
# satisfy the redundant internal coordinates, after smoothing.
# 
# We use a Levenberg-Marquardt least squares minimization to find the "most consistent"
# cartesian coordinates.  The least squares error vector includes the differences between 
# - The internal coordinates generated from the Cartesian coordinates being optimized, and
# - The smoothed internal coordinates (which we want to reproduce).
# 
# The internal coordinates contains the pairwise distances between bonded atoms, 
# all of the angles between sets of three atoms, a-b-c, that actually get "bonded" 
# during the trajectory, and all of the dihedral angles between sets of 4 atoms, 
# a-b-c-d, that actually get "bonded" during the trajectory.  If this script is 
# called with --all-pairs, we're using ALL pairwise distances, but the angles / 
# dihedrals are still computed only for triplets and quartets of bonded atoms.
#-----
# A pathological case is when the path does not correctly go to the final point (when we use anchoring).
# When this happens, find the path that goes backward and "patch" it with the forward path if they
# intersect.  If the forward and backward paths never intersect, then decrease the smoothing width.
# This is repeated until a consistent path is found.
#-----
def try_smooth_internal(*args, **kwargs):
    try:
        return smooth_internal(*args, **kwargs)
    except:
        traceback.print_exc()
        print("Interpolator failed due to error - copying input to output instead.")
        shutil.copy2(input_filename, output_filename)
        sys.exit()
    
while True: 
    smoothed, errors = try_smooth_internal(xyzlist, atom_names, width=smoothing_width, bond_width=smoothing_width, angle_width = smoothing_width, 
                                       dihedral_width = smoothing_width, allpairs=args.allpairs, w_morse = args.morse, rep=args.repulsive, 
                                       anchor=args.anchor, window=args.window)
    if errors[-1] > 1e-3:
        print("\x1b[1;91mRedoing reverse path!\x1b[0m (errors[-1]=%f)" % errors[-1])
        smoothed_, errors_ = try_smooth_internal(xyzlist[::-1], atom_names, width=smoothing_width, bond_width=smoothing_width, angle_width = smoothing_width, 
                                             dihedral_width = smoothing_width, allpairs=args.allpairs, w_morse = args.morse, rep=args.repulsive, 
                                             anchor=args.anchor, window=args.window, xyzlist_match = smoothed[::-1])
        if errors_[-1] > 1e-3:
            smoothing_width = int(smoothing_width*0.5)
            if smoothing_width%2 == 0:
                smoothing_width += 1
            print("\x1b[1;91mDecreasing smoothing width to %i!\x1b[0m" % smoothing_width)
        else:
            smoothed = smoothed_[::-1]
            print("\x1b[92mFinished (reverse)\x1b[0m")
            break
    else: 
        print("\x1b[92mFinished (forward)\x1b[0m")
        break

print('Saving output to', output_filename)

#-----
# The cartesian smoothing step optionally runs after the internal coordinates 
# smoother. The point of this is ONLY to correct for "jitters" in the 
# xyz coordinates that are introduced by imperfections in the 
# redundant internal coordinate -> xyz coordinate step
#-----
jitter_free = smooth_cartesian(smoothed,
                               strength=xyz_smoothing_strength,
                               weights=1.0/errors)

# Equalize distances between frames (distance metric is max displacement.)
Equalize = True
if Equalize:
    equalized_v = equal_velocity(jitter_free)
    with XYZFile(output_filename, 'w') as f:
        f.write_trajectory(equalized_v / nm_in_angstrom, atom_names)
else:
    with XYZFile(output_filename, 'w') as f:
        f.write_trajectory(jitter_free / nm_in_angstrom, atom_names)

