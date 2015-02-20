#!/usr/bin/env python

import os, sys, time
import numpy as np
import argparse
from collections import OrderedDict
from nanoreactor.output import logger
# Utility functions and classes used to be a part of this script
from nanoreactor.rxndb import create_work_queue, parse_input_files, get_trajectory_home, Trajectory, wq_reactor

#===========================================#
#| Nanoreactor refinement script version 2 |#
#|        Author: Lee-Ping Wang            |#
#===========================================#

def parse_command():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', help='Input: reaction.xyz trajectory files,' 
                        'list of file names, or list of directories containing these files')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='Set higher numbers to have more printout. Roughly speaking: 0: Be very quiet.' 
                        '1: Show essential reaction-level information and pathway results. 2: Show optimization-level and pathway-level information.' 
                        '3: Show internal workings (i.e. job status and system calls.)')
    parser.add_argument('--read_only', action='store_true', help="Read calculation status but don't actually run any calculations.")
    parser.add_argument('--fast_restart', action='store_true', help="Read calculation status from disk and skip over completed / failed calculations; faster but less careful.")
    parser.add_argument('--small_first', action='store_true', help="Do the smallest trajectories (i.e. fewest atoms) first")
    parser.add_argument('--subsample', type=int, default=10, help='Frame interval for subsampling trajectories')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port number for the Work Queue master; leave blank to run locally')
    parser.add_argument('--methods', type=str, nargs='+', default=['b3lyp'], help='Which electronic structure method to use. ' 
                        'Provide 2 names if you want the final TS refinement + IRC to use a different method.')
    parser.add_argument('--bases', type=str, nargs='+', default=['6-31g(d)', '6-31+g(d,p)'], help='Which basis set to use. '
                        'Provide 2 names if you want the final TS refinement + IRC to use a different basis set.')
    parser.add_argument('--draw', type=int, default=2, help='Choose the drawing level for summary PDFs. 0: Draw nothing. '
                        '1: Draw correct reactions only. 2: Draw all reactions that do anything. 3: Redraw all reactions.')
    parser.add_argument('--images', type=int, default=21, help='Number of images along the path in a growing string or NEB calculation')
    parser.add_argument('--gsmax', type=int, default=600, help='Maximum number of growing string cycles prior to termination')
    parser.add_argument('--pathmax', type=int, default=1000, help='Maximum length of pathways to build')
    parser.add_argument('--dynmax', type=int, default=2000, help='Maximum length of dynamics trajectory to consider')
    parser.add_argument('--atomax', type=int, default=50, help='Maximum number of atoms to consider')
    parser.add_argument('--trivial', action='store_true', help='Include energy refinement of trivial rearrangements')
    parser.add_argument('--spectators', action='store_true', help='Keep spectators as part of the reaction')
    parser.add_argument('--ts_branch', action='store_true', help='Transition state calculations branch off growing string segments and run in parallel; faster but less efficient')
    args, sys.argv = parser.parse_known_args(sys.argv[1:])
    return args

def main():
    # global WQ
    # Get command line arguments.
    args = parse_command()
    if args.draw == 3: args.fast_restart = False
    # Set verbosity level
    logger.set_verbosity(args.verbose)
    # Create the Work Queue.
    if args.port != 0:
        create_work_queue(args.port)
        # print "Created WQ"
        # print WQ
        # sys.exit()
    # Obtain a list of dynamics trajectory files.
    trajectory_fnms = parse_input_files(args.input)
    # We have the options of doing the smallest trajectories first.
    if args.small_first:
        order = list(np.argsort([int(open(fnm).next().split()[0]) for fnm in trajectory_fnms]))
    else:
        order = range(len(trajectory_fnms))
    # Execute calculations.
    Trajectories = OrderedDict()
    t0 = time.time()
    for i, ixyz in enumerate(order):
        xyz = trajectory_fnms[ixyz]
        xyzhome = get_trajectory_home(xyz)
        xyzname = xyzhome.replace(os.getcwd(), '').strip('/')
        Trajectories[xyzname] = Trajectory(xyz, xyzhome, name=xyzname, methods=args.methods, bases=args.bases, 
                                           subsample=args.subsample, fast_restart=args.fast_restart, read_only=args.read_only,
                                           verbose=args.verbose, images=args.images, dynmax=args.dynmax, atomax=args.atomax, 
                                           pathmax=args.pathmax, priority=10*(len(trajectory_fnms)-i), draw=args.draw, 
                                           gsmax=args.gsmax, trivial=args.trivial, ts_branch=args.ts_branch, spectators=args.spectators)
        Trajectories[xyzname].launch()
        # Enter the reactor loop once in a while so we don't waste
        # time during the setup phase.
        if (args.port != 0) and ((time.time()-t0) > 60):
            wq_reactor(wait_time=1, iters=10)
            t0 = time.time()
    # Enter the reactor loop.
    if args.port != 0:
        wq_reactor()

if __name__ == "__main__":
    main()
