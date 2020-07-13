#!/usr/bin/env python

from __future__ import print_function
import os, sys, argparse
from nanoreactor import Nanoreactor

#==========================#
#     Parse arguments.     #
#==========================#

# Taken from MSMBulder - it allows for easy addition of arguments and allows "-h" for help.
def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d
    group.add_argument(*args, **kwargs)

parser = argparse.ArgumentParser()
add_argument(parser, 'xyzin', metavar='input.xyz', help='Input coordinate file \x1b[1;91m(Required)\x1b[0m', type=str)
add_argument(parser, '-q', dest='qsin', help='xyz formatted file with charges on x-coordinate and spins on y-coordinate',
             default='charge-spin.txt', type=str)
add_argument(parser, '-B', dest='boin', help='File containing pairwise bond orders',
             default='bond_order.list', type=str)
add_argument(parser, '-T', dest='bothre', help='Bond order threshold (set nonzero to use)',
             default=0.0, type=float)
add_argument(parser, '-e', dest='enhance', help='Enhancement factor for bond detection; larger number = more bonds. For hydroxide we used 1.25 * (1.2 / 1.2125).',
             default=1.4, type=float)
add_argument(parser, '-m', dest='mindist', help='Below this distance (in Angstrom) all atoms are considered to be bonded.',
             default=1.0, type=float)
add_argument(parser, '-s', dest='dt_fs', help='Provide time step in femtoseconds (only used if properties.txt does not exist)',
             default=0.0, type=float)
add_argument(parser, '-p', dest='printlvl', help='Print level.  1: Print out unrectified time series. 2: Print out empirical formulas as they are discovered.',
             default=0, type=int)
add_argument(parser, '-c', dest='cutoff', help='Cutoff frequency for lowpass filter in cm^-1. 100 cm^-1 is equivalent to 333.6 fs vibrational period. Pass zero to skip lowpass filtering.',
             default=100.0, type=float)
add_argument(parser, '-t', dest='learntime', help='Molecules that exist for at least this number of *femtoseconds* are recognized as stable in reaction event detection.',
             default=100.0, type=float)
add_argument(parser, '-k', dest='known', nargs='+', help='Known empirical formulas not to be colored (type All to include all molecules in the first frame)',
             default=['all'], type=str)
add_argument(parser, '-E', dest='exclude', nargs='+', help='Empirical formulas to be excluded from all reaction events',
             default=[], type=str)
add_argument(parser, '-M', '--save_molecules', help='Extract stable molecules as well as reactions', action='store_true')
add_argument(parser, '-N', '--neutralize', help='Extract nearby molecules to neutralize the system', action='store_true')
add_argument(parser, '--pbc', help='Simple periodic boundary support, specify cubic NVT box size in Angstrom.', default=0, type=float)
add_argument(parser, '--align', action='store_true', help='Align molecules and reactions prior to output.')
add_argument(parser, '--radii', type=str, nargs="+", default=["Na","0.0","K","0.0"], help='Custom atomic radii for bond detection.')
add_argument(parser, '--plot', action='store_true', help='Save interatomic distance or bond order time series to files.')

print("LearnReactions.py called with the following arguments:")
print(' '.join(sys.argv))
print("\n#=========================================#")
print("#       Reactive MD learning script       #")
print("#  Use the -h argument for detailed help  #")
print("#=========================================#\n")
args = parser.parse_args()
    
def main():
    if not os.path.exists(args.qsin):
        if os.path.exists(args.qsin+'.bz2'):
            print("%s doesn't exist - unzipping %s.bz2" % (args.qsin, args.qsin))
            os.system('bunzip2 %s.bz2' % args.qsin)
    RS = Nanoreactor(**dict(args._get_kwargs())) # _get_kwargs takes the ArgumentParser object and turns it into a dictionary
    RS.Output()
    print("Reaction product identification finished.  color.dat and bonds.dat generated.  Now run: vmd -e reactions.vmd -args %s" % args.xyzin)

if __name__ == "__main__":
    main()
