#!/usr/bin/env python

import os, argparse
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
add_argument(parser, '-s', dest='stride', help='Skip number of frames when analyzing trajectory, default is don\'t skip.',
             default=1, type=int)
add_argument(parser, '-e', dest='enhance', help='Enhancement factor for bond detection; larger number = more bonds. For hydroxide we used 1.25 * (1.2 / 1.2125).',
             default=1.4, type=float)
add_argument(parser, '-m', dest='mindist', help='Below this distance (in Angstrom) all atoms are considered to be bonded.',
             default=1.0, type=float)
add_argument(parser, '--metastability', help='Metastability parameter for Hidden Markov Model used to stabilize time series (closer to 1.0 = more aggressive ; <= 0 to turn off).',
             default=0.999, type=float)
add_argument(parser, '--pcorrectemit', help='Correctness probability parameter for Hidden Markov Model used to stabilize time series (closer to 0.5 = more aggressive; <= 0 to turn off).',
             default=0.6, type=float)
add_argument(parser, '-p', dest='printlvl', help='Print level.  1: Print out unrectified time series. 2: Print out empirical formulas as they are discovered.',
             default=0, type=int)
# add_argument(parser, '-P', dest='padtime', help='Pad reactive trajectories by this many frames. Defaults to learntime/4.',
#              default=0, type=int)
add_argument(parser, '-t', dest='learntime', help='Molecules existing for at least (learntime*stride) frames are recognized and given a color.',
             default=200, type=int)
# Default behavior is to exact reactions but not reaction products. :)
add_argument(parser, '-X', dest='extract', help='Extract recognized molecules and write their trajectories to extract_###.xyz .',
             action='store_true', default=False) 
add_argument(parser, '-R', dest='saverxn', help='Extract recognized chemical reactions and write their trajectories to reaction_###.xyz .',
             action='store_true', default=True)
add_argument(parser, '-f', dest='frames', help='Trajectory length to process (out of total frames in XYZ).  Passing zero specifies all frames.',
             default=0, type=int)
add_argument(parser, '-o', dest='xyzout', help='Output coordinate file containing selected frames.  Enter "None" to prevent writing output.',
             default='None', type=str)
add_argument(parser, '-b', dest='boring', nargs='+', help='Boring molecules to be excluded from learning.  Ignore isomers in the first frame that match the given alphabetically-ordered empirical formula (or type All to ignore all isomers in the first frame.)',
             default=['all'], type=str)
add_argument(parser, '-D', dest='disallow', nargs='+', help='Disallowed molecules to be excluded from recognition (by empirical formula).  This constitutes manual intervention into the algorithm and should be avoided.',
             default=[], type=str)
add_argument(parser, '-N', '--neutralize', help='Extract nearby molecules to neutralize the system', action='store_true')



print
print " #=========================================#"
print " #       Reactive MD learning script       #"
print " #  Use the -h argument for detailed help  #"
print " #=========================================#"
print
args = parser.parse_args()
    
def main():
    if not os.path.exists(args.qsin):
        if os.path.exists(args.qsin+'.bz2'):
            print "%s doesn't exist - unzipping %s.bz2" % (args.qsin, args.qsin)
            os.system('bunzip2 %s.bz2' % args.qsin)
    RS = Nanoreactor(**dict(args._get_kwargs())) # _get_kwargs takes the ArgumentParser object and turns it into a dictionary
    RS.Output()
    print "Reaction product identification finished.  color.dat and bonds.dat generated.  Now run: vmd -e reactions.vmd -args %s" % RS.fout

if __name__ == "__main__":
    main()
