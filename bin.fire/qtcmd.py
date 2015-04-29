#!/home/leeping/local/bin/python

#=========================================================================#
#|                                                                       |#
#|                  TeraChem Molecular Dynamics Script                   |#
#|                                                                       |#
#|             Author: Lee-Ping Wang (leeping@stanford.edu)              |#
#|                                                                       |#
#| - Allows a long MD job to run on a runtime restricted SGE queue       |#
#|                                                                       |#
#| - Keeps track of MD simulation chunks and creates an unbroken         |#
#|   and non-overlapping trajectory                                      |#
#|                                                                       |#
#| - Script resubmits itself so that MD simulation runs continuously     |#
#|                                                                       |#
#| - Executes nanoreactor learning program alongside TeraChem MD         |#
#|                                                                       |#
#| - After job is done, scratch files on compute node are copied back.   |#
#|                                                                       |#
#|                             Instructions:                             |#
#|                                                                       |#
#| 1) Make sure directory doesn't contain a running job already          |#
#|                                                                       |#
#| 2) The directory should contain: start.xyz, start.in, restart.in      |#
#|                                                                       |#
#| In case a job needs to be manually restarted, delete this job and the |#
#| self-submitted job from the queue.                                    |#
#|                                                                       |#
#| The .chunk file should be 1 higher than the highest chunk_xxxx folder |#
#|                                                                       |#
#=========================================================================#

import os
import argparse

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
add_argument(parser, '--gpus', help='Specify the number of graphics processing units.',
             default=4, type=int)
add_argument(parser, '--rsync', help='Time interval (s) for rsyncing the scratch directories, zero to disable.',
             default=180, type=int)
add_argument(parser, '--auto', help='Use this argument of the job is automatically submitted.',
             action='store_true', default=False) 
add_argument(parser, '--long', help='Submit the queue to the Long Queue.',
             action='store_true', default=False) 
add_argument(parser, '--name', help='Specify the name of the job being submitted.',
             default='default', type=str) 
add_argument(parser, '--hold', help='Specify the job number used to hold the submitted job (useful if a job is resubmitting using this script).',
             default=0, type=int)
add_argument(parser, '--gpucc', help='Specify the required GPU compute capability (default 2.0 for Fermi cards, but possible to go to 1.3).',
             default="2.0|3.0|3.5|5.0", type=str)

print
print " #=========================================#"
print " #     Nanoreactor MD launching script     #"
print " #  Use the -h argument for detailed help  #"
print " #=========================================#"
print
args = parser.parse_args()

#==========================#
#  Determine chunk number  #
#   and other arguments.   #
#==========================#
chunk = 0
if os.path.exists('.chunk'):
    chunk = float(open('.chunk').readlines()[0].strip())

dnm = "chunk_%04i" % chunk
prevd = "chunk_%04i" % (chunk-1)
cwd = os.getcwd()
if args.name == 'default':
    if os.path.exists('.jobname'):
        jobname = open('.jobname').readlines()[0].strip()
    else:
        jobname = os.path.split(cwd)[-1]
else:
    jobname = args.name

if args.long:
    queue="gpulongq"
    hrt="330:00:00"
else:
    # queue="gpuq@fire-20-*\n#$-q gpuq@fire-09-*"
    queue="gpuq"
    hrt="36:00:00"

hold=""
if args.hold > 0:
    hold = "#$ -hold_jid %i\n" % args.hold

# Crash if the job already exists in the queue.
# Currently doesn't work because recently deleted jobs are still reported by qstat -j.
# Crash = True
# if args.auto:
#     Crash = False
# for line in os.popen("qstat -j %s" % jobname).readlines():
#     if 'jobs do not exist' in line:
#         Crash = False
# if Crash:
#     raise Exception("A running job already exists with the same name.  Either delete the running job or rename this one.")

if args.rsync == 0:
    interval = 8640000 # 100 days.
else:
    interval = args.rsync

#===========================#
#   Check if files exist:   #
#  start.xyz, (re)start.in  #
#===========================#
fnm = 'start.xyz'
if not os.path.exists(fnm):
    raise Exception('The %s file does not exist!' % fnm)

if os.path.exists(dnm):
    if not os.path.exists(os.path.join(dnm,'run.out')) and not os.path.exists(os.path.join(dnm,'scr','coors.xyz')):
        print "The directory pointed to by the .chunk file exists but contains no data - ok."
    else:
        raise Exception("The directory pointed to by the .chunk file exists and contains data - script will not continue.")
os.makedirs(dnm)
if chunk > 0 and not os.path.exists(prevd):
    raise Exception("The directory containing the previous chunk (%s) doesn't exist!" % prevd)

#===========================#
#  Create copy of TeraChem  #
#   input file with #gpus   #
#===========================#
fnm = "start.in"
if not os.path.exists(fnm):
    raise Exception('The %s file does not exist!' % fnm)
f = open(fnm).readlines()
o = open(os.path.join(dnm,'run.in'),'w')
have_restart = False
for l in f:
    s = [i.lower() for i in l.split()]
    if len(s) >= 1 and s[0] == 'gpus':
        print >> o, "gpus %i" % args.gpus
    elif len(s) >= 1 and s[0] == 'restartmd':
        have_restart = True
        if chunk == 0:
            continue
        else:
            print >> o, l,
    elif len(s) >= 1 and s[0] == 'end':
        if not have_restart and chunk > 0:
            print >> o, 'restartmd restart.md'
        print >> o, l,
    else:
        print >> o, l,
o.close()

os.system('cp start.xyz %s' % dnm)

#============================#
#  Create copy of fixed_atom #
#    and surface.xyz files   #
#     in chunk directory     #
#============================#

fixedatoms = 0
fnm = "fixed_atoms"
if os.path.exists(fnm):
    os.system('cp fixed_atoms %s' % dnm)
    fixedatoms = 1

surface_on = 0
fnm = "surface.xyz"
if os.path.exists(fnm):
    os.system('cp surface.xyz %s' % dnm)
    surface_on = 1

os.chdir(dnm)

# -fout scr/coors.xyz {dnm}/scr/coors.xyz
# -fout scr/restart.md {dnm}/scr/restart.md
# -fout scr/restart.mdRnd {dnm}/scr/restart.mdRnd
# -fout scr/charge.xls {dnm}/scr/charge.xls
# -fout scr/vel.log {dnm}/scr/vel.log
# -fout scr/log.xls {dnm}/scr/log.xls


fout="""\
#!/bin/bash
#$ -N {jobname}
#$ -l gpus=1
#$ -l h_rss=8G
#$ -l gpucc={gpucc}
#$ -l h_rt={hrt}
#$ -pe smp {gpus}
#$ -cwd
#$ -q "{queue}"
# -fin start.xyz
# -fin run.in
{fixed}{surface}# -fout scr/* scr/
{hold}
# This file was generated by {scriptname}
# Job is submitted in chunk_xxxx directory
# The "$working" directory is one level up.
# Not copying back the restart.md saves space, as long as
# we remember that restart.md always comes from the scratch folder
# of the LAST chunk.
# =fout restart.md
# =fout restart.mdRnd

rsync_fork() {{
while true; do
    sleep {interval}
    rsync -auvz $SGE_O_TEMPDIR/scr/ $SGE_O_WORKDIR/scr/
done
}}

learn_fork() {{
# Consolidate output files.
gather-info.py
mkdir -p Consolidated
mv energies.txt temperatures.txt all-coors.xyz charge-spin.txt bond-orders.txt Consolidated
# Identify reaction products!
cd Consolidated
rm -f extract_*.xyz
# Run LearnReactions.py.  
# Custom arguments are available by creating a script somewhere above
# the folder where this is run. The closer the script to the current
# folder, the higher the priority. (This is like a PATH environment variable)
d=$PWD
while true ; do
    echo $d
    if [[ -f $d/learn.sh ]] ; then
        chmod +x $d/learn.sh
        cmd=$d/learn.sh
        break
    elif [[ ${{#d}} -eq 1 ]] ; then
        cmd="LearnReactions.py all-coors.xyz"
        break
    fi
    d=$(dirname $d)
done
echo $cmd
$cmd
rm charge-spin.txt.bz2
rm bond-orders.txt.bz2
bzip2 charge-spin.txt
bzip2 bond-orders.txt
cd ..
}}

#======================================#
#|   Personal environment variables   |#
#======================================#
. /etc/bashrc
. /etc/profile
working={cwd}
export PATH=$HOME/bin:$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
#======================================#
#|      Intel compiler variables      |#
#======================================#
. /opt/intel/composer_xe_2013_sp1.3.174/bin/iccvars.sh intel64
. /opt/intel/composer_xe_2013_sp1.3.174/bin/ifortvars.sh intel64
#======================================#
#|     CUDA environment variables     |#
#|        From Fire modulefile        |#
#======================================#
module unload cuda
module load cuda/6.5
#======================================#
#|   TeraChem environment variables   |#
#======================================#
export TeraChem=$HOME/opt/terachem
export PATH=$TeraChem/bin:$PATH
export LD_LIBRARY_PATH=$TeraChem/lib:$LD_LIBRARY_PATH
#======================================#
#|  Grid Engine environment variables |#
#======================================#
export SGE_ROOT=/opt/sge
export SGE_CELL=default
export SGE_CLUSTER_NAME=p6444
export PATH=$SGE_ROOT/bin/linux-x64:$PATH
#======================================#
#|    Go into my working directory    |#
#======================================#
cd $working
#======================================#
#|   Determine which chunk of the     |#
#|   simulation we're currently on    |#
#|   Also increment the chunk file    |#
#======================================#
if [ -f .chunk ] ; then
    chunk=$(cat .chunk)
    learn_fork &
    # Move pesky SGE output files to their own directory.
    # Now they are generated in the chunk directories...
    # mkdir -p sge_logfiles
    # find . -maxdepth 2 -name \*[oe][0-9]* -exec mv {{}} sge_logfiles \; 2> /dev/null
else
    chunk=0
fi
next_chunk=$(( chunk + 1 ))
echo $next_chunk > .chunk
#======================================#
#|        Submit the next job         |#
#======================================#
if (( chunk < 200 )) ; then
    qtcmd.py --auto --hold $JOB_ID --gpus {gpus} --gpucc "{gpucc}" --name "{jobname}" {long}
fi
#======================================#
#|  Go into the temporary directory   |#
#======================================#
dnm=$(printf "chunk_%04i" $chunk)
mkdir -p $dnm/scr
echo "$HOSTNAME:$SGE_O_TEMPDIR/scr" > $dnm/scr/scratch.remote
touch $dnm/scr/rsync_every_{interval}_seconds
cd $SGE_O_TEMPDIR
#======================================#
#|  Grab restart files from last job  |#
#======================================#
if (( chunk != 0 )) ; then
    last_chunk=$(( chunk - 1 ))
    last_dnm=$(printf "chunk_%04i" $last_chunk)
    cp $working/$last_dnm/scr/restart.md .
    cp $working/$last_dnm/scr/restart.mdRnd .
    # Space-saving measures
    rm $working/$last_dnm/scr/restart.md.*
    rm $working/$last_dnm/restart.md*
    bzip2 $working/$last_dnm/scr/vel.log
fi
rsync_fork &
#======================================#
#|    Now do what needs to be done!   |#
#======================================#
terachem run.in > $working/$dnm/run.out 2> $working/$dnm/run.err
rsync -auvz $SGE_O_TEMPDIR/scr/ $SGE_O_WORKDIR/scr/
"""

with open('qsub.sh','w') as f: print >> f, fout.format(jobname=jobname, cwd=cwd, dnm=dnm, hold=hold, scriptname=__file__, hrt=hrt, interval=interval, queue=queue, gpus=args.gpus, gpucc=args.gpucc, long="--long" if args.long else "", fixed="# -fin fixed_atoms\n" if fixedatoms else "", surface="# -fin surface.xyz\n" if surface_on else "")
os.system('qsub qsub.sh')
