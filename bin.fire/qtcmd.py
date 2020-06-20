#!/usr/bin/env python

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
import re
import argparse
from numpy.random import randint
from collections import OrderedDict

#==========================#
#     Parse arguments.     #
#==========================#

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=1, help='Specify the number of graphics processing units.')
parser.add_argument('--auto', action='store_true', help='Use this argument of the job is automatically submitted.') 
parser.add_argument('--time', type=str, default="24:00:00", help='Specify a hh:mm:ss time limit for the job.')
parser.add_argument('--name', type=str, default='default', help='Specify the name of the job being submitted.')
parser.add_argument('--tera', type=str, default='/home/leeping/opt/terachem/current/bin/terachem', help='Specify absolute path of TeraChem executable.')
parser.add_argument('--hold', type=int, default=0, help='Specify the job number used to hold the submitted job (not necessary if submitting by hand).')

print()
print(" #=========================================#")
print(" #     Nanoreactor MD launching script     #")
print(" #  Use the -h argument for detailed help  #")
print(" #=========================================#")
print()
args = parser.parse_args()

def edit_tcin(fin=None, fout=None, options={}, defaults={}):
    """
    Parse, modify, and/or create a TeraChem input file.

    Parameters
    ----------
    fin : str, optional
        Name of the TeraChem input file to be read
    fout : str, optional
        Name of the TeraChem output file to be written, if desired
    options : dict, optional
        Dictionary of options to overrule TeraChem input file. Pass None as value to delete a key.
    defaults : dict, optional
        Dictionary of options to add to the end
    
    Returns
    -------
    dictionary
        Keys mapped to values as strings.  Certain keys will be changed to integers (e.g. charge, spinmult).
        Keys are standardized to lowercase.
    """
    intkeys = ['charge', 'spinmult']
    Answer = OrderedDict()
    # Read from the input if provided
    if fin is not None:
        for line in open(fin).readlines():
            line = line.split("#")[0].strip()
            if len(line) == 0: continue
            if line == 'end': break
            s = line.split(' ', 1)
            k = s[0].lower()
            v = s[1].strip()
            if k == 'coordinates':
                if not os.path.exists(v.strip()):
                    raise RuntimeError("TeraChem coordinate file does not exist")
            if k in intkeys:
                v = int(v)
            if k in Answer:
                raise RuntimeError("Found duplicate key in TeraChem input file: %s" % k)
            Answer[k] = v
    # Replace existing keys with ones from options
    for k, v in list(options.items()):
        Answer[k] = v
    # Append defaults to the end
    for k, v in list(defaults.items()):
        if k not in list(Answer.keys()):
            Answer[k] = v
    for k, v in list(Answer.items()):
        if v is None:
            del Answer[k]
    # Print to the output if provided
    havekeys = []
    if fout is not None:
        with open(fout, 'w') as f:
            # If input file is provided, try to preserve the formatting
            if fin is not None:
                for line in open(fin).readlines():
                    # Find if the line contains a key
                    haveKey = False
                    uncomm = line.split("#", 1)[0].strip()
                    # Don't keep anything past the 'end' keyword
                    if uncomm.lower() == 'end': break
                    if len(uncomm) > 0: 
                        haveKey = True
                        comm = line.split("#", 1)[1].replace('\n','') if len(line.split("#", 1)) == 2 else ''
                        s = line.split(' ', 1)
                        w = re.findall('[ ]+',uncomm)[0]
                        k = s[0].lower()
                        if k in Answer:
                            line_out = k + w + str(Answer[k]) + comm
                            havekeys.append(k)
                        else:
                            line_out = line.replace('\n', '')
                    else:
                        line_out = line.replace('\n', '')
                    print(line_out, file=f)
            for k, v in list(Answer.items()):
                if k not in havekeys:
                    print("%-15s %s" % (k, str(v)), file=f)
    return Answer

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

# if args.long:
#     queue="gpulongq"
#     hrt="330:00:00"
# else:
#     # queue="gpuq@fire-20-*\n#$-q gpuq@fire-09-*"
#     queue="gpuq"
#     hrt="36:00:00"

hold=""
if args.hold > 0:
    hold = "#SBATCH -d afterany:%i\n" % args.hold

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

# if args.rsync == 0:
#     interval = 8640000 # 100 days.
# else:
#     interval = args.rsync

#===========================#
#   Check if files exist:   #
#  start.xyz, (re)start.in  #
#===========================#
fnm = 'start.xyz'
if not os.path.exists(fnm):
    raise Exception('The %s file does not exist!' % fnm)

if os.path.exists(dnm):
    if not os.path.exists(os.path.join(dnm,'run.out')) and not os.path.exists(os.path.join(dnm,'scr','coors.xyz')):
        print("The directory pointed to by the .chunk file exists but contains no data - ok.")
    else:
        raise Exception("The directory pointed to by the .chunk file exists and contains data - script will not continue.")
else:
    os.makedirs(dnm)
if chunk > 0 and not os.path.exists(prevd):
    raise Exception("The directory containing the previous chunk (%s) doesn't exist!" % prevd)

#===========================#
#  Create copy of TeraChem  #
#   input file with #gpus   #
#===========================#

fnm = "guess.in"
makeGuess = False
if chunk == 0 and os.path.exists(fnm):
    makeGuess = True
    tcg = edit_tcin(fin=fnm)
    tcg['gpus'] = args.gpus
    edit_tcin(fin=fnm, fout=os.path.join(dnm, 'guess.in'), options=tcg)

fnm = "start.in"
if not os.path.exists(fnm):
    raise Exception('The %s file does not exist!' % fnm)

tcin = edit_tcin(fin=fnm)
tcin['gpus'] = args.gpus
if tcin.get('coordinates','none') != 'start.xyz':
    raise Exception('TeraChem input file must have coordinates start.xyz')
if chunk > 0: 
    tcin['restartmd'] = 'restart.md'
    if 'guess' in tcin: del tcin['guess']
else:
    if 'restartmd' in tcin: del tcin['restartmd']
    if makeGuess:
        if tcin['method'][0] == 'u':
            tcin['guess'] = 'ca0 cb0'
        else:
            tcin['guess'] = 'c0'

edit_tcin(fin=fnm, fout=os.path.join(dnm, 'run.in'), options=tcin)

os.system('cp start.xyz %s' % dnm)

#============================#
#  Create copy of fixed_atom #
#    and surface.xyz files   #
#     in chunk directory     #
#============================#

# LPW: Enable this functionality later

# fixedatoms = 0
# fnm = "fixed_atoms"
# if os.path.exists(fnm):
#     os.system('cp fixed_atoms %s' % dnm)
#     fixedatoms = 1

# surface_on = 0
# fnm = "surface.xyz"
# if os.path.exists(fnm):
#     os.system('cp surface.xyz %s' % dnm)
#     surface_on = 1

os.chdir(dnm)

fout="""\
#!/bin/bash -l
#SBATCH -J {jobname}
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -n {gpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
{hold}

# This file was generated by {scriptname}
# Job is submitted in chunk_xxxx directory
# The "$working" directory is one level up.
# Not copying back the restart.md saves space, as long as
# we remember that restart.md always comes from the scratch folder
# of the LAST chunk.

# LPW: Implement this later.
# learn_fork() {{
# # Consolidate output files.
# gather-info.py
# mkdir -p Consolidated
# mv energies.txt temperatures.txt all-coors.xyz charge-spin.txt bond-orders.txt Consolidated
# # Identify reaction products!
# cd Consolidated
# rm -f extract_*.xyz
# # Run LearnReactions.py.  
# # Custom arguments are available by creating a script somewhere above
# # the folder where this is run. The closer the script to the current
# # folder, the higher the priority. (This is like a PATH environment variable)
# d=$PWD
# while true ; do
#     echo $d
#     if [[ -f $d/learn.sh ]] ; then
#         chmod +x $d/learn.sh
#         cmd=$d/learn.sh
#         break
#     elif [[ ${{#d}} -eq 1 ]] ; then
#         cmd="LearnReactions.py all-coors.xyz"
#         break
#     fi
#     d=$(dirname $d)
# done
# echo $cmd
# $cmd
# rm charge-spin.txt.bz2
# rm bond-orders.txt.bz2
# bzip2 charge-spin.txt
# bzip2 bond-orders.txt
# cd ..
# }}

#======================================#
#|   Personal environment variables   |#
#======================================#
# . /etc/bashrc
# . /etc/profile
working={cwd}
# export PATH=$HOME/bin:$HOME/local/bin:$PATH
# export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
#======================================#
#|      Intel compiler variables      |#
#======================================#
# . /opt/intel/composer_xe_2013_sp1.3.174/bin/iccvars.sh intel64
# . /opt/intel/composer_xe_2013_sp1.3.174/bin/ifortvars.sh intel64
#======================================#
#|     CUDA environment variables     |#
#|        From Fire modulefile        |#
#======================================#
# module unload cuda
# module load cuda/9.0
#======================================#
#|   TeraChem environment variables   |#
#======================================#
export TeraChem={terapath}
export PATH=$TeraChem/bin:$PATH
export LD_LIBRARY_PATH=$TeraChem/lib:$LD_LIBRARY_PATH
#======================================#
#|  Grid Engine environment variables |#
#======================================#
# export SGE_ROOT=/opt/sge
# export SGE_CELL=default
# export SGE_CLUSTER_NAME=p6444
# export PATH=$SGE_ROOT/bin/linux-x64:$PATH
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
#     learn_fork &
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
    qtcmd.py --auto --hold $SLURM_JOB_ID --gpus {gpus} --name "{jobname}" --time {time} --tera {tera}
fi
#======================================#
#|  Go into the temporary directory   |#
#======================================#
dnm=$(printf "chunk_%04i" $chunk)
# mkdir -p $dnm/scr
# echo "$HOSTNAME:$SGE_O_TEMPDIR/scr" > $dnm/scr/scratch.remote
# touch $dnm/scr/rsync_every_{{interval}}_seconds
# cd $SGE_O_TEMPDIR
cd $dnm
#======================================#
#|  Grab restart files from last job  |#
#======================================#
if (( chunk != 0 )) ; then
    last_chunk=$(( chunk - 1 ))
    last_dnm=$(printf "chunk_%04i" $last_chunk)
    ln -s ../$last_dnm/scr/restart.md
    ln -s ../$last_dnm/scr/restart.mdRnd
    # cp $working/$last_dnm/scr/restart.md .
    # cp $working/$last_dnm/scr/restart.mdRnd .
    # Space-saving measures
    # rm $working/$last_dnm/scr/restart.md.*
    # rm $working/$last_dnm/restart.md*
    # bzip2 $working/$last_dnm/scr/vel.log
else
    if [ -f guess.in ] ; then
        terachem guess.in > guess.out 2> guess.err
        ln -s scr/c*0 .
    fi
fi
# rsync_fork &
#======================================#
#|    Now do what needs to be done!   |#
#======================================#
terachem run.in > run.out 2> run.err
# If the job finishes or crashes for some reason, then the next job should be deleted.
submitted_job=$(tail -1 submit.txt | awk '{{print $NF}}')
scancel $submitted_job

# rsync -auvz $SGE_O_TEMPDIR/scr/ $SGE_O_WORKDIR/scr/
"""

terapath=os.path.dirname(os.path.dirname(args.tera))

print(cwd)
with open('sbatch.sh','w') as f: print(fout.format(jobname=jobname, cwd=cwd, hold=hold, scriptname=__file__, time=args.time, gpus=args.gpus, mem=args.gpus*8000, terapath=terapath, tera=args.tera), file=f)
os.system('sbatch sbatch.sh | tee .submit.txt')
