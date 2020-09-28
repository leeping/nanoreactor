#!/home/leeping/local/bin/python
from __future__ import print_function
import os
import argparse

cwd = os.getcwd()
jobname = os.path.split(cwd)[-1]

queue="cpuq"
hrt="36:00:00"

fout="""\
#!/bin/bash
#$ -N lrn_{jobname}
#$ -l h_rss=4G
#$ -l h_rt={hrt}
#$ -pe smp 4
#$ -cwd
#$ -q {queue}

learn_fork() {{
# Consolidate output files.
gather-info.py
mkdir -p Consolidated
mv energies.txt temperatures.txt all-coors.xyz charge-spin.txt Consolidated
# Identify reaction products!
cd Consolidated
rm -f extract_*.xyz reaction_*.xyz
LearnReactions.py all-coors.xyz
bzip2 charge-spin.txt
cd ..
}}

#======================================#
#|   Personal environment variables   |#
#======================================#
working={cwd}
export PATH=$HOME/bin:$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
#======================================#
#|      Intel compiler variables      |#
#======================================#
. /opt/intel/composer_xe_2011_sp1.6.233/bin/iccvars.sh intel64
. /opt/intel/composer_xe_2011_sp1.6.233/bin/ifortvars.sh intel64
#======================================#
#|    Go into my working directory    |#
#======================================#
cd $working
learn_fork
"""

with open('qlearn.sh','w') as f: print(fout.format(jobname=jobname, cwd=cwd, hrt=hrt, queue=queue), file=f)
os.system('qsub qlearn.sh')
