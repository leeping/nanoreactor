#!/bin/bash
#$ -N lrn_smash7.3-o
#$ -l h_rss=2G
#$ -l h_rt=36:00:00
#$ -pe smp 8
#$ -cwd
#$ -q cpuq@fire-0*

learn_fork() {
# Consolidate output files.
gather-info.py
mkdir -p Consolidated
mv energies.txt temperatures.txt all-coors.xyz charge-spin.txt Consolidated
# Identify reaction products!
cd Consolidated
rm -f extract_*.xyz reaction_*.xyz
LearnReactions.py all-coors.xyz
rm charge-spin.txt.bz2
bzip2 charge-spin.txt
cd ..
}

#======================================#
#|   Personal environment variables   |#
#======================================#
working=/home/leeping/projects/MTZ10-Convergers/UM2/smash7.3-o
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
export OMP_NUM_THREADS=8
cd $working
learn_fork

