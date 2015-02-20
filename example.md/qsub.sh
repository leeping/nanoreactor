#!/bin/bash
#$ -N smash7.3
#$ -l gpus=1
#$ -l gpuarch=fermi
#$ -l h_rss=4G
#$ -l h_rt=360:00:00
#$ -pe smp 8
#$ -cwd
#$ -q gpulongq@fire-09-07

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
#|                             Instructions:                             |#
#|                                                                       |#
#| 1) Make sure directory doesn't contain a running job already          |#
#|                                                                       |#
#| 2) Make sure the gpus setting in the TeraChem files match this script |#
#|                                                                       |#
#| 3) The directory should contain: start.xyz, start.in, restart.in      |#
#|                                                                       |#
#| 4) Submit this script by running ---> qsub qsub.sh <---               |#
#|                                                                       |#
#| In case a job needs to be manually restarted, delete this job and the |#
#| self-submitted job from the queue, then repeat step 4                 |#
#|                                                                       |#
#| The .chunk file should be 1 higher than the highest chunk_xxxx folder |#
#|                                                                       |#
#=========================================================================#

#======================================#
#|   Personal environment variables   |#
#======================================#
working=$HOME/projects/MTZ10-Convergers/UM2/smash7.3
export PATH=$HOME/bin:$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
#======================================#
#|      Intel compiler variables      |#
#======================================#
. /opt/intel/composer_xe_2011_sp1.6.233/bin/iccvars.sh intel64
. /opt/intel/composer_xe_2011_sp1.6.233/bin/ifortvars.sh intel64
#======================================#
#|     CUDA environment variables     |#
#|        From Fire modulefile        |#
#|  Sometimes 'modules' doesn't work  |#
#======================================#
export PATH=/opt/CUDA/cuda4.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/CUDA/cuda4.0/lib64:/opt/CUDA/cuda4.0/lib:$LD_LIBRARY_PATH
#======================================#
#|   TeraChem environment variables   |#
#======================================#
export TeraChem=$HOME/src/terachem/production/terachem
export PATH=$TeraChem:$PATH
#======================================#
#|  Grid Engine environment variables |#
#======================================#
export SGE_ROOT=/opt/sge
export SGE_CELL=default
export SGE_CLUSTER_NAME=p6444
export PATH=$SGE_ROOT/bin/lx26-amd64:$PATH
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
    # Consolidate output files.
    gather-info.py
    mkdir -p Consolidated
    mv energies.txt temperatures.txt all-coors.xyz Consolidated
    # Identify reaction products!
    cd Consolidated
    rm -f extract_*.xyz reaction_*.xyz
    $HOME/local/bin/LearnReactions.py all-coors.xyz &
    cd ..
    # Move pesky SGE output files to their own directory.
    mkdir -p sge_logfiles
    find . -maxdepth 1 -name \*[oe][0-9]* -exec mv {} sge_logfiles \;
else
    chunk=0
fi
next_chunk=$(( chunk + 1 ))
echo $next_chunk > .chunk
#======================================#
#|        Submit the next job         |#
#======================================#
if (( chunk < 100 )) ; then
    qsub -hold_jid $JOB_ID qsub.sh
fi
#======================================#
#|    Go into the chunk directory     |#
#======================================#
dnm=$(printf "chunk_%04i" $chunk)
mkdir -p $dnm
cd $dnm
#======================================#
#|    Now do what needs to be done!   |#
#======================================#
if (( chunk == 0 )) ; then
    ln -s ../start.in ./run.in
else
    last_chunk=$(( chunk - 1 ))
    last_dnm=$(printf "chunk_%04i" $last_chunk)
    ln -s ../restart.in ./run.in
    ln -s ../$last_dnm/scr/restart.md
    ln -s ../$last_dnm/scr/restart.mdRnd
fi
ln -s ../start.xyz .
int run.in > run.out 2> run.err
