#!/bin/bash

# Set up TeraChem environment variables on Fire.

#======================================#
#|   Personal environment variables   |#
#======================================#
export PATH=$HOME/bin:$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED="y"
#======================================#
#|      Intel compiler variables      |#
#======================================#
. /opt/intel/composer_xe_2011_sp1.6.233/bin/iccvars.sh intel64
. /opt/intel/composer_xe_2011_sp1.6.233/bin/ifortvars.sh intel64
#======================================#
#|     CUDA environment variables     |#
#======================================#
export PATH=/opt/CUDA/cuda-5.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/CUDA/cuda-5.0/lib64:/opt/CUDA/cuda-5.0/lib:$LD_LIBRARY_PATH
export INCLUDE=/opt/CUDA/cuda-5.0/include:$INCLUDE
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
#|    Now do what needs to be done!   |#
#======================================#

echo "#=======================#"
echo "# ENVIRONMENT VARIABLES #"
echo "#=======================#"
echo

set

echo
echo "#=======================#"
echo "# STARTING CALCULATION! #"
echo "#=======================#"
echo
echo $@

time $@

