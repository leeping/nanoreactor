README

Author: Laszlo R Seress
Originally written .... 12/10/2015
Last updated .......... 01/05/2016

These files are used to generate inputs for the nanoreactor.  Below is a description of each file.

***************
atomicradii.csv
***************

Contains atomic radii for elements from H to Rn (1 to 86)

The atomic radii stored in atomicradii.csv are approximate and have NOT been double-checked for accurary.  (i.e. they are from wikipedia - https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page) -  accessed 12/10/2015).  Values were taken from the "calculated" column because this column had the most entries.  For missing entries, a reasonable value was chosen. 

The calculation of the total volume taken up by the atoms in the input file is APPROXIMATE IN NATURE and is not meant to be quantitatively precise.  User beware.  



***********
inputgen.py
***********

This script needs python 2.7.  

On fire, run "module load python" before running the script. 

On xstream, 
module load CUDA/7.5.18
module load intel
module load Python

This script generates inputs for the nanoreactor. The flags can be used to vary what goes into the input file(s) and run files.

There is very minimal error checking in this software, so you can use it to do stupid things.  E.g. you can set the basis set to equal 'martinez' or you can request 1000 GB of RAM.  User beware.

Usage with a coordinate file:
python inputgen.py YOUR_JOB NAME DESIRED_PATH_TO_OUTPUT_FOLDER_FOR_FOLDERS_AND_FILES QUEUE_SYSTEM -c COORDINATE_FILE_PATH/COORDINATE_FILE.xyz -other_optional_flags

Usage with smiles:
Need to "module load packmol"

Exit error codes:
1 - Coordinate file input was not .xyz
2 - levelshiftvala was given when levelshift = no
3 - levelshiftvalb was given when levelshift = no
4 - queue system was not sge or slurm
50 - inner radius generated was less than recommended minimum radius

***********************
inputparamsgenmodule.py
***********************

This is a module that contains functions that find the parameters for input files (e.g. calculating what radius to use, etc.)

***************************
inputsubmissiongenmodule.py
***************************

This module contains the functions that actually write the input and run.sh files, as well as the functions that 
submit jobs to sge and slurm queues.

***********
scratch.txt
***********

A place for spare bits of code, data, and other things that I wanted to save but didn't have a home for.