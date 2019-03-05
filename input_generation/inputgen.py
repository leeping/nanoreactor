# input generation script for nanoreactor
# Author Laszlo R Seress - seress@stanford.edu

# Version History: 

# 1.0 - 12/10/2015
# Reads in atomic radii and tabulates them.  Reads in a coordinate file (.xyz) and gives a recommended inner radius for the nanoreactor.
# 1.1 - 12/11/2015
# Generates input files, makes folders, etc. 
# 2.0 - 12/14/2015
# makes run.sh files, submits jobs.  some bug fixes from previous versions.
# 3.0 - 1/5/2016
# now uses modules


import csv
import numpy as np
import argparse
import sys
import os
import math
import subprocess
import inputparamsgenmodule as ipg 
import inputsubmissiongenmodule as isg

# argument parsing
parser = argparse.ArgumentParser()
# optional to read in a coordinate file
parser.add_argument("-c", "--coordinate_filename", help="filename of the coordinates, please include the FULL PATH and file extension - must be .XYZ")
parser.add_argument("-ivr", "--inner_volume_ratio", type=float, help="ratio of volume of inner sphere to sum of volumes of atoms, should be a float.  Default is 1.45")
parser.add_argument("-ovr", "--outer_volume_ratio", type=float, help="ratio of volume of outer sphere to volume of inner sphere, should be a float.  Default is 3.0")
parser.add_argument("-ch", "--charge_input", type=int, help="the charge for the input file. Default is 0")
parser.add_argument("-b", "--basis_set_input", help="the basis set for the input file.  Default is lanl2dz_ecp")
parser.add_argument("-tinit", "--t_initial_input", type=int, help="the tinit value for the input file. Default is 1200")
parser.add_argument("-d", "--dispersion_input", help="whether dispersion is yes or no in input file. Default is no")
parser.add_argument("-th", "--thermostat_input", help="type of thermostat for input file. Default is langevin")
parser.add_argument("-t0", "--t_zero_input", type=int, help="the t0 value for the input file. Default is 1500")
parser.add_argument("-lnv", "--lnvtime_input", type=int, help="the lnvtime value for the input file. Default is 100")
parser.add_argument("-conv", "--convthre_input", help="the convthre value for the input file. Default is 0.005")
parser.add_argument("-ls", "--level_shift_input", help="whether level shift is yes or no in input file. Default is yes")
parser.add_argument("-lsa", "--level_shift_vala_input", help="value of levelshiftvala in input file. Default is 0.3")
parser.add_argument("-lsb", "--level_shift_valb_input", help="value of levelshiftvalb in input file. Default is 0.1")
parser.add_argument("-mg", "--mix_guess_input", help="value of mixguess in input file. Default is 0")
parser.add_argument("-int", "--integrator_input", help="integerator in input file. Default is regular")
parser.add_argument("-r", "--run_input", help="run type in input file. Default is md")
parser.add_argument("-m", "--method_input", help="method type in input file. Default is uhf")
parser.add_argument("-scf", "--scf_input", help="scf type in input file. Default is diis+a")
parser.add_argument("-ti", "--timings_input", help="whether timings is set to yes or no in input file. Default is yes")
parser.add_argument("-n", "--nsteps_input", type=int, help="the nsteps value for the input file. Default is 5000")
parser.add_argument("-g", "--gpus_input", type=int, help="the gpus value for the input file. Default is 4")
parser.add_argument("-max", "--maxit_input", type=int, help="the maxit value for the input file. Default is 300")
parser.add_argument("-bc", "--mdbc_input", help="the md boundary condition for the input file. Default is spherical")
parser.add_argument("-bch", "--mdbc_hydrogen_input", help="whether mdbc_hydrogen is yes or no in the input file. Default is yes")
parser.add_argument("-r2", "--mdr2_count_input", type=int, help="the number of r2 values to iterate over for the input files. Default is 1")
parser.add_argument("-r2f", "--mdr2_factor", help="the factor that determines the spread in r2 values sample. Default is 0.03 i.e. will sample (1.0, 0.97. 1.03, ...) * recommended radius")
parser.add_argument("-k2", "--mdk2_count_input", type=int, help="the number of k2 values to iterate over for the input files. Default is 1, max is 20")
parser.add_argument("-ms", "--mdbc_mass_scaled_input", help="whether mdbc_mass_scaled set to yes or no in input file. can be yes, no, or both (makes 2x as many files). Default is yes")
parser.add_argument("job_name", help="the name of the job - only 6 characters - more will be truncated.")
parser.add_argument("output_folder", help="path to folder where output is desired. ")
parser.add_argument("queue_system", help="which queue system the cluster uses: sge or slurm? ")
parser.add_argument("-mem", "--memory_run", type=int, help="how much memory to use in GB. Default = 5")
parser.add_argument("-t", "--time_run",  help="how much time to use in Hr:Min:Sec format. Default = 35:59:00")
parser.add_argument("-nt", "--ntasks_run", type=int, help="how many tasks per node. Default = 1")
parser.add_argument("-s", "--submit_run", action='store_true', help="whether or not to submit run.sh scripts. If the flag is included, it will submit.")
args = parser.parse_args()

# handle the command line arguments

# check to make sure that coordinate file is .xyz
if args.coordinate_filename:
	if (args.coordinate_filename[-4]+args.coordinate_filename[-3]+args.coordinate_filename[-2]+args.coordinate_filename[-1] != ".xyz"):
		print "The coordinate file type must be .xyz"
		sys.exit(1)

# set inner volume ratio
if args.inner_volume_ratio:
	inner_volume_ratio = float(args.inner_volume_ratio)
else:
	inner_volume_ratio = 1.45

# set outer volume ratio
if args.outer_volume_ratio:
	outer_volume_ratio = float(args.outer_volume_ratio)
else:
	outer_volume_ratio = 3.0

# set charge
if args.charge_input:
	charge_input = int(args.charge_input)
else:
	charge_input = 0

# set basis set
if args.basis_set_input:
	basis_set_input = args.basis_set_input
else:
	basis_set_input = 'lanl2dz_ecp'

# set tinit
if args.t_initial_input:
	t_initial_input = int(args.t_initial_input)
else:
	t_initial_input = 1200

# set dispersion
if args.dispersion_input:
	if args.dispersion_input == 'yes':
		dispersion_input = 'yes'
	else:
		dispersion_input = 'no'
else:
	dispersion_input = 'no'

# set thermostat
if args.thermostat_input:
	thermostat_input = args.thermostat_input
else:
	thermostat_input = 'langevin'

# set t0
if args.t_zero_input:
	t_zero_input = int(args.t_zero_input)
else:
	t_zero_input = 1500

# set lnvtime
if args.lnvtime_input:
	lnvtime_input = int(args.lnvtime_input)
else:
	lnvtime_input = 100

# set convthre
if args.convthre_input:
	convthre_input = args.convthre_input
else:
	convthre_input = '0.005'

# set level shift
if args.level_shift_input:
	if args.level_shift_input == 'no':
		level_shift_input = 'no'
	else:
		level_shift_input = 'yes'
else:
	level_shift_input = 'yes'

# set level shift values a and b
level_shift_vala_input = '0.3'
level_shift_valb_input = '0.1'

if args.level_shift_vala_input:
	if level_shift_input == 'no':
		print "levelshiftvala was provided when levelshift = no"
		sys.exit(2)
	else:
		level_shift_vala_input = args.level_shift_vala_input


if args.level_shift_valb_input:
	if level_shift_input == 'no':
		print "levelshiftvalb was provided when levelshift = no"
		sys.exit(3)
	else:
		level_shift_valb_input = args.level_shift_valb_input

# set mixguess
if args.mix_guess_input:
	mix_guess_input = args.mix_guess_input
else:
	mix_guess_input = '0'

# set integrator
if args.integrator_input:
	integrator_input = args.integrator_input
else:
	integrator_input = 'regular'

# set run type
if args.run_input:
	run_input = args.run_input
else:
	run_input = 'md'

# set method
if args.method_input:
	method_input = args.method_input
else:
	method_input = 'uhf'

# set scf
if args.scf_input:
	scf_input = args.scf_input
else:
	scf_input = 'diis+a'

# set timings
if args.timings_input:
	if args.timings_input== 'no':
		timings_input = 'no'
	else:
		timings_input = 'yes'
else:
	timings_input = 'yes'

# set nsteps
if args.nsteps_input:
	nsteps_input = int(args.nsteps_input)
else:
	nsteps_input = 5000

# set gpus
if args.gpus_input:
	gpus_input = int(args.gpus_input)
else:
	gpus_input = 4

# set maxit
if args.maxit_input:
	maxit_input = int(args.maxit_input)
else:
	maxit_input = 300

# set mdbc - molecular dynamics boundary conditions
if args.mdbc_input:
	mdbc_input = args.mdbc_input
else:
	mdbc_input = 'spherical'

# set mdbc_hydrogen
if args.mdbc_hydrogen_input:
	if args.mdbc_hydrogen_input== 'no':
		mdbc_hydrogen_input = 'no'
	else:
		mdbc_hydrogen_input = 'yes'
else:
	mdbc_hydrogen_input = 'yes'

# set mdr2_count_input
if args.mdr2_count_input:
	mdr2_count_input = int(args.mdr2_count_input)
else:
	mdr2_count_input = 1 

# set mdr2_factor
if args.mdr2_factor:
	mdr2_factor = float(args.mdr2_factor)
else:
	mdr2_factor = 0.03

# set mdk2_count_input
if args.mdk2_count_input:
	if args.mdk2_count_input > 0 and args.mdk2_count_input < 21:
		mdk2_count_input = int(args.mdk2_count_input)
	else:
		mdk2_count_input = 1
else:
	mdk2_count_input = 1 

# set mdbc_mass_scaled
if args.mdbc_mass_scaled_input:
	if args.mdbc_mass_scaled_input == 'no':
		ynlist = ['no']
	elif args.mdbc_mass_scaled_input == 'both':
		dispersion_input = ynlist = ['yes','no']
	else:
		ynlist = ['yes']
else:
	ynlist = ['yes']


# set name
if len(args.job_name) > 6:
	job_name = args.job_name[:6]
else:
	job_name = args.job_name

# set queue system
if args.queue_system == 'sge':
	queue_system = 'sge'
elif args.queue_system == 'slurm':
	queue_system = 'slurm'
else:
	print "queue system must be sge or slurm"	
	sys.exit(4)

# set memory
if args.memory_run:
	memory_run = int(args.memory_run)
else:
	memory_run = 5

# set time for job
if args.time_run:
	time_run = args.time_run
else:
	time_run = '35:59:00'

# set memory
if args.ntasks_run:
	ntasks_run = int(args.ntasks_run)
else:
	ntasks_run = 1

# read in XYZ file and estimate total volume of atoms in input

if args.coordinate_filename:
    min_inner_radius = ipg.findminradius(args.coordinate_filename)
    inner_radius = ipg.findinnerradius(min_inner_radius, inner_volume_ratio)
    outer_radius = ipg.findouterradius(inner_radius,outer_volume_ratio)
    print "Recommended inner radius is " + str(inner_radius)
    print "Minimum inner radius is " + str(min_inner_radius)
    print "Recommended outer radius is " + str(outer_radius)

# need to generate mdr2list and mdk2list based on count given by user at command line

mdr2list = ipg.mdr2listgenerator(mdr2_count_input,inner_radius,min_inner_radius)

mdk2list = ipg.mdk2listgenerator(mdk2_count_input)

# move to folder where coordinates are located
# make a file to store which input file corresponds to which parameters
subprocess.call(['mkdir',args.output_folder])
log_file = open(args.output_folder +'/'+ job_name+'.log','w')

mdr1 = outer_radius
mdk1 = 3.0

# loop over values of r2, k2, and whether or not to mass scale the boundary condition
file_index = 0
for mdr2 in mdr2list:
	for mdk2 in mdk2list:
		for yn in ynlist:
			# make an input file
			filename = job_name+str(file_index)
			subprocess.call(['mkdir',args.output_folder+'/'+filename])
			isg.generateinputfile(args.output_folder, filename, args.coordinate_filename, charge_input, basis_set_input, 
			t_initial_input, dispersion_input, thermostat_input, t_zero_input, lnvtime_input, convthre_input,
			level_shift_input, level_shift_vala_input, level_shift_valb_input, mix_guess_input, integrator_input,
			run_input, method_input, scf_input, timings_input, nsteps_input, gpus_input, maxit_input,
			mdbc_input, mdr1, mdk1, mdr2, mdk2, mdbc_hydrogen_input, yn)
			
			# write to the log  
			# TODO fix this --- database???
			log_file.write(filename+', r2 = '+ str("{:6.2f}".format(mdr2))+ ', k2 = '+ str("{:3.1f}".format(mdk2)) + ', mdbc_h_scaled = '+ yn+ '\n')
			
			# make the run.sh file
			if queue_system == 'sge':
				isg.generatesgerunfile(args.output_folder, filename, gpus_input, memory_run, time_run)
			elif queue_system == 'slurm':
				isg.generateslurmrunfile(args.output_folder, filename, ntasks_run, time_run, memory_run, gpus_input)

			# submit job if user flagged -s
			if args.submit_run:
				if queue_system == 'sge':
					isg.submitsgejob(args.coordinate_filename, args.output_folder, filename)
				elif queue_system == 'slurm':
					isg.submitslurmjob(args.coordinate_filename, args.output_folder, filename)
			file_index += 1

log_file.close()

print "Generated " + str(file_index) + " input and run files in " + args.output_folder

if args.submit_run:
	print "Submitted " + str(file_index) + " jobs"



