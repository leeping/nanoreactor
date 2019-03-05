# input submission generation module for nanoreactor
# Author Laszlo R Seress - seress@stanford.edu

# Version History: 

# 1.0 - 1/5/16
# Initial creation - generation of input and run.sh files, submission of sge and slurm jobs
# TODO: interface with database/log file to track which job is which

import os

def generateinputfile(output_folder, filename, coordinate_filename, charge_input, basis_set_input, 
	t_initial_input, dispersion_input, thermostat_input, t_zero_input, lnvtime_input, convthre_input,
	level_shift_input, level_shift_vala_input, level_shift_valb_input, mix_guess_input, integrator_input,
	run_input, method_input, scf_input, timings_input, nsteps_input, gpus_input, maxit_input,
	mdbc_input, mdr1, mdk1, mdr2, mdk2, mdbc_hydrogen_input, yn):
	current_file = open(output_folder+'/'+filename +'/input','w')
	current_file.write('coordinates ' + coordinate_filename+'\n')
	current_file.write('charge      ' + str(charge_input)+'\n')
	current_file.write('basis       ' + basis_set_input+'\n')
	current_file.write('tinit       ' + str(t_initial_input)+'\n')
	current_file.write('dispersion  ' + dispersion_input+'\n')
	current_file.write('thermostat  ' + thermostat_input+'\n')
	current_file.write('t0          ' + str(t_zero_input)+'\n')
	current_file.write('lnvtime     ' + str(lnvtime_input)+'\n')
	current_file.write('convthre    ' + convthre_input+'\n')
	current_file.write('levelshift  ' + level_shift_input+'\n')
	if level_shift_input == 'yes':
		current_file.write('levelshiftvala ' + str(level_shift_vala_input)+'\n')
		current_file.write('levelshiftvalb ' + str(level_shift_valb_input)+'\n')
	current_file.write('mixguess    ' + mix_guess_input+'\n')
	current_file.write('integrator  ' + integrator_input+'\n')
	current_file.write('\n')
	current_file.write('run         ' + run_input+'\n')
	current_file.write('method      ' + method_input+'\n')
	current_file.write('scf	        ' + scf_input+'\n')
	current_file.write('\n')
	current_file.write('timings     ' + timings_input+'\n')
	current_file.write('nstep       ' + str(nsteps_input)+'\n')
	current_file.write('gpus        ' + str(gpus_input)+'\n')
	current_file.write('maxit       ' + str(maxit_input)+'\n')
	current_file.write('\n')
	current_file.write('mdbc                 '+ mdbc_input+'\n')
	current_file.write('md_r1                    ' + str("{:6.2f}".format(mdr1))+'\n')
	current_file.write('md_k1                      '+ str("{:3.1f}".format(mdk1))+'\n')
	current_file.write('md_r2                    ' + str("{:6.2f}".format(mdr2))+'\n')
	current_file.write('md_k2                      '+ str("{:3.1f}".format(mdk2))+'\n')
	current_file.write('mdbc_hydrogen              '+ mdbc_hydrogen_input +'\n')
	current_file.write('mdbc_mass_scaled           '+ yn +'\n')
	current_file.write('mdbc_t1                    500' +'\n')
	current_file.write('mdbc_t2                    150' +'\n')
	current_file.write('end\n')
	current_file.close()
	return

def generatesgerunfile(output_folder, filename, gpus_input, memory_run, time_run):
	run_file = open(output_folder+'/'+filename +'/'+ 'run.sh','w')
	run_file.write('#!/bin/bash\n')
	run_file.write('\n')
	run_file.write('dir='+output_folder+'/'+filename+'\n')
	run_file.write('\n')
	run_file.write('module load cuda\n')
	run_file.write('module load sge\n')
	run_file.write('\n')
	run_file.write('#$ -pe smp 1\n')
	run_file.write('#$ -l gpus='+str(gpus_input) +'\n')
	run_file.write('#$ -l h_rss='+str(memory_run)+'G' +'\n')
	run_file.write('#$ -l h_rt='+ time_run +'\n')
	#run_file.write('#$ -o '+args.output_folder+'/'+filename+ '/job.out' +'\n')
	#run_file.write('#$ -e '+args.output_folder+'/'+filename+ '/job.err' +'\n')
	run_file.write('#$ -j y\n')
	run_file.write('#$ -N '+ filename + '\n')
	run_file.write('\n')
	run_file.write('cp ${dir}/input .\n')		
	run_file.write('cp ${dir}/*xyz .\n') 
	run_file.write('\n')
	run_file.write('$Terachemloc/terachem input > out \n')
	run_file.write('\n')						
	run_file.write('cp -r * ${dir}\n')
	run_file.close()
	return


def generateslurmrunfile(output_folder, filename, ntasks_run, time_run, memory_run, gpus_input):
	run_file = open(output_folder+'/'+filename +'/'+ 'run.sh','w')
	run_file.write('#!/bin/bash\n')
	run_file.write('\n')
	run_file.write('#SBATCH --job-name=' + filename + '\n')
	run_file.write('#SBATCH --output='+ filename +'.log\n')
	run_file.write('#\n')
	run_file.write('#SBATCH --ntasks='+ str(ntasks_run) +'\n')
	run_file.write('#SBATCH --time=' + time_run + '\n')
	run_file.write('#SBATCH --mem='+ str(memory_run*1024)+'\n')
	run_file.write('#SBATCH --gres gpu:'+str(gpus_input) +'\n')
	run_file.write('\n')
	run_file.write('srun echo Path location:\n')
	run_file.write('srun pwd\n')
	run_file.write('srun echo Files:\n')
	run_file.write('srun ls\n')
	run_file.write('srun echo GPU info\n')
	run_file.write('srun /usr/bin/nvidia-smi\n')
	run_file.write('srun echo Running TeraChem '+ filename +'\n')
	run_file.write('srun -n 1 -c '+ str(gpus_input) + ' $TeraChem/bin/terachem input > out \n')
	run_file.close()	
	return

def submitsgejob(coordinate_filename, output_folder, filename):
    os.chdir(output_folder+'/'+filename)
    # coordinate_filename needs to be the full path to the coordinate file including the file itself
    os.system('cp ' + coordinate_filename + ' ' + output_folder+'/'+filename)
    os.system("qsub < "+ output_folder+'/'+filename +'/'+ 'run.sh')
    return

def submitslurmjob(coordinate_filename, output_folder, filename):
    os.chdir(output_folder+'/'+filename)
    # coordinate_filename needs to be the full path to the coordinate file including the file itself
    os.system('cp ' + coordinate_filename + ' ' + output_folder+'/'+filename)
    os.system("sbatch run.sh")
    return
