#===========================================#
#|    Utility functions and classes for    |#
#|     managing database of reactions      |#
#|  Authors: Lee-Ping Wang, Leah Bendavid  |#
#===========================================#
from __future__ import print_function
import os, sys, re, shutil, time
import numpy as np
import argparse
import traceback
from copy import deepcopy
from collections import Counter, OrderedDict
import subprocess
from .molecule import Molecule, TopEqual, MolEqual, Elements, extract_int, arc, EqualSpacing
from .nifty import _exec, natural_sort, extract_tar
from .output import logger
# json is used for saving dictionaries to file.
import json
try:
    import work_queue
    work_queue.set_debug_flag('all')
except:
    work_queue = None
    logger.warning("Failed to import Work Queue library")

# Global variable for the Work Queue
WQ = None

def create_work_queue(port):
    global WQ
    # Near clone of the function in nifty.py
    if work_queue == None:
        logger.error("Cannot create Work Queue because work_queue module not found")
        raise RuntimeError
    WQ = work_queue.WorkQueue(port=port, shutdown=False)
    WQ.specify_name('refine')
    WQ.specify_keepalive_interval(8640000)
    # Calculations submitted first get run first
    # Specify LIFO instead of FIFO for last-in-first-out
    WQ.specify_task_order(work_queue.WORK_QUEUE_TASK_ORDER_FIFO)
    logger.info('Work Queue listening on %d' % (WQ.port))

def get_trajectory_home(pth):
    """ Get a home directory for the dynamics trajectory file. """
    if os.getcwd() not in os.path.abspath(pth):
        logger.error('Cannot run refinement on a file outside of the current folder')
        raise RuntimeError
    # First replace the top level directory name
    pth = pth.replace('Nanoreactor', 'Refinement')
    # Then strip the file extension
    pth = pth.replace('.xyz', '')
    # Then get rid of the word Consolidated
    pth = pth.replace('Consolidated', '')
    # Strip all leading and trailing slashes
    pth = pth.strip('/')
    # Get the absolute path
    pth = os.path.abspath(os.path.join(os.getcwd(), pth))
    return pth

def check_xyz(pth, start=''):
    """
    Determine whether a path corresponds to a valid .xyz file for refinement.
    We don't check the file content, simply the name and extension.  Will throw
    an exception if the path doesn't exist.
    
    Parameters
    ----------
    pth : string
        Input path
    start : string
        Filter for the file name start

    Returns
    -------
    bool
        True if path is valid 
    """
    if not os.path.exists(pth):
        logger.error('Path %s does not exist' % pth)
        raise RuntimeError
    if os.path.isdir(pth):
        return False
    if os.path.basename(pth).startswith(start) and pth.endswith('.xyz'):
        return True
    return False

def find_groups(sl1, sl2):
    """ 
    Given two lists of atom lists, find the groups of sets in each
    list that only contain each others' elements (i.e. if somehow we
    have two parallel reactions in one.)
    
    Parameters
    ----------
    sl1, sl2 : list of lists
        List of lists of atom indices corresponding to molecules.

    Returns
    -------
    list of lists
        Atom indices corresponding to separately reacting groups of molecules.
    """
    # Convert to sets
    sl1c = [set(s) for s in sl1]
    sl2c = [set(s) for s in sl2]
    # Iterate this while loop until we find a single set of atom groups
    while set([tuple(sorted(list(s))) for s in sl1c]) != set([tuple(sorted(list(s))) for s in sl2c]):
        # Double loop over molecule atom indices
        for s1 in sl1c:
            for s2 in sl2c:
                # For any pair of molecules that have any overlapping atoms,
                # add all atoms in each atom set to the other.
                if len(s1.intersection(s2)) > 0:
                    s1.update(s2)
                    s2.update(s1)
    result = sorted([list(t) for t in list(set([tuple(sorted(list(s))) for s in sl1c]))])
    return result

def find_reacting_groups(m1, m2):
    """
    Given two Molecule objects, determine the groups of atoms that
    reacted with each other (i.e. formed different molecules.)  This will
    remove spectator atoms (ones that didn't react at all) and separate 
    concurrent reactions occuring in different places.

    Parameters
    ----------
    m1, m2 : Molecule
        Length-1 Molecule objects corresponding to reactant and product
        frames.  For the sake of future electronic structure calculations,
        these objects must have qm_mulliken_charges and qm_mulliken_spins.

    Returns
    -------
    extracts: list of 3-tuple of list, int, int
        Each 3-tuple is a group of atoms that reacted, and their associated
        charge / multiplicity.
    """
    if not isinstance(m1, Molecule) or len(m1) != 1:
        logger.error("Please only pass length-1 Molecule objects")
        raise RuntimeError
    if not isinstance(m2, Molecule) or len(m2) != 1:
        logger.error("Please only pass length-1 Molecule objects")
        raise RuntimeError

    # Get a list of atom indices belonging to each molecule.
    m1_mol_atoms = [g.L() for g in m1.molecules]
    m2_mol_atoms = [g.L() for g in m2.molecules]
    # Count the number of atoms in spectator molecules that don't
    # react at all, and store their molecular formulas (for public
    # shaming).
    n_spectator_atoms = 0
    spectator_formulas = []
    strrxns = []
    # The results: extract groups of atoms to extract corresponding to
    # individual reaction pathways, and the net charge / multiplicity
    # belonging to 
    extract_groups = []
    extract_charges = []
    extract_mults = []
    # Separate atoms into groups of separately reacting molecules.
    # This is also effective at finding spectator molecules that don't react at all.
    do_extract = False
    for atom_group in find_groups(m1_mol_atoms, m2_mol_atoms):
        m1g = m1.atom_select(atom_group)
        m2g = m2.atom_select(atom_group)
        spectator_atoms = []
        logger.info("atom group: %s" % str(atom_group), printlvl=5)
        logger.info("m1 molecules: %s" % str([[atom_group[i] for i in g.L()] for g in m1g.molecules]), printlvl=5)
        logger.info("m2 molecules: %s" % str([[atom_group[i] for i in g.L()] for g in m2g.molecules]), printlvl=5)
        for g1 in m1g.molecules:
            # logger.info("atoms in molecule: %s" % str(g1.L()), printlvl=5)
            for g2 in m2g.molecules:
                # Graphs are usually compared by comparing elements and
                # topology, but a spectator molecule also has the same
                # atom numbers.
                if g1 == g2 and g1.L() == g2.L():
                    spectator_atoms += g1.L()
        # Since we already separated the atoms into groups of separately reacting ones,
        # any atom group with spectator atoms is expected to be a single spectator molecule.
        if len(spectator_atoms) > 0:
            logger.info("spectator atoms: %s" % str([atom_group[i] for i in spectator_atoms]), printlvl=5)
        if len(spectator_atoms) == m1g.na:
            if len(m1g.molecules) != 1:
                logger.error("I expected an atom group with all spectators to be a single molecule")
                raise RuntimeError
            n_spectator_atoms += len(spectator_atoms)
            spectator_formulas.append(m1g.molecules[0].ef())
            continue
        elif len(spectator_atoms) > 0:
            logger.error("I expected an atom group with any spectators to be a single molecule")
            raise RuntimeError
        else:
            strrxn = ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter([m.ef() for m in m1g.molecules]).items())])
            strrxn += ' -> '
            strrxn += ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter([m.ef() for m in m2g.molecules]).items())])
            strrxns.append(strrxn)
            
        # Now we have a group of reacting atoms that we can extract from the
        # pathway, but we should perform some sanity checks first.
        mjoin = m1g + m2g
        # A bit of code copied from extract_pop.  Verify that the reacting
        # atoms have consistent charge and spin in the two passed Molecule
        # objects.  If not consistent, then we cannot extract spectator atoms.
        Chgs = np.array([sum(i) for i in mjoin.qm_mulliken_charges])
        SpnZs = np.array([sum(i) for i in mjoin.qm_mulliken_spins])
        chg, chgpass = extract_int(Chgs, 0.3, 1.0, label="charge")
        spn, spnpass = extract_int(abs(SpnZs), 0.3, 1.0, label="spin-z")
        nproton = sum([Elements.index(i) for i in m1g.elem])
        nelectron = nproton + chg
        # If the sanity checks fail, then do not extract the spectator atoms
        # and simply return a list of all the atoms at the end.
        do_extract = True
        if ((nelectron-spn)//2)*2 != (nelectron-spn):
            logger.info("\x1b[91mThe number of electrons (%i; charge %i) is inconsistent with the spin-z (%i)\x1b[0m" % (nelectron, chg, spn), printlvl=1)
            do_extract = False
            break
        if (not chgpass or not spnpass):
            logger.info("\x1b[91mCannot determine a consistent set of spins/charges after extracting spectators\x1b[0m", printlvl=1)
            do_extract = False
            break
        extract_groups.append(np.array(atom_group))
        extract_charges.append(chg)
        extract_mults.append(abs(spn)+1)
    if do_extract:
        message = "Initial Reaction : " + ' ; '.join(strrxns)
        if n_spectator_atoms > 0:
            # I know it's supposed to be spelled 'spectator', but it's fun to say 'speculator' :)
            message += " ; Speculators (removed) : \x1b[91m%s\x1b[0m" % (' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(spectator_formulas).items())]))
        logger.info(message, printlvl=2)
        return list(zip(extract_groups, extract_charges, extract_mults))
    else:
        logger.info("Unable to split reaction pathway into groups")
        return list(zip([np.arange(m1.na)], [m1.charge], [m1.mult]))

def analyze_path(xyz, nrg, cwd, xyz0=None, label="Reaction", draw=2):
    """
    Analyze the results of a reaction path.
    
    Parameters
    ----------
    xyz : str 
        .xyz file name containing coordinates of path.
    nrg : str
        Two column file containing path parameterization and energies in kcal/mol.
    cwd : str
        Directory containing the files.
    xyz0 : str
        Coordinates of initial path in .xyz format (for checking if we've got the correct molecules).
    label : str
        Label to attach to printout.
    draw : int
        If 0, then don't draw anything.
        If 1, then draw the summary PDF if the reaction is correct.
        If 2, then draw the summary PDF even if the reaction is incorrect.
        If 3, then redraw the PDF even if it exists on disk.

    Returns
    -------
    status : str
        Final status of the path (correct or incorrect).
    fwd : bool
        Whether the reaction was reversed.
    """
    fwd = True
    xyz = os.path.join(cwd, xyz)
    nrg = os.path.join(cwd, nrg)
    path = Molecule(xyz, ftype='xyz')
    nrgarr = np.loadtxt(nrg)
    # If there's only one structure in the IRC then it's obviously wrong.
    if nrgarr.ndim != 2:
        return 'incorrect', True
    energy = nrgarr[:, 1]
    # Rebuild the topology object for the initial and final frame.
    pathR = path[0]
    pathP = path[-1]
    pathR.build_topology()
    pathP.build_topology()
    status = ''
    message = ''
    if xyz0 != None:
        xyz0 = os.path.join(cwd, xyz0)
        init = Molecule(xyz0, ftype='xyz')
        initR = init[0]
        initP = init[-1]
        initR.build_topology()
        initP.build_topology()
        if TopEqual(pathR, initR) and TopEqual(pathP, initP):
            # The perfect result, the reactants and products of the
            # path reflect the initial path.
            status = 'correct'
            message = 'perfect'
        elif TopEqual(pathR, initP) and TopEqual(pathP, initR):
            # The reactants and products of the path are the initial path
            # in reverse order.
            status = 'correct'
            message = 'reversed'
            fwd = False
            path = path[::-1]
            energy = energy[::-1]
        elif TopEqual(pathR, pathP):
            # If the topologies are completely equal then there wasn't a reaction at all.
            status = 'incorrect'
            message = 'no reaction'
            if draw < 3: draw = 0
        elif MolEqual(pathR, initR) and MolEqual(pathP, initP):
            # The path and initial path have matching reactant and product 
            # molecules but the atoms in molecules (AIM) are different.
            status = 'correct'
            message = 'different AIM'
        elif MolEqual(pathR, initP) and MolEqual(pathP, initR):
            # Same as above, but atoms are in reverse order.
            status = 'correct'
            message = 'reversed, different AIM'
            fwd = False
            path = path[::-1]
            energy = energy[::-1]
        else:
            status = 'incorrect'
            if MolEqual(pathR, pathP):
                # If the molecules haven't changed then it's a trivial rearrangement.
                message = 'same molecules'
                if draw < 3: draw = 0
            elif any([MolEqual(pathR, initR), MolEqual(pathP, initP)]):
                message = 'matched one side'
            elif any([MolEqual(pathR, initP), MolEqual(pathP, initR)]):
                message = 'matched one side (reversed)'
                fwd = False
                path = path[::-1]
                energy = energy[::-1]
            else:
                message = 'all different'
    # Ensure that the reaction energy starts at zero
    energy -= energy[0]
    # Reaction energy
    DE = energy[-1] - energy[0]
    # Activation energy
    Ea = max(energy)
    # Loop through and identify reactant, product and spectator molecular formulas.
    formulaR = []
    formulaP = []
    formulaS = []
    for m in pathR.molecules:
        if any([(m == m_ and m.L() == m_.L()) for m_ in pathP.molecules]):
            formulaS.append(m.ef())
        else:
            formulaR.append(m.ef())
    for m in pathP.molecules:
        if any([(m == m_ and m.L() == m_.L()) for m_ in pathR.molecules]):
            pass
        else:
            formulaP.append(m.ef())
    if len(formulaR) > 0:
        if len(formulaP) == 0:
            logger.error('How can I have reactants but no products?')
            raise RuntimeError
        strrxn = 'Reaction: ' + ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(formulaR).items())])
        strrxn += ' -> '
        strrxn += ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(formulaP).items())])
    else:
        strrxn = 'Reaction: None'
    if len(formulaS) > 0:
        strrxn += ', Speculators: '
        strrxn += ' '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(formulaS).items())])
    if status == 'correct':
        color = '\x1b[1;92m'
    elif status == 'incorrect':
        color = '\x1b[1;93m'
    # Print summary information and draw the summary PDF.
    logger.info('=> Result: %s --%s-- \x1b[0m (%s)' % (color, status, message), printlvl=1)
    logger.info('=> ' + strrxn, printlvl=1)
    logger.info('=> (Electronic energy only) DE = % .4f Ea = %.4f (kcal/mol)' % (DE, Ea))
    if draw == 2: draw = (not os.path.exists('%s/reaction.pdf' % cwd))
    elif draw == 1: draw = (status == 'correct') and (not os.path.exists('%s/reaction.pdf' % cwd))
    if draw < 3: draw = (draw and not MolEqual(pathR, pathP))
    if draw:
        owd = os.getcwd()
        os.chdir(cwd)
        os.system('draw-reaction.py irc.xyz irc.pop irc.nrg')
        os.chdir(owd)
    if os.path.exists('%s/reaction.pdf' % cwd):
        logger.info('=> Summary: %s/reaction.pdf' % cwd)
    return status, fwd

def parse_irc_error(log):
    # Parse log file for error messages.
    errmsg = []
    for line in open(log).readlines():
        if 'Forward direction:' in line and 'Ok' not in line:
            errmsg.append('Fwd: %s' % line.replace('Forward direction:','').strip())
        if 'IRC calculation failed in forward direction, error =' in line:
            errmsg.append('Fwd: %s ' % line.replace('IRC calculation failed in forward direction, error =','').strip())
        if 'Backward direction:' in line and 'Ok' not in line:
            errmsg.append('Bak: %s' % line.replace('Backward direction:','').strip())
        if 'IRC calculation failed in backward direction, error =' in line:
            errmsg.append('Bak: %s ' % line.replace('IRC calculation failed in backward direction, error =','').strip())
        if 'Segments are too short' in line:
            errmsg.append('IRC too short')
        if 'Calculation encountered a fatal error!' in line:
            errmsg.append(line.split('error!')[1].strip().replace(')','').replace('(',''))
    if not os.path.exists(os.path.join(os.path.dirname(log), 'irc.xyz')):
        errmsg.append("IRC coordinates do not exist")
    return errmsg

def pid_table():
    """ Return a list of currently running process IDs. Limited to run once per ten seconds. """
    if (pid_table.pids == None or (time.time() - pid_table.t0) > 10.0):
        pid_table.pids = [int(i.strip()) for i in os.popen("ps ef | awk '/^ *[0-9]/ {print $1}'").readlines()]
        pid_table.t0 = time.time()
    return pid_table.pids
pid_table.t0 = time.time()
pid_table.pids = None

def parse_input_files(inps):
    """
    Recursive function for determining input .xyz files at start of refinement.
    This makes it convenient for the user to start a calculation.  The inputs
    are .xyz file names, directories containing .xyz files, or files containing 
    lists of .xyz files, directories, or other lists.

    In the case of a .xyz file name, it's simply added to the list.  In the case
    of a directory, the files inside starting with 'reaction_' and ending with '.xyz'
    are added to the list.  Any other file is assumed to be a "list file", and this 
    function will be recursively called for each line in the file.  If any path
    doesn't exist in the list file, it will crash.

    Note that we don't recursively go into directories.
 
    Parameters
    ----------
    inps : str or list of str
        Path or list of paths.  Paths may be folders, .xyz file, or "input file" 
        containing list of such.
    
    Returns
    -------
    fnms : list of str
        List of paths of .xyz files.
    """
    # List of .xyz file names
    fnms = []
    # Simple type checking
    if isinstance(inps, str):
        inps = [inps]
    # Iterate over input paths
    for inp in inps:
        # Remove comment lines
        inp = inp.split("#")[0].strip()
        if len(inp) == 0: continue
        if os.path.isdir(inp):
            nfid = 0
            # If a directory, loop over the contents
            for fnm in natural_sort(os.listdir(inp)):
                absfnm = os.path.join(inp, fnm)
                # .xyz files included from directories must start with "reaction_"
                if check_xyz(absfnm, start='reaction_'):
                    fnms.append(absfnm)
                    nfid += 1
            logger.info("%s contains %i dynamics trajectories" % (inp, nfid), printlvl=1)
        elif os.path.isfile(inp):
            if check_xyz(inp):
                # Explicitly included .xyz files may have any name
                logger.info("%s is an individual trajectory" % inp, printlvl=1)
                fnms.append(inp)
            else:
                # Any other file is assumed to be a list of references to other .xyz files / folders / lists
                logger.info("%s contains %i references" % (inp, len(open(inp).readlines())), printlvl=1)
                fnms += parse_input_files([l.strip() for l in open(inp).readlines()])
        else:
            logger.error('File %s does not exist' % inp)
            raise RuntimeError
    return fnms

def wq_reactor(wait_time=1, newline_time=3600, success_time=3600, iters=np.inf):
    """ 
    Reactor Loop: Waits for tasks to finish in the Work Queue and
    executes follow-up functions if necessary.  When running in Work Queue mode, this 

    Parameters
    ----------
    wait_time : int
        How long to wait in each cycle of the reactor loop.
    newline_time : int
        Time interval between printing newlines
    success_time : int
        Minimum length of a job for printing "job finished successfully"
    """
    global WQ
    niter = 0
    while not WQ.empty():
        task = WQ.wait(wait_time)
        niter += 1
        nbusy = WQ.stats.workers_busy
        logger.info("%s : %i/%i workers busy; %i/%i jobs complete\r" % 
                    (time.ctime(), nbusy, (WQ.stats.total_workers_joined - WQ.stats.total_workers_removed),
                     WQ.stats.total_tasks_complete, WQ.stats.total_tasks_dispatched), newline=False)
        if time.time() - wq_reactor.t0 > newline_time:
            wq_reactor.t0 = time.time()
            logger.info('')
        if task:
            exectime = task.cmd_execution_time/1000000
            if task.result != 0:
                oldid = task.id
                oldhost = task.hostname
                taskid = WQ.submit(task)
                if hasattr(task, 'calc'):
                    task.calc.wqids.append(taskid)
                logger.warning("Task '%s' (id %i) failed on host %s (%i seconds), resubmitted:"
                               "id %i" % (task.tag, oldid, oldhost, exectime, taskid))
            else:
                logger.info("Task '%s' (id %i) returned from %s (%i seconds)"
                            % (task.tag, task.id, task.hostname, exectime), printlvl=(1 if exectime > success_time else 2))
                # Launch the next calculation!
                if hasattr(task, 'calc'):
                    task.calc.saveStatus('ready', display=False)
                    task.calc.wqids.remove(task.id)
                    task.calc.launch()
                del task
        elif (niter >= iters): break
    if iters == np.inf:
        logger.info("Reactor loop has no more tasks!")
    else:
        logger.info("\n")
wq_reactor.t0 = time.time()

def make_task(cmd, cwd, inputs=[], outputs=[], tag=None, calc=None, verbose=0, priority=None):
    """ 
    Run a task locally or submit it to the Work Queue. 

    Parameters
    ----------
    cmd : str
        Command to be executed using a system call.
    cwd : str
        Working directory for the calculation.
    inputs : list
        (For WQ) Names of input files inside the working directory, to be sent to the worker
    outputs : list
        (For WQ) Names of output files, to be written back to the working directory
        (Not for WQ) The locally run calculation will have 
    tag : str
        (For WQ) Descriptive name for the task, if None then use the command.
    calc : Calculation
        (For WQ) The calculation object is an attribute of the WQ task object, 
        and will allow a completed task to execute the next step.
    verbose : int
        Print information out to the screen
    priority : int
        The priority of this task when waiting in the queue
    """
    global WQ
    # if calc != None and calc.read_only:
    #     logger.error("I should never get here - make_task called with read enabled")
    #     raise RuntimeError
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]
    # Actually print the command to the output folder. :)
    with open(os.path.join(cwd, 'command.sh'), 'w') as f:
        print(cmd, file=f)
    if WQ != None:
        # Create and submit Work Queue Task object.
        task = work_queue.Task(cmd)
        # The task priority is either an argument or a field of the calculation object
        if priority == None:
            if calc != None:
                priority = calc.priority
            else:
                priority = 0
        task.specify_priority(priority)
        input_paths = [os.path.abspath(os.path.join(cwd, f)) for f in inputs]
        output_paths = [os.path.abspath(os.path.join(cwd, f)) for f in outputs]
        for f in input_paths:
            task.specify_input_file(f, os.path.basename(f), cache=False)
        for f in output_paths:
            task.specify_output_file(f, os.path.basename(f), cache=False)
        task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
        if tag != None:
            task.specify_tag(tag)
        else:
            task.specify_tag(cmd)
        taskid = WQ.submit(task)
        # Keep track of the work queue task IDs belonging to each task
        if calc != None:
            task.calc = calc
            task.calc.wqids.append(taskid)
            task.calc.saveStatus('launch')
        logger.info("\x1b[94mWQ task\x1b[0m '%s'; taskid %i priority %i" % (task.tag, taskid, priority), printlvl=3)
    else:
        # Run the calculation locally.
        if calc != None: calc.saveStatus('launch')
        _exec(cmd, print_command=(verbose>=3), persist=True, cwd=cwd)
        # After executing the task, run launch() again 
        # because launch() is designed to be multi-pass.
        if calc != None: calc.launch()

class Calculation(object):
    """
    Class representing a general refinement calculation in the workflow. 
    """
    calctype = "Calculation"
    statlvl = 1
    def __init__(self, initial, home, **kwargs):
        """
        Initialize the calculation.  This function is intended to be
        called at the end of the constructor of the derived class
        (which simply figures out the home folder).

        Parameters
        ----------
        initial : str or Molecule object
            Initial Molecule object or file name.
        home : str
            Home folder for the calculation.
        fast_restart : bool
            Read calculation status from disk and skip over ones marked as complete, failed, etc.
        parent : Calculation or None
            The calculation may contain a reference to its parent calculation
        charge : int
            Net charge of the atoms in the xyz file.  If provided this takes highest priority.
        mult : int
            Spin multiplicity of the atoms in the xyz file (2*<Sz>+1).  If provided this takes highest priority.

        """
        # Initial Molecule object or file name.
        self.initial = initial
        # Specify and create the home folder.
        self.home = os.path.abspath(home)
        if not os.path.exists(self.home):
            os.makedirs(self.home)
        # Get the calculation name.
        self.name = kwargs.pop('name', self.home.replace(os.getcwd(), '').strip('/'))
        # If we created the object using a file from another folder,
        # write the location of this file to source.txt.
        if isinstance(initial, str) and self.home not in os.path.abspath(initial):
            with open(os.path.join(self.home, 'source.txt'), 'w') as f:
                print(os.path.abspath(initial), file=f)
        # Calculations have the ability to access their parent.
        self.parent = kwargs.pop('parent', None)
        # Set charge and multiplicity.
        self.charge = kwargs.pop('charge', None)
        self.mult = kwargs.pop('mult', None)
        # If set to False, calculations marked as complete will be skipped.
        self.fast_restart = kwargs.get('fast_restart', False)
        # Read calculations only; don't run them.
        self.read_only = kwargs.get('read_only', False)
        # Set the verbosity level.
        self.verbose = kwargs.get('verbose', 0)
        # Set the priority of the calculation in Work Queue.
        self.priority = kwargs.pop('priority', 0)
        # Drawing level (see analyze_path).
        self.draw = kwargs.get('draw', 2)
        # Maximum number of growing string cycles before termination.
        self.gsmax = kwargs.get('gsmax', 600)
        # Number of images in a growing string / NEB calculation.
        self.images = kwargs.get('images', 21)
        # Whether to keep spectators in the pathway.
        self.spectators = kwargs.get('spectators', 0)
        # Whether to include trivial rearrangements (e.g. H3N + H4N+ -> H4N+ + H3N)
        self.trivial = kwargs.get('trivial', 0)
        # The sequential path for running GS and TS calculations looks like this -
        # TS calculations are launched after GS if below a certain threshold:
        # 
        # GS->GS->TS->GS->TS->GS->TS(Conv)
        # 
        # The parallel path for running GS and TS calculations looks like this -
        # typically there are two calculations running in parallel.  This is faster
        # for completing individual pathways but less efficient in general:
        # 
        #      TS  TS  TS(Conv)
        #      /   /   /
        # GS->GS->GS->GS->GS(Ter)
        # 
        self.ts_branch = kwargs.get('ts_branch', 0)
        # Store list of methods and bases for different calculations.
        self.methods = kwargs['methods'][:]
        self.bases = kwargs['bases'][:]
        # Save a list of Work Queue IDs belonging to this calculation.
        self.wqids = []
        # If more methods are provided than bases, then assume the biggest basis 
        # is used for the later calculations (and vice versa).
        if len(self.methods) > len(self.bases):
            self.bases += [self.bases[-1] for i in range(len(self.methods)-len(self.bases))]
        elif len(self.bases) > len(self.methods):
            self.methods += [self.methods[-1] for i in range(len(self.bases)-len(self.methods))]
        # Certain keyword arguments like fast_restart get passed onto calculations
        # that they create.
        self.kwargs = deepcopy(kwargs)
        # Read calculation status from file (if exists).
        self.initStatus()

    def initStatus(self):
        """ 
        Read calculation status from the .status file 
        which lives in the home folder of the calculation.

        Possible states on disk are:
        complete: Calculation is complete, don't descend into this branch unless forced
        continue: Calculation may continue, run "make" on this branch
        busy.12345: Calculation is running on process ID 12345, don't interfere
    
        Variables set here:
        self.status: Internal status of this calculation that affects behavior of action()
        self.message: Optional informative 1-line message 
        """
        self.message = ''
        statpath = os.path.join(self.home, '.status.%s' % self.calctype.lower())
        if os.path.exists(statpath) and os.path.getsize(statpath) > 0:
            statline = open(statpath).readlines()[0].strip()
        else:
            statline = 'ready'
        statword = statline.split(None, 1)[0]
        message = statline.split(None, 1)[1] if len(statline.split()) > 1 else ''
        if statword.startswith('busy.'):
            busypid = int(statword.replace('busy.', ''))
            if busypid in pid_table():
                self.saveStatus('busy')
            else:
                self.saveStatus('ready', display=(self.verbose>=1))
        elif self.fast_restart:
            display = (self.verbose>=3) if statword == 'ready' else 1
            self.saveStatus(statword, message=message, display=display)
        else:
            self.saveStatus('ready', display=(self.verbose>=3))

    def saveStatus(self, status, ansi="\x1b[93m", message=None, display=True, to_disk=True):
        """ 
        Set calculation status and also write to status file 
        which lives in the home folder of the calculation.
        """
        statpath = os.path.join(self.home, '.status.%s' % self.calctype.lower())
        statout = status
        
        if (hasattr(self, 'status') and self.status == status):
            display = False
        self.status = status
        # If calculation is busy, append the pid to the status.
        if status == 'busy':
            statout += '.%i' % os.getpid()
        # Append a status message if desired.
        if message != None:
            self.message = message
            statout += ' ' + message
        # Save status to disk if desired.
        if self.read_only: to_disk = False
        if to_disk:
            with open(statpath, 'w') as f:
                print(statout, file=f)
        # Print status to the terminal.
        if display:
            self.printStatus(ansi=ansi)

    def printStatus(self, ansi="\x1b[93m"):
        """
        Print calculation status to the terminal in a compact and legible way.
        """
        if self.status.lower() == 'launch':
            ansi = "\x1b[1;44;91m"
        elif self.status.lower() in ['converged', 'complete', 'correct']:
            ansi = "\x1b[92m"
        elif self.status.lower() == 'busy':
            ansi = "\x1b[94m"
        elif self.status.lower() == 'failed':
            ansi = "\x1b[91m"
        elif self.status.lower() == 'ready':
            ansi = "\x1b[96m"
        spaces = max(0, (14 - len(self.status)))
        logger.info("%-15s %s%s\x1b[0m%s in %-60s %s" % (self.calctype, ansi, self.status, ' '*spaces, os.path.abspath(self.home).replace(os.getcwd(), '').strip('/'), self.message), printlvl=self.statlvl)
    
    def synchronizeChargeMult(self, M):
        """
        Synchronize the charge and multiplicity for the calculation and Molecule object.
        Operations are resolved in this order:
        
        1) If Calculation object has either charge or mult set to
        None, then get the charge/mult from the Molecule object.
        1a) If Molecule object doesn't have charge or mult, try to read
        it from the comments.  Otherwise crash.
        1b) Set the charge/mult in the Calculation object.
        2) If Calculation object has charge or mult, then set the Molecule
        object to have the same.
        
        Parameters
        ----------
        M : Molecule
            A Molecule object

        """
        # If charge and multiplicity are not explicitly provided,
        # set them by reading the comment strings.
        if self.charge == None or self.mult == None:
            if 'charge' not in M.Data or 'mult' not in M.Data:
                M.read_comm_charge_mult(verbose=(self.verbose>=2))
            self.charge = M.charge
            self.mult = M.mult
        else:
            M.charge = self.charge
            M.mult = self.mult

    def launch(self):
        # Check various statuses and don't continue for certain ones.
        if self.status == 'failed':
            logger.info("%s returning because failed" % self.name, printlvl=3)
            return
        if self.status == 'no-reaction':
            logger.info("%s returning because no reaction" % self.name, printlvl=3)
            return
        # If the status has been set to complete / complete status loaded from disk,
        # then don't descend into this branch.  This saves us from going into
        # folders with completed calculations but it's also kind of a pain point.
        if self.status == 'complete': 
            logger.info("%s returning because complete" % self.name, printlvl=3)
            return
        # This is not really an error, but the growing string / NEB has
        # gone on for so long that we just kill the job.
        if self.status == 'terminated': 
            logger.info("%s returning because terminated" % self.name, printlvl=3)
            return
        # Don't go into launch() for a job that already has a running Work Queue task
        if self.status == 'busy' and not self.read_only:
            logger.info("%s returning because busy" % self.name, printlvl=3)
            return
        # This function is specific to the particular calculation
        self.launch_()
        # If there are Work Queue tasks, mark the calculation as busy.
        if len(self.wqids) > 0:
            self.saveStatus('busy', display=(self.verbose>=4))
    
    def launch_(self):
        raise NotImplementedError

    def Equal(self, m1, m2):
        """ 
        Return whether two molecule objects are "equal".  If we are
        including trivial rearrangements (i.e. self.trivial == True),
        then the comparison will check the atom indices in addition 
        to the elements / connectivity.
        """
        return TopEqual(m1, m2) if self.trivial else MolEqual(m1, m2)

class FragmentID(Calculation):
    """
    Class representing the identification of the fragment molecules in a reaction.
    """
    calctype = "FragmentID"
    statlvl = 2

    def __init__(self, initial, home, **kwargs):
        super(FragmentID, self).__init__(initial, home, **kwargs)
        # Failure counter.  When this hits three, the calculation deletes itself :P
        self.fails = 0

    def launch_(self):
        """
        Launch a fragment identification calculation.
        """
        # Check to see if part 1 of fragment optimization has already been completed
        extract_tar(os.path.join(self.home, 'fragmentid.tar.bz2'), ['fragmentid.txt'])
        if os.path.exists(os.path.join(self.home, 'fragmentid.txt')):
            if hasattr(self.parent, 'countFragmentIDs'):
                complete, total = self.parent.countFragmentIDs()
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
            else:
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False)
            # Once ANY fragment optimization job is finished, we pass through the parent object again.
            self.parent.launch()
            return
        elif os.path.exists(os.path.join(self.home, 'fragmentid.log')):
            self.fails += 1
            logger.info("%s has fragmentid.log but not fragmentid.txt - it may have failed (%i tries)" % (self.name, self.fails))
            shutil.move(os.path.join(self.home, 'fragmentid.log'), os.path.join(self.home, 'fragmentid.%i.log' % self.fails))
            # Number of attempts set to infinity, but really I should be parsing 
            # the output to see what is failing.
            if self.fails >= np.inf or self.read_only:
                self.saveStatus('failed', to_disk=False, message='gave up after %i tries' % self.fails)
                self.parent.launch()
            else:
                self.launch()
            return
        if self.read_only: return
        # Otherwise we run the fragment identification calculation.
        # Ensure that this calculation contains a Molecule object.
        if not isinstance(self.initial, Molecule):
            M = Molecule(self.initial)
        else:
            M = deepcopy(self.initial)
        self.synchronizeChargeMult(M)
        if self.charge == -999:
            self.saveStatus('failed', message='Charge and spin inconsistent')
            return
        if len(M) != 1:
            logger.error("Fragment identification can only handle length-1 Molecule objects")
            raise RuntimeError
        M.write(os.path.join(self.home, 'initial.xyz'))
        # Note that the "first" method and basis set is used for fragment identification.
        make_task("identify-fragments.py initial.xyz --method %s --basis \"%s\" --charge %i --mult %i &> fragmentid.log" % 
                  (self.methods[0], self.bases[0], self.charge, self.mult), 
                  self.home, inputs=["initial.xyz"], outputs=["fragmentid.log", "fragmentid.tar.bz2"], 
                  tag=self.name, calc=self, verbose=self.verbose)

class FragmentOpt(Calculation):
    """
    Class that optimizes the fragments of the reaction.
    """
    calctype = "FragmentOpt"
    statlvl = 2

    def __init__(self, initial, home, **kwargs):
        super(FragmentOpt, self).__init__(initial, home, **kwargs)
        # Failure counter.  When this hits three, the calculation deletes itself :P
        self.fails = 0
    
    def launch_(self):
        """
        Launch a fragment optimization calculation.
        """
        # Check to see if fragment optimization has already been completed
        # Need to change this since we're splitting it into two parts of the calculation.
        extract_tar(os.path.join(self.home, 'fragmentopt.tar.bz2'), ['fragmentopt.xyz', 'fragmentopt.nrg'])
        if os.path.exists(os.path.join(self.home, 'fragmentopt.nrg')):
            if hasattr(self.parent, 'countFragmentOpts'):
                complete, total = self.parent.countFragmentOpts()
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
            else:
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False)
            # Once ANY fragment optimization job is finished, we pass through the parent object again.
            self.parent.launch()
            return
        elif os.path.exists(os.path.join(self.home, 'fragmentopt.log')):
            self.fails += 1
            logger.info("%s has fragmentopt.log but not fragmentopt.nrg - it may have failed (%i tries)" % (self.name, self.fails))
            shutil.move(os.path.join(self.home, 'fragmentopt.log'), os.path.join(self.home, 'fragmentopt.%i.log' % self.fails))
            # Number of attempts set to infinity, but really I should be parsing 
            # the output to see what is failing.
            if self.fails >= np.inf or self.read_only:
                self.saveStatus('failed', to_disk=False, message='gave up after %i tries' % self.fails)
                self.parent.launch()
            else:
                self.launch()
            return
        if self.read_only: return
        # Note that the "last" method and basis set is used for the fragment optimization
        make_task("optimize-fragments.py --method %s --basis \"%s\" &> fragmentopt.log" % 
                  (self.methods[-1], self.bases[-1]), 
                  self.home, outputs=["fragmentopt.log", "fragmentopt.tar.bz2"], 
                  tag=self.name, calc=self, verbose=self.verbose)



class Optimization(Calculation):
    """
    Class representing a geometry optimization.
    """
    calctype = "Optimization"
    statlvl = 2

    def __init__(self, initial, home, **kwargs):
        super(Optimization, self).__init__(initial, home, **kwargs)
        # Failure counter.  When this hits three, the calculation deletes itself :P
        self.fails = 0

    def launch_(self):
        """
        Launch an optimization calculation.
        """
        # If this method is called and optimize.xyz exists, that means
        # the optimization is completed!
        extract_tar(os.path.join(self.home, 'optimize.tar.bz2'), ['optimize.xyz', 'optimize.pop'])
        if os.path.exists(os.path.join(self.home, 'optimize.xyz')):
            if hasattr(self.parent, 'countOptimizations'):
                complete, total = self.parent.countOptimizations()
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
            else:
                self.saveStatus('converged', display=(self.verbose>=2), to_disk=False)
            # Once ANY optimization job is finished, we pass through the parent object again.
            self.parent.launch()
            return
        elif os.path.exists(os.path.join(self.home, 'optimize.log')):
            self.fails += 1
            logger.info("%s has optimize.log but not optimize.xyz - it may have failed (%i tries)" % (self.name, self.fails))
            shutil.move(os.path.join(self.home, 'optimize.log'), os.path.join(self.home, 'optimize.%i.log' % self.fails))
            # Optimizations can fail for a number of reasons.  If the
            # Q-Chem job crashes because it's not set up correctly, we
            # get a nasty failure.  We also get a failure if the
            # geometry optimization does not converge (rare).  The
            # problem is that we don't want to give up submitting the
            # jobs in the former case, because otherwise they fall out
            # of the workflow.  So I set the number of attempts to
            # infinity, but really I should be parsing the output to
            # see what is failing.
            if self.fails >= np.inf or self.read_only:
                self.saveStatus('failed', to_disk=False, message='gave up after %i tries' % self.fails)
                self.parent.launch()
            else:
                self.launch()
            return
        if self.read_only: return
        # Otherwise we run the optimization calculation.
        # Ensure that this calculation contains a Molecule object.
        if not isinstance(self.initial, Molecule):
            M = Molecule(self.initial)
        else:
            M = deepcopy(self.initial)
        self.synchronizeChargeMult(M)
        if self.charge == -999:
            self.saveStatus('failed', message='Charge and spin inconsistent')
            return
        if len(M) != 1:
            logger.error("Optimization can only handle length-1 Molecule objects")
            raise RuntimeError
        M.write(os.path.join(self.home, 'initial.xyz'))
        # Note that the "first" method and basis set is used for the geometry optimization.
        make_task("optimize-geometry.py initial.xyz --method %s --basis \"%s\" --charge %i --mult %i &> optimize.log" % 
                  (self.methods[0], self.bases[0], self.charge, self.mult), 
                  self.home, inputs=["initial.xyz"], outputs=["optimize.log", "optimize.tar.bz2"], 
                  tag=self.name, calc=self, verbose=self.verbose)

class TransitionState(Calculation):
    """
    Class representing a transition state optimization and IRC calculation.
    """

    def __init__(self, initial, home, **kwargs):
        """
        Parameters
        ----------
        initpath : Molecule object containing the initial pathway
        """
        self.initpath = kwargs['initpath']
        super(TransitionState, self).__init__(initial, home, **kwargs)

    calctype = "TransitionState"
    
    def launch_(self):
        """
        Launch an transition state calculation.
        """
        # If this method is called and optimize.xyz exists, that means
        # the optimization is completed!
        extract_tar(os.path.join(self.home, 'transition-state.tar.bz2'), ['ts.xyz', 'irc.nrg', 'irc.pop', 'irc.xyz', 'initpath.xyz', 
                                                                          'irc_spaced.xyz', 'irc_reactant.bnd', 'irc_product.bnd', 
                                                                          'irc_transition.bnd', 'irc_transition.vib', 'deltaG.nrg'])
        extract_tar(os.path.join(self.home, 'ts-analyze.tar.bz2'), ['irc_transition.bnd', 'irc_transition.vib'])
        if os.path.exists(os.path.join(self.home, 'transition-state.log')):
            log = os.path.join(self.home, 'transition-state.log')
            errmsg = parse_irc_error(log)
            if len(errmsg) > 1:
                self.saveStatus('failed')
                for line in errmsg:
                    logger.info(line, printlvl=2)
                if (not self.ts_branch): self.parent.launch()
                return

        if os.path.exists(os.path.join(self.home, 'irc.xyz')):
            self.saveStatus('analysis', display=True, to_disk=False, ansi='\x1b[1;96m')
            status, fwd = analyze_path('irc.xyz', 'irc.nrg', cwd=self.home, xyz0='initpath.xyz', label='Transition State', draw=self.draw)
            # The status is used by growing string to decide whether to continue.
            # Since it's printed to the terminal in analyze_irc, we don't display it again.
            self.saveStatus(status, display=False, to_disk=False)
            # This bit of code adds a bit of transition state data
            # that was missing from earlier versions of the code, it
            # can be deleted later.
            if not os.path.exists(os.path.join(self.home, 'irc_transition.vib')):
                make_task("ts-analyze.py ts.xyz --method %s --basis \"%s\" --charge %i --mult %i &> ts-analyze.log" % 
                          (self.methods[-1], self.bases[-1], self.charge, self.mult),
                          self.home, inputs=["ts.xyz"], outputs=["ts-analyze.log", "ts-analyze.tar.bz2", "irc_transition.bnd", "irc_transition.vib"], 
                          tag=self.name+':AN', calc=None, verbose=self.verbose, priority=self.priority+1e6)
            # This here code assumes that the parent calculations of a TransitionState are (GS/NEB) and Pathway
            if status == 'correct':
                self.parent.saveStatus('correct', message='Correct transition state found')
                self.parent.parent.saveStatus('complete', message='Correct transition state found')
                # Print DeltaG's from the calculation
                if os.path.exists(os.path.join(self.home, 'deltaG.nrg')):
                    for line in open(os.path.join(self.home, 'deltaG.nrg'), 'r').readlines():
                        logger.info(line)
                else:
                    logger.info("deltaG.nrg file missing, can't report energy information")                
            if (not self.ts_branch): self.parent.launch()
        else:
            if os.path.exists(os.path.join(self.home, 'transition-state.log')):
                logger.info("Log file is present, no error but result is missing. Check this log file:", printlvl=2)
                for line in open(os.path.join(self.home, 'transition-state.log')).readlines():
                    logger.info(line, printlvl=2)
            if self.read_only: return
            # Otherwise we run the transition state calculation.
            # Create the Molecule object (if one wasn't passed in),
            # set charge and multiplicity.
            if not isinstance(self.initial, Molecule):
                M = Molecule(self.initial)[-1]
            else:
                M = deepcopy(self.initial)[-1]
            if not isinstance(self.initpath, Molecule):
                self.initpath = Molecule(self.initpath)
            self.synchronizeChargeMult(M)
            if self.charge == -999:
                self.saveStatus('failed', message='Charge and spin inconsistent')
                return
            if len(M) != 1:
                logger.error("TS calculation can only handle length-1 Molecule objects")
                raise RuntimeError
            M.write(os.path.join(self.home, 'initial.xyz'))
            self.initpath.write(os.path.join(self.home, 'initpath.xyz'))
            # Launch the transition state calculation!
            make_task("transition-state.py initial.xyz --initpath initpath.xyz --methods %s --bases %s --charge %i --mult %i &> transition-state.log" % 
                      (' '.join(["\"%s\"" % i for i in self.methods]), ' '.join(["\"%s\"" % i for i in self.bases]), self.charge, self.mult),
                      self.home, inputs=["initial.xyz", "initpath.xyz"], outputs=["transition-state.log", "transition-state.tar.bz2"], 
                      tag=self.name, calc=self, verbose=self.verbose)

class FreezingString(Calculation):
    """ 
    Class representing a freezing string calculation.
    """
    calctype = "FreezingString"

    def launch_(self):
        """ Launch a freezing string calculation. """
        # Extract .tar file contents.
        extract_tar(os.path.join(self.home, 'freezing-string.tar.bz2'), ['Vfile.txt', 'stringfile.txt', 'ts.xyz', 'irc.nrg', 
                                                                         'irc.pop', 'irc.xyz', 'irc_spaced.xyz', 'irc_reactant.bnd', 
                                                                         'irc_product.bnd', 'irc_transition.bnd', 'irc_transition.vib',
                                                                         'deltaG.nrg'])
        extract_tar(os.path.join(self.home, 'ts-analyze.tar.bz2'), ['irc_transition.bnd', 'irc_transition.vib'])
        # Read freezing string results if exist, and return.
        if os.path.exists(os.path.join(self.home, 'freezing-string.log')) and os.path.exists(os.path.join(self.home, 'stringfile.txt')):
            log = os.path.join(self.home, 'freezing-string.log')
            errmsg = parse_irc_error(log)
            if len(errmsg) > 1:
                self.saveStatus('failed')
                for line in errmsg:
                    logger.info(line, printlvl=2)
                return
        if os.path.exists(os.path.join(self.home, 'irc.xyz')) and os.path.exists(os.path.join(self.home, 'stringfile.txt')):
            self.saveStatus('analysis', display=True, to_disk=False, ansi='\x1b[1;96m')
            status, fwd = analyze_path('irc.xyz', 'irc.nrg', cwd=self.home, xyz0='stringfile.txt', label='TS from FS', draw=self.draw)
            # This bit of code adds a bit of transition state data
            # that was missing from earlier versions of the code, it
            # can be deleted later.
            if not os.path.exists(os.path.join(self.home, 'irc_transition.vib')):
                make_task("ts-analyze.py ts.xyz --method %s --basis \"%s\" --charge %i --mult %i &> ts-analyze.log" % 
                          (self.methods[-1], self.bases[-1], self.charge, self.mult),
                          self.home, inputs=["ts.xyz"], outputs=["ts-analyze.log", "ts-analyze.tar.bz2", "irc_transition.bnd", "irc_transition.vib"], 
                          tag=self.name+':AN', calc=None, verbose=self.verbose, priority=self.priority+1e6)
            # Save status as complete no matter the result; no jobs come after freezing string 
            # and it doesn't set the status in the parent.
            self.saveStatus('complete')
            # Print DeltaG's from the calculation
            if os.path.exists(os.path.join(self.home, 'deltaG.nrg')):
                for line in open(os.path.join(self.home, 'deltaG.nrg'), 'r').readlines():
                    logger.info(line)
            else:
                logger.info("deltaG.nrg file missing, can't report energy information") 
        else:
            if os.path.exists(os.path.join(self.home, 'freezing-string.log')):
                logger.info("Log file is present, no error but result is missing. Check this log file:", printlvl=2)
                for line in open(os.path.join(self.home, 'freezing-string.log')).readlines():
                    logger.info(line, printlvl=2)
            if self.read_only: return
            # Otherwise we run the calculation.
            if not isinstance(self.initial, Molecule):
                M = Molecule(self.initial)
            else:
                M = deepcopy(self.initial)
            self.synchronizeChargeMult(M)
            if self.charge == -999:
                self.saveStatus('failed', message='Charge and spin inconsistent')
                return
            # Obtain just the first and last frames.
            M = M[0] + M[-1]
            M.write(os.path.join(self.home, 'initial.xyz'))
            # Launch the task.
            make_task("freezing-string.py initial.xyz --methods %s --bases %s --charge %i --mult %i &> freezing-string.log" % 
                      (' '.join(["\"%s\"" % i for i in self.methods]), ' '.join(["\"%s\"" % i for i in self.bases]), self.charge, self.mult),
                      self.home, inputs=["initial.xyz"], outputs=["freezing-string.log", "freezing-string.tar.bz2"], 
                      tag=self.name, calc=self, verbose=self.verbose)

#class GrowingString(Calculation):
#    """ 
#    Class representing a growing string calculation.  This calculation
#    is actually run in several steps so it has the following
#    structure:
#
#    ---GS:00---GS:01---GS:02---GS:03 (Converged)
#                 |       |       |
#                 |       |       |
#               TS:01   TS:02   TS:03
#    
#    Basically, the GS calculation is so long that it is run in
#    "chunks" of several string iterations (say, 30).  When the
#    perpendicular gradient falls below some threshold (looser than
#    convergence), transition state calculations are launched.  If a TS
#    calculation produces an IRC that goes back to the correct reactant
#    and product, the calculation is marked as "complete".  If the GS
#    calculation has too many "chunks", or if the perpendicular
#    gradient is no longer decreasing (i.e. it's stuck), the
#    calculation is terminated.
#    """
#    calctype = "GrowingString"
#
#    def __init__(self, initial, home, **kwargs):
#        """
#        Parameters
#        ----------
#        stability_analysis : bool
#            Whether to include stability analyses as a part of the calculation.
#            This significantly increases the cost but could help when the
#            potential energy surface is discontinuous.
#        """
#        # Dictionary of transition state calculations.
#        # These are launched from concluded growing string calculations.
#        self.TransitionStates = OrderedDict()
#        # Whether to include stability analyses as a part of the calculation.
#        self.stability_analysis = kwargs.pop("stability_analysis", False)
#        # Initialize base class.
#        super(GrowingString, self).__init__(initial, home, **kwargs)
#        self.printed = []
#
#    def one(self, ncalc):
#        """ 
#        Read calculation status from a growing string log file, 
#        and launch transition state calculation if necessary.
#        Returns an error if the transition state estimate doesn't exist.
#        """
#        # The path of the individual growing string calculation.
#        rdir = os.path.join(self.home, '%02i' % ncalc)
#        # Extract the growing string archive file.
#        extract_tar(os.path.join(rdir, 'growing-string.tar.bz2'), ['Vfile.txt', 'tsestimate.xyz', 'final-string.xyz', 'final-string.pop'])
#        # Large initial value, used in case there are no pgrads.
#        pgrads = []
#        # Number of iterations.
#        niter = 0
#        #-----
#        # One of the following options:
#        # maxiter  = This calculation reached the maximum number of iterations
#        # cnvgd    = Converged (the best result, but rare)
#        # failed   = Calculation quit with error message, cannot continue
#        # unknown  = Something that we didn't account for
#        gsstat = 'unknown'
#        # Error message if any.
#        errmsg = ''
#        # pgrad convergence tolerance
#        cvg_grad = 0.002
#        # A relaxed convergence tolerance 
#        # where we may launch a TS search
#        ts_grad = 0.02
#        # Expected path of growing string log file.
#        log = os.path.join(rdir, 'growing-string.log')
#        if os.path.exists(log):
#            if not os.path.exists(os.path.join(rdir, 'tsestimate.xyz')):
#                logger.info("%s does not have tsestimate.xyz" % rdir)
#                shutil.move(log, log+'.bak')
#                return [], "error", "no TS estimate", False
#            if not os.path.exists(os.path.join(rdir, 'final-string.xyz')):
#                self.parent.saveStatus('failed', message='Final string does not exist')
#                return [], "error", "no string", False
#        else:
#            return [], "new", "no log file", False
#        # Parse the log file.
#        for line in open(log):
#            if "MAX_PERP_GRAD_FOR_A_NODE" in line:
#                pgrads.append(float(line.split()[2]))
#                niter += 1
#                if pgrads[-1] < cvg_grad:
#                    gsstat = 'cnvgd'
#            if "Reached Maximum iterations, exiting" in line:
#                gsstat = 'maxiter'
#                break
#            if "You have finished the Growing String run" in line:
#                gsstat = 'cnvgd'
#                break
#            if "MAX reached in spline interpolation" in line:
#                gsstat = 'error'
#                errmsg = 'Spline interpolation error'
#        if ncalc not in self.printed:
#            logger.info("GrowingString   segment \x1b[94m%i\x1b[0m : %i cycles, pgrad = %.3f, status: %s" % (ncalc, len(pgrads), pgrads[-1], gsstat), printlvl=2)
#            self.printed.append(ncalc)
#        # If the growing string calculation meets these criteria, then launch the transition state search.
#        ts_launch = False
#        if gsstat == 'cnvgd' or (gsstat == 'maxiter' and pgrads[-1] < ts_grad):
#            if ncalc not in self.TransitionStates:
#                self.TransitionStates[ncalc] = TransitionState(os.path.join(rdir, 'tsestimate.xyz'), home=os.path.join(rdir, 'TS'), 
#                                                               initpath=os.path.join(rdir, 'final-string.xyz'), parent=self, 
#                                                               charge=self.charge, mult=self.mult, priority=self.priority+self.dprio+100, **self.kwargs)
#                self.TransitionStates[ncalc].launch()
#                ts_launch = True
#        return pgrads, gsstat, errmsg, ts_launch
#
#    def launch_(self):
#        """
#        Process growing string results and launch the calculation.
#        """
#        # At least one transition state converged to an IRC consistent with the reactant and product.
#        # If so, there is no reason to continue the growing string calculation.
#        if any([calc.status == 'correct' for calc in list(self.TransitionStates.values())]):
#            self.saveStatus('correct', message='Correct transition state found')
#            self.parent.saveStatus('complete', message='Correct transition state found')
#            return
#
#        ncalc = 0
#        pgrads = []
#        gsstat = 'new'
#        self.dprio = 0
#        while True:
#            # The number of growing string cycles increases for each segment in the following schedule:
#            # Set the calculation status based on the status of the parent pathway.
#            # This may be updated within any transition state calculation, that's why it's in the loop.
#            if self.parent.status == 'complete':
#                self.saveStatus('complete', message='Pathway complete')
#                return
#            if self.parent.status == 'failed':
#                self.saveStatus('failed', message='Pathway failed')
#                return
#            # The initial growing string iterations have the highest priority.  After 200 iterations it
#            # becomes quite a bit lower.
#            self.dprio = 100-(len(pgrads))
#            # Because this is calculation is organized into a series of segments, we loop
#            # over them and set the status based on the final segment.
#            if len(pgrads) > self.gsmax:
#                # There were too many growing string segments, and it's unlikely to ever finish
#                self.saveStatus('terminated', message='Too many growing string cycles (%i > %i)' % (len(pgrads), self.gsmax))
#                return
#            if not os.path.exists(os.path.join(self.home, '%02i' % ncalc)): break
#            pgrad_cyc, gsstat, errmsg, ts_launch = self.one(ncalc)
#            pgrads += pgrad_cyc
#            # If following the sequential path, the TS calculation will make us return.
#            # However, when the TS calculation is finished it should trigger this function again.
#            if ts_launch and (not self.ts_branch): return
#            # Break out of the loop if any calculation is empty.
#            if len(pgrad_cyc) == 0: break
#            ncalc += 1
#        
#        # Set the calculation status based on the status of the latest segment.
#        if gsstat in ['error', 'unknown']:
#            # Calculation has failed, or the parser encountered something it did not expect.
#            logger.info('Calculation %s status %s' % (self.name, gsstat))
#            self.saveStatus('failed')
#            return
#        elif gsstat == 'cnvgd':
#            # Growing string has converged.
#            self.saveStatus('converged')
#            return
#
#        if self.read_only: return
#        self.saveStatus('launch', message='%s from cycle %i' % ('Starting' if len(pgrads) == 0 else 'Continuing', len(pgrads)))
#        # At this point in the code, we will create a new growing string calculation.
#        if ncalc > 0:
#            # If previous calculations exist, then get the final string from the previous segment.
#            M = Molecule(os.path.join(self.home, '%02i' % (ncalc-1), 'final-string.xyz'))
#        else:
#            # If this is a new calculation, then create the initial string.
#            if isinstance(self.initial, Molecule):
#                M = deepcopy(self.initial)
#            else:
#                M = Molecule(self.initial)
#        
#        # Write the initial string to the folder where the calculation will actually be run.
#        nextd = os.path.join(self.home, '%02i' % ncalc)
#        if not os.path.exists(nextd): os.makedirs(nextd)
#        M.write(os.path.join(nextd, 'initial.xyz'))
#        
#        # The number of growing string cycles increases as a function of the segment number.
#        ncycles = {0:20, 1:30, 2:50, 3:100}
#
#        # Launch the task.
#        make_task("growing-string.py initial.xyz --method %s --basis \"%s\" --charge %i --mult %i --cycles %i --images %i %s &> growing-string.log" % 
#                  (self.methods[0], self.bases[0], self.charge, self.mult, ncycles.get(ncalc, 100), self.images, '--stab' if self.stability_analysis else ''), 
#                  nextd, inputs=["initial.xyz"], outputs=["growing-string.log", "growing-string.tar.bz2"], 
#                  tag=self.name, calc=self, verbose=self.verbose, priority=self.priority + self.dprio)
#

class Interpolation(Calculation):
    """
    Class representing internal coordinate interpolation.
    """
    calctype = "Interpolation"
    def launch_(self):
        """
        Launch internal coordinate interpolation.
        """
        if os.path.exists(os.path.join(self.home, 'interpolated.xyz')):
            self.saveStatus('complete')
            self.parent.launch()
            return
        elif os.path.exists(os.path.join(self.home, 'interpolate.log')):
            # On the other hand, if the interpolation fails, then the whole
            # pathway has failed.
            self.saveStatus('failed')
            return
        if self.read_only: return
        if not isinstance(self.initial, Molecule):
            M = Molecule(self.initial)
        else:
            M = deepcopy(self.initial)
        M.write(os.path.join(self.home, '.interpolate.in.xyz'))
        # Note that the "first" method and basis set is used for the geometry optimization.
        make_task("Nebterpolate.py --morse 1e-2 --repulsive --allpairs --anchor 2 .interpolate.in.xyz interpolated.xyz &> interpolate.log",
                  self.home, inputs=[".interpolate.in.xyz"], outputs=["interpolate.log", "interpolated.xyz"], 
                  tag=self.name+"_interpolate", calc=self, verbose=self.verbose)

class Pathway(Calculation):
    """
    Class representing an individual reaction pathway.  Note that many
    of these may be created from a single reaction.xyz.
    """
    calctype = "Pathway"

    def countFragmentIDs(self):
        return sum([calc.status == 'converged' for calc in list(self.FragmentIDs.values())]), len(list(self.FragmentIDs.values()))

    def countFragmentOpts(self):
        return sum([calc.status == 'converged' for calc in list(self.FragmentOpts.values())]), len(list(self.FragmentOpts.values()))

    def countOptimizations(self):
        return sum([calc.status == 'converged' for calc in list(self.Optimizations.values())]), len(list(self.Optimizations.values()))

    def launch_(self):
        """
        Launch pathway-based calculations.
        """


        # Equally spaced .xyz file with re-optimized endpoints.
        self.M1 = os.path.join(self.home, 'respaced.xyz')
        # If this file does not exist, then we need to create it.
        if not os.path.exists(self.M1):
            # Read in the Molecule object, and set charge / multiplicity.
            if isinstance(self.initial, Molecule):
                self.M0 = deepcopy(self.initial)
            else:
                self.M0 = Molecule(self.initial)
            self.synchronizeChargeMult(self.M0)
            if self.charge == -999:
                self.saveStatus('failed', message='Charge and spin inconsistent')
                return
            # Continue optimizations of endpoints.
            if not hasattr(self, 'Optimizations'):
                self.Optimizations = OrderedDict()
                self.Optimizations[0] = Optimization(initial=self.M0[0], home=os.path.join(self.home, "opt-init"), parent=self, priority=self.priority+1000, **self.kwargs)
                self.Optimizations[1] = Optimization(initial=self.M0[-1], home=os.path.join(self.home, "opt-final"), parent=self, priority=self.priority+1000, **self.kwargs)
                for calc in list(self.Optimizations.values()): 
                    calc.launch()
                return
            if (len(self.Optimizations) == 2) and all([calc.status == 'converged' for calc in list(self.Optimizations.values())]):
                OptMols = OrderedDict()
                for frm, calc in list(self.Optimizations.items()):
                    OptMols[frm] = Molecule(os.path.join(calc.home, 'optimize.xyz'), topframe=-1)
                    OptMols[frm].load_popxyz(os.path.join(calc.home, 'optimize.pop'))
                # Catch the *specific case* that after reoptimizing the 
                # endpoints, the molecules became the same again.
                if self.Equal(OptMols[0], OptMols[1]):
                    logger.info("After reoptimizing endpoints, %s no longer contains a reaction" % self.name, printlvl=2)
                    self.saveStatus('no-reaction')
                    return
                # If the initial and final molecules are different, we
                # include them all in the pathway without rechecking topology.  
                Joined = (OptMols[0][::-1].without('qm_mulliken_charges', 'qm_mulliken_spins') + self.M0 + 
                          OptMols[1].without('qm_mulliken_charges', 'qm_mulliken_spins'))
                Spaced = EqualSpacing(Joined, dx=0.05)
                Joined.write(os.path.join(self.home, 'rejoined.xyz'))
                Spaced.write(os.path.join(self.home, 'respaced.xyz'))
                self.M1 = Spaced
            elif (len(self.Optimizations) == 2) and any([calc.status == 'failed' for calc in list(self.Optimizations.values())]):
                self.saveStatus('failed', message='At least one endpoint optimization has failed')
                return
            else:
                return
                
        # Create freezing string calculation.
        if not hasattr(self, 'FS'):
            self.FS = FreezingString(self.M1, home=os.path.join(self.home, 'FS'), 
                                     charge=self.charge, mult=self.mult, priority=self.priority+100, **self.kwargs)
            self.FS.launch()

        # Create internal coordinate interpolation.
        if not hasattr(self, 'Interpolation'):
            self.Interpolation = Interpolation(self.M1, home=self.home, parent=self, priority=self.priority+1000, **self.kwargs)
            self.Interpolation.launch()
            
        if self.Interpolation.status != 'complete':
            return

        # Obtain equally-spaced, interpolated frames for growing string.
        if not os.path.exists(os.path.join(self.home, 'interspaced.xyz')):
            Interpolated = Molecule(os.path.join(self.home, 'interpolated.xyz'))
            InterSpaced = EqualSpacing(Interpolated, frames=self.images)
            InterSpaced.write(os.path.join(self.home, 'interspaced.xyz'))
        else:
            InterSpaced = os.path.join(self.home, 'interspaced.xyz')

        # Create growing string calculation.  These calculations are
        # run in segments and can be extended based on the status of
        # the last growing string calculation.
        #if not hasattr(self, 'GS'):
        #    # With MPI, stability analysis is quite affordable so we'll enable it by default
        #    self.GS = GrowingString(InterSpaced, home=os.path.join(self.home, 'GS'), 
        #                            parent=self, charge=self.charge, mult=self.mult, stability_analysis=True, priority=self.priority, **self.kwargs)
        #    self.GS.launch()
        #    # self.GSSA = GrowingString(InterSpaced, home=os.path.join(self.home, 'GSSA'), 
        #    #                         parent=self, charge=self.charge, mult=self.mult, stability_analysis=True, priority=self.priority, **self.kwargs)
        #    # self.GSSA.launch()
            
class Trajectory(Calculation):
    """
    Class representing a reactive dynamics trajectory from the
    the nanoreactor.  It is initialized from a .xyz file and
    contains all of the energy refinement calculations that
    result in individual reaction pathways.
    """
    calctype = "Trajectory"
    
    def __init__(self, initial, home, **kwargs):
        """
        Create a Trajectory object.
        
        Parameters
        ----------
        initial : str
            Path of an xyz file, typically a dynamics trajectory from the nanoreactor (reaction_123.xyz).  
            If charge and multiplicity are not provided, they will be read from the comment line.
            If charge and multiplicity cannot be determined by passing charge/mult -or- reading the xyz, it will crash.
        charge : int
            Net charge of the atoms in the xyz file.  If provided this takes precedence 
            over any charges provided in the xyz comment lines.
        mult : int
            Spin multiplicity of the atoms in the xyz file (2*<Sz>+1).  
            If provided this takes precedence over any charges provided in the xyz comment lines.
        subsample : int
            Frequency of subsampling the trajectory in the geometry optimization and pathway calculations.
        pathmax : int
            Maximum length (in MD frames) for considering a dynamics trajectory.
        """
        # Initialize base class but don't launch.
        super(Trajectory, self).__init__(initial, home, **kwargs)
        # Copy initial xyz file into the home folder.
        if not os.path.exists(os.path.join(self.home, 'initial.xyz')):
            shutil.copy2(initial, os.path.join(self.home, 'initial.xyz'))
        # Subsampling frequency for geometry optimization + pathway analysis.
        self.subsample = kwargs.pop('subsample', 10)
        # Don't perform pathway calculations for frames spaced apart by more than this.
        self.pathmax = kwargs.pop('pathmax', 1000)
        # Dynamics trajectories that are longer than this are ignored for the time being.
        self.dynmax = kwargs.pop('dynmax', 2000)
        # Trajectories that contain too many atoms can be excluded using this filter.
        self.atomax = kwargs.pop('atomax', 50)
        # Do optimizations of fragments and calculate fragment-based Delta G's
        self.doFrags = kwargs.pop('frags', False)
        # Create the fragments, structure, and pathway folders.
        self.fragmentFolder = os.path.join(self.home, 'fragments')
        self.structureFolder = os.path.join(self.home, 'structures')
        self.pathwayFolder = os.path.join(self.home, 'pathways')
        # Denotes whether the class contains the full complement of optimizations.
        if not os.path.exists(self.fragmentFolder): os.makedirs(self.fragmentFolder)
        if not os.path.exists(self.structureFolder): os.makedirs(self.structureFolder)
        if not os.path.exists(self.pathwayFolder): os.makedirs(self.pathwayFolder)

    def makeFragments(self):
        """ Create and launch fragment identifications. This code is called ONCE per trajectory"""
        self.FragmentIDs = OrderedDict()
        self.frames = list(range(0, len(self.M), self.subsample))
        if (len(self.M)-1) not in self.frames:
            self.frames.append(len(self.M)-1)
        for frm in self.frames:
            ohome = os.path.join(self.fragmentFolder, "%%0%ii" % self.fdigits % frm)
            oname = ohome.replace(os.getcwd(), '').strip('/')
            logger.info("Fragment geometry identification in %s" % (ohome), printlvl=3)
            # Note: This code may return to the parent and rerun Trajectory.launch() before the dictionary is filled.
            self.FragmentIDs[frm] = FragmentID(initial=self.M[frm], name=oname, home=ohome, 
                                                   parent=self, priority=self.priority, **self.kwargs)
            self.FragmentIDs[frm].launch()
        # Potentially have code to process the output and print the delta G's?

    def makeFragOpts(self):
        """ Create and launch fragment optimizations. This code is called ONCE per trajectory"""
        FragIDs = OrderedDict()
        # Determine which frames to actually optimize
        for frm in self.frames:
            ohome = os.path.join(self.fragmentFolder, "%%0%ii" % self.fdigits % frm)    
            fragidtxt = open(os.path.join(ohome,"fragmentid.txt"))
            formulas = sorted(fragidtxt.readline().split())
            bondfactor = fragidtxt.readline().split()[1]
            validity = fragidtxt.readline().strip()
            fragidtxt.close()
            if validity != "invalid": 
                FragIDs[frm]=(formulas, bondfactor)
        # Sort by bondfactor in descending order
        FragIDs = OrderedDict(sorted(list(FragIDs.items()), key=lambda item: item[1][1], reverse = True))
        # Pick out frame with maximum bondfactor for fragment group
        self.optlist = []
        fraglist = []
        logger.info("Identifying frames for unique fragment sets with maximum bonding:")
        for frm,frag in list(FragIDs.items()):
            if frag[0] not in fraglist:
                logger.info("Frame: %d; Fragments: %s" % (frm, frag[0]))
                self.optlist.append(frm)
                fraglist.append(frag[0])
        self.optlist = sorted(self.optlist)
        self.FragmentOpts = OrderedDict()
        # Actually run fragment optimizations on each fragment
        for frm in self.optlist:
            ohome = os.path.join(self.fragmentFolder, "%%0%ii" % self.fdigits % frm, "opt")
            oname = ohome.replace(os.getcwd(), '').strip('/')
            logger.info("Fragment geometry optimization in %s" % (ohome), printlvl=3)
            # Note: This code may return to the parent and rerun Trajectory.launch() before the dictionary is filled.
            self.FragmentOpts[frm] = FragmentOpt(initial=self.M[frm], name=oname, home=ohome, 
                                                   parent=self, priority=self.priority, **self.kwargs)
            self.FragmentOpts[frm].launch()
    
    def calcDeltaGs(self):
        """ Calculate Delta-G's from sums of fragment energies. This code is called once per trajectory"""
        formulas = {}
        nrg = {}
        zpe = {}
        entr = {}
        enth = {}
        validity = {}
        Ha_to_kcalmol = 627.5096080305927
        for frm in self.optlist:
            ohome = os.path.join(self.fragmentFolder, "%%0%ii" % self.fdigits % frm, "opt")
            nrgfile = open(os.path.join(ohome, 'fragmentopt.nrg'))
            formulas[frm] = nrgfile.readline().strip()
            nrg[frm] = float(nrgfile.readline().split()[3])
            zpe[frm] = float(nrgfile.readline().split()[2])
            entr[frm] = float(nrgfile.readline().split()[3])
            enth[frm] = float(nrgfile.readline().split()[3])
            validity[frm] = nrgfile.readline().strip()
            nrgfile.close()
        self.DeltaG = 0
        DeltaE = 0
        for frm in self.optlist:
            if validity[frm] == "invalid":
                logger.info("Optimization from frame %s was invalid - molecular formulas not preserved" % frm)
                logger.info("No Delta-G will be calculated from frame %s" % frm)
        for fi in self.optlist:
            for fj in self.optlist:
                if fj > fi and validity[fi] != "invalid" and validity[fj] != "invalid":
                    Efi = nrg[fi]*Ha_to_kcalmol + zpe[fi]
                    Gfi = Efi - entr[fi]*0.29815 + enth[fi]
                    Efj = nrg[fj]*Ha_to_kcalmol + zpe[fj]
                    Gfj = Efj - entr[fj]*0.29815 + enth[fj]
                    DeltaE = Efj - Efi
                    self.DeltaG = Gfj - Gfi
                    strrxn = ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(formulas[fi].split()).items())])
                    strrxn += ' -> '
                    strrxn += ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in list(Counter(formulas[fj].split()).items())])
                    logger.info("=> Frame %s -> %s: Reaction %s; Delta-H (0K) = %.4f kcal/mol; Delta-G (STP) = %.4f kcal/mol" 
                                % (fi, fj, strrxn, DeltaE, self.DeltaG))

    def makeOptimizations(self):
        """ Create and launch geometry optimizations.  This code is called ONCE per trajectory. """
        self.Optimizations = OrderedDict()
        # Create molecule object and set charge / multiplicity.
        # Create a list of frames separated by the stride and 
        # including the last frame.
        self.frames = list(range(0, len(self.M), self.subsample))
        if (len(self.M)-1) not in self.frames:
            self.frames.append(len(self.M)-1)
        for frm in self.frames:
            # The funny string creates an integer with the correct number of leading zeros.
            ohome = os.path.join(self.structureFolder, "%%0%ii" % self.fdigits % frm)
            oname = ohome.replace(os.getcwd(), '').strip('/')
            # oname = self.name + ("/optimize_%%0%ii" % self.fdigits % frm)
            logger.info("Geometry optimization in %s" % (ohome), printlvl=3)
            # Note: This code may return to the parent and rerun Trajectory.launch() before the dictionary is filled.
            self.Optimizations[frm] = Optimization(initial=self.M[frm], name=oname, home=ohome, 
                                                   parent=self, priority=self.priority, **self.kwargs)
            self.Optimizations[frm].launch()

    def makePathways(self):
        """
        Identify the individual reaction pathways.  This method is
        executed after all geometry optimizations are complete.  
        Like makeOptimizations, this code is called ONCE per trajectory.

        After running geometry optimizations on frames subsampled from
        the dynamics, we have a number of geometries that look like this:

             (Minimized)         (Minimized)
                  |                   |
                  |                   |
         /\      / \    /\     /\    / \     
        /  \ /\_/   \/\/  \ /\/  \/\/   \/\/\ (Dynamics)
            |              |
            |              |
            |              |     
        (Minimized)    (Minimized)
        
        In general, we will likely see several minimizations that go to the same
        molecule (catchment basin on the PES), followed by some that go to another
        molecule, such as: A, A, A, B, B, A, A, C, C
        
        The goal is to get all continuous connecting segments between
        DISTINCT optimized structures.  Thus, this method does the following:

        1) Inspects pairs of sequential optimizations to see whether the
        molecular topology is different
        2) Each frame pair is a potential initial / final frame of a pathway
        3) Looping over all potential initial and final frames:
        3a) Make sure initial and final frames have different molecules (no A, B, A)
        3b) Create a connecting segment by reversing initial optimization, 
            then concatenating the dynamics and the final optimization
        3c) Write the segment to "pathways/123-456/joined.xyz": 123, 456 are frame #s
        3d) Write a new segment with equally spaced coordinates to "pathways/123-456/spaced.xyz"
        3e) Create the pathway calculation which begins with interpolation / FS and ends with transition states.

        """
        self.Pathways = OrderedDict()
        # If the pathway information is cached to disk, then we load the text file; it saves us a lot of work.
        if self.fast_restart and os.path.exists(os.path.join(self.pathwayFolder, 'path-info.txt')):
            logger.info("Reading pathways from %s" % self.pathwayFolder, printlvl=2)
            def ascii_encode_dict(data):
                """ Convert Unicode strings in dictionary (e.g. JSON-loaded) to ascii. """
                def ascii_encode(x):
                    if isinstance(x, str):
                        return x.encode('ascii')
                    else:
                        return x
                return OrderedDict(list(map(ascii_encode, pair)) for pair in list(data.items()))
            ok = True
            PathInfo = json.load(open(os.path.join(self.pathwayFolder, 'path-info.txt')), object_hook=ascii_encode_dict)
            for label, pathparams in list(PathInfo.items()):
                if not os.path.exists(os.path.join(pathparams['home'], 'spaced.xyz')):
                    ok = False
                    break
                self.Pathways[label] = Pathway(os.path.join(pathparams['home'], 'spaced.xyz'), home=pathparams['home'], name=pathparams['name'], 
                                               charge=pathparams['charge'], mult=pathparams['mult'], parent=self, priority=self.priority, **self.kwargs)
            if ok:
                for P in list(self.Pathways.values()):
                    P.launch()
                return
        # If the cached path information doesn't exist or something went wrong, then we determine all of the pathways.
        PathInfo = OrderedDict()
        logger.info("Identifying pathways for trajectory %s" % self.name, printlvl=1)
        # Create a Molecule object for each optimized .xyz file.
        # Note that topology is determined by the final frame
        OptMols = OrderedDict()
        for frm, calc in list(self.Optimizations.items()):
            if calc.status == 'failed': continue
            OptMols[frm] = Molecule(os.path.join(calc.home, 'optimize.xyz'), topframe=-1)
            OptMols[frm].load_popxyz(os.path.join(calc.home, 'optimize.pop'))
            self.synchronizeChargeMult(OptMols[frm])
            if self.charge == -999:
                self.saveStatus('failed', message='Charge and spin inconsistent')
                return
        # Identify all potential initial and final frames of pathways
        # Each initial pathway frame is the final frame of a given catchment basin, up to (N-1)
        # Each final pathway frame is the initial frame of a given catchment basin, starting from the 2nd one
        path_initial = []
        path_final = []
        for fi, fj in zip(list(OptMols.keys())[:-1], list(OptMols.keys())[1:]):
            if not self.Equal(OptMols[fi], OptMols[fj]):
                # fi and fj mark the boundaries of a difference in catchment basin
                path_initial.append(fi)
                path_final.append(fj)
        # We now have a potential explosion in the number of pathways.  For all pathways
        # that begin and end in the same pair of molecules, we pick out the one that is
        # separated by the smallest number of frames.
        MolPairs = []
        FramePairs = []
        for fi in path_initial:
            for fj in path_final:
                if fj > fi and (not self.Equal(OptMols[fi], OptMols[fj])):
                    if (fj - fi) > self.pathmax: continue
                    NewPair = True
                    for imp, (m1, m2) in enumerate(MolPairs):
                        if self.Equal(OptMols[fi], m1) and self.Equal(OptMols[fj], m2):
                            FramePairs[imp].append((fi, fj))
                            NewPair = False
                            break
                        elif self.Equal(OptMols[fi], m2) and self.Equal(OptMols[fj], m1):
                            FramePairs[imp].append((fi, fj))
                            NewPair = False
                            break
                    if NewPair:
                        MolPairs.append((OptMols[fi], OptMols[fj]))
                        FramePairs.append([(fi, fj)])
        # Development note: To disable the above behavior, uncomment the below code.
        # for fi in path_initial:
        #     for fj in path_final:
        #         if fj > fi and (not self.Equal(OptMols[fi], OptMols[fj])):
        #             if (fj - fi) > self.pathmax: continue
        #             FramePairs.append([(fi, fj)])

        for fp in FramePairs:
            (fi, fj) = fp[np.argmin([(j-i) for (i, j) in fp])]
            if len(fp) > 1:
                logger.info("\x1b[1;94mAdding a pathway\x1b[0m connecting %i-%i (chosen from %s)" % (fi, fj, ', '.join(['%i-%i' % (i, j) for (i, j) in fp])), printlvl=1)
            else:
                logger.info("\x1b[1;94mAdding a pathway\x1b[0m connecting %i-%i" % (fi, fj), printlvl=1)
            # The initial dynamics pathway doesn't come with Mulliken charges and spins, unfortunately due to
            # LearnReactions.py not saving them.  Moving forward, I need to save the reaction_123.pop files, but
            # for now I need to pop off the qm_mulliken_* 
            Raw_Joined = (OptMols[fi][::-1].without('qm_mulliken_charges', 'qm_mulliken_spins') + self.M[fi:fj] + 
                          OptMols[fj].without('qm_mulliken_charges', 'qm_mulliken_spins'))
            # Identify the atoms that reacted (i.e. remove spectators)
            # There is sometimes more than one reacting group if multiple concurrent reactions occur
            if self.spectators:
                reacting_groups = [(list(range(OptMols[fi][-1].na)), self.charge, self.mult)]
            else:
                reacting_groups = find_reacting_groups(OptMols[fi][-1], OptMols[fj][-1])
            for rgrp, (ratoms, rcharge, rmult) in enumerate(reacting_groups):
                Joined = Raw_Joined.atom_select(ratoms)
                # Write the "joined" pathway and the "equally spaced pathway"
                Spaced = EqualSpacing(Joined, dx=0.05)
                # The output directory is named by the initial frame, final frame, and optional letter for the 
                # atom group (if there are multiple concurrent reactions)
                label = (("%%0%ii" % self.fdigits % fi) + ("-" + "%%0%ii" % self.fdigits % fj) + 
                         (('-%s' % 'abcdefghjiklmnopqrstuvwxyz'[rgrp]) if len(reacting_groups) > 1 else ""))
                pathhome = os.path.join(self.pathwayFolder, label)
                pathname = pathhome.replace(os.getcwd(),'').strip('/')
                if not os.path.exists(pathhome):
                    os.makedirs(pathhome)
                Joined.write(os.path.join(pathhome, 'joined.xyz'))
                Spaced.write(os.path.join(pathhome, 'spaced.xyz'))
                self.Pathways[label] = Pathway(Spaced, home=pathhome, name=pathname, charge=rcharge, 
                                                        parent=self, mult=rmult, priority=self.priority, **self.kwargs)
                self.Pathways[label].launch()
                PathInfo[label] = OrderedDict([('home', pathhome), ('name', pathname), ('charge', rcharge), ('mult', rmult)])
        with open(os.path.join(self.pathwayFolder, 'path-info.txt'), 'w') as f: json.dump(PathInfo, f, ensure_ascii=True)
        if len(list(self.Pathways.keys())) == 0:
            logger.info("%s has no pathways after optimizations" % self.name, printlvl=2)
            self.saveStatus('complete', message='no pathways')
        # LPW Attempt to save some memory
        del OptMols

    def countFragmentIDs(self):
        return sum([calc.status == 'converged' for calc in list(self.FragmentIDs.values())]), len(self.frames)

    def countFragmentOpts(self):
        return sum([calc.status == 'converged' for calc in list(self.FragmentOpts.values())]), len(self.optlist)
    
    def countOptimizations(self):
        return sum([calc.status == 'converged' for calc in list(self.Optimizations.values())]), len(self.frames)

    def launchOptimizations(self):
        # Create geometry optimizations if we haven't done so already.
        if not hasattr(self, 'Optimizations'):
            self.makeOptimizations()
        # If optimizations are complete, create pathway calculations;
        # otherwise print optimization status.  Once pathways are
        # created, this method will no longer be called so we don't
        # need to cycle through again.
        else:
            for calc in list(self.Optimizations.values()):
                if os.path.exists(os.path.join(calc.home, 'optimize.xyz')):
                    complete, total = self.countOptimizations()
                    calc.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
    
        if (len(self.Optimizations) == len(self.frames)) and all([calc.status in ['converged', 'failed'] for calc in list(self.Optimizations.values())]):
            if not hasattr(self, 'Pathways'):
                self.makePathways()

    def launch_(self):
        """ Main method for doing the calculation. """
        # Create initial Molecule object
        if not hasattr(self, 'M'):
            if isinstance(self.initial, Molecule):
                self.M = deepcopy(self.initial)
            else:
                self.M = Molecule(self.initial)
            logger.info("\x1b[1;95mDynamics Trajectory\x1b[0m : %s has %i atoms and %i frames" % (self.name, self.M.na, len(self.M)), printlvl=1)
            # Make sure stored charge and multiplicity are synchronized with Molecule object.
            self.synchronizeChargeMult(self.M)
            if self.charge == -999:
                self.saveStatus('failed', message='Charge and spin inconsistent')
                return
            # Number of digits in the number of frames.
            self.fdigits = len(str(len(self.M)))
        if len(self.M) > self.dynmax:
            self.saveStatus('skip', message='Dynamics too long (%i > %i ; use --dynmax to increase)' % (len(self.M), self.dynmax), to_disk=False)
            return
        if self.M.na > self.atomax:
            self.saveStatus('skip', message='Too many atoms (%i > %i ; use --atomax to increase)' % (self.M.na, self.atomax), to_disk=False)
            return

        if self.doFrags:
            #===================================#
            #| Leah's added code for fragments |#
            #===================================#
            # Create fragment identifications if we haven't done so already.
            if not hasattr(self, 'FragmentIDs'):
                self.makeFragments()
            else:
                for calc in list(self.FragmentIDs.values()):
                    if os.path.exists(os.path.join(calc.home, 'fragmentid.txt')):
                        complete, total = self.countFragmentIDs()
                        calc.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
            # Optimize fragments if we haven't done so already
            if (len(self.FragmentIDs) == len(self.frames)) and all([calc.status in ['converged', 'failed'] for calc in list(self.FragmentIDs.values())]):
                if not hasattr(self, 'FragmentOpts'):
                    self.makeFragOpts()
                else:
                    for calc in list(self.FragmentOpts.values()):
                        if os.path.exists(os.path.join(calc.home, 'fragmentopt.txt')):
                            complete, total = self.countFragmentOpts()
                            calc.saveStatus('converged', display=(self.verbose>=2), to_disk=False, message='%i/%i complete' % (complete+1, total))
    
            if (len(self.FragmentIDs) == len(self.frames)) and all([calc.status in ['converged', 'failed'] for calc in list(self.FragmentIDs.values())]):
                if (len(self.FragmentOpts) == len(self.optlist)) and all([calc.status in ['converged', 'failed'] for calc in list(self.FragmentOpts.values())]):        
                    # Calculate Delta-G's of the reaction from fragments if we haven't done so already
                    if not hasattr(self, 'DeltaG'):
                        self.calcDeltaGs()
                    self.launchOptimizations()
        else:
            self.launchOptimizations()
