#!/usr/bin/env python

import os, sys, shutil, re
import glob
import nanoreactor
import subprocess
from nanoreactor import Nanoreactor
from nanoreactor.molecule import Molecule
from nanoreactor.nanoreactor import NanoEqual, make_monotonic
from nanoreactor.nifty import *
from nanoreactor.nifty import _exec
from collections import namedtuple, defaultdict, OrderedDict
import work_queue
import argparse
import itertools

#=================================#
#| Nanoreactor refinement script |#
#|        Lee-Ping Wang          |#
#=================================#
def print_stat(calctype, action, dnm, msg="", ansi="\x1b[93m"):
    if action == 'Launch':
        ansi = "\x1b[1;44;91m"
    elif action in ['converged', 'finished']:
        ansi = "\x1b[92m"
    elif action == 'running':
        ansi = "\x1b[94m"
    elif action == 'continue':
        ansi = "\x1b[93m"
    elif action == 'failed':
        ansi = "\x1b[91m"
    elif action == 'ready':
        ansi = "\x1b[96m"
    spaces = max(0, (14 - len(action)))
    print "%-15s %s%s\x1b[0m%s in %-60s" % (calctype, ansi, action, ' '*spaces, dnm),
    if msg != "":
        print msg
    else:
        print

# Touch a file.
def touch(fname, times=None):
    with file(fname, 'a'):
        os.utime(fname, times)

# Move a file but overwrite if exists.
def movef(src, dest):
    if os.path.isfile(dest): 
        os.remove(dest)
    elif os.path.isdir(dest) and os.path.isfile(os.path.join(dest, src)):
        os.remove(os.path.join(dest, src))
    shutil.move(src, dest)

# Force-remove a file.
def rmf(dest):
    if os.path.isdir(dest):
        return
    elif os.path.isfile(dest) or os.path.islink(dest):
        os.remove(dest)

pid_table = [int(i.strip()) for i in os.popen("ps ef | awk '/^ *[0-9]/ {print $1}'").readlines()]

def pid_exists(pid):
    # try:
    #     subprocess.check_call("kill -0 %i 2> /dev/null" % pid, shell=True)
    #     return 1
    # except subprocess.CalledProcessError:
    #     return 0
    # try:
    #     os.system("kill -0 %i" % pid)
    #     return 1
    # except:
    #     return 0
    # ananke from #linux
    # ps -o pid= -p $PID_HERE >/dev/null ; echo $?
    return pid in pid_table

# A listing of all directories containing simulations.

scriptd = os.path.join(os.path.split(nanoreactor.__file__)[0],"scripts")

parser = argparse.ArgumentParser()
parser.add_argument('--patch', action="store_true", help='Patch over calculations (simple hack)')
parser.add_argument('--pic', type=int, default=0, help='Generate TS pictures (1 for TS only, 2 for string too)')

parser.add_argument('-rg', action="store_true", help='Read growing string calculations (stage 1)')
parser.add_argument('-rt', action="store_true", help='Read transition state calculations (stage 2)')
parser.add_argument('-rf', action="store_true", help='Read freezing string calculations (stage 1)')

parser.add_argument('-qa', action="store_true", help='Do path analysis and splitting (stage 0)')
parser.add_argument('-qg', action="store_true", help='Do growing string calculations (stage 1)')
parser.add_argument('-qt', action="store_true", help='Do transition state calculations (stage 2)')
parser.add_argument('-qf', action="store_true", help='Do freezing string calculations (stage 1)')
parser.add_argument('-dt', action="store_true", help='Save space by deleting large (>1MB) files from failed transition state results (stage 2)')
parser.add_argument('-ft', action="store_true", help='Force transition state calculations for all folders with no data (stage 2)')

parser.add_argument('-p', type=int, default=5815, help='Port for Work Queue')

args, sys.argv= parser.parse_known_args(sys.argv)

# if args.p == 5812:
#     dirs = ["UM2/smash7.3", "UM2/smash7", 
#             "UM2/smash7.3b", "UM2/smash7.3bo", "UM2/smash7.3-o", "UM2/smash7-m", "UM2/smash7-m-2t", 
#             "UM2/smash8.3", "UM2/smash8-o", "da/da1", "da/da1.3", "da/da1.3.r9", "da/da1.r9", 
#             "UM3/UM3.r8", "UM3/UM3.r8-o", "UM3/UM3.r9", "UM3/UM3.r9-o", 
#             "UM/minus-neutralize", "UM/minusone", "UM/plusone", "UM/pt88step", "UM/smash5d", "UM/smash6", 
#             "UM/smash6.3", "UM/smash6.6d"]
# elif args.p == 5813:
#     dirs = ["UM2/smash7.3-Aug", "UM4/smash7.3-fhp2", "UM4/gly10-o-1k", "UM4/fdm-fmc/fdm-fmc-cold", 
#             "UM4/fdm-fmc/fdm-fmc-o", "UM4/fdm-fmc/fdm-fmc-c", "UM4/smash7.3-fewH", 
#             "UM4/hcn40/hcn40-c-r9", "UM4/hcn40/hcn40-c-r8", "UM4/hcn35/hcn35-o", "UM4/hcn35/hcn35-c"]
# else:
#     raise RuntimeError('Please hard-code more directories to analyze')

dirs = ["UM2/smash7.3", "UM2/smash7", 
        "UM2/smash7.3b", "UM2/smash7.3bo", "UM2/smash7.3-o", "UM2/smash7-m", "UM2/smash7-m-2t", 
        "UM2/smash8.3", "UM2/smash8-o", "da/da1", "da/da1.3", "da/da1.3.r9", "da/da1.r9", 
        "UM3/UM3.r8", "UM3/UM3.r8-o", "UM3/UM3.r9", "UM3/UM3.r9-o", 
        "UM/minus-neutralize", "UM/minusone", "UM/plusone", "UM/pt88step", "UM/smash5d", "UM/smash6", 
        "UM/smash6.3", "UM/smash6.6d", "UM2/smash7.3-Aug", 
        "UM4/co-n2", "UM4/fdm-fmc/fdm-fmc-c-big", "UM4/fdm-fmc/fdm-fmc-c", 
        "UM4/fdm-fmc/fdm-fmc-cold", "UM4/fdm-fmc/fdm-fmc-o", "UM4/gly10-c-2k", 
        "UM4/gly10-o-1k", "UM4/gly10-o-2k", "UM4/hcn35/hcn35-c", "UM4/hcn35/hcn35-o", 
        "UM4/hcn40/hcn40-c-r8", "UM4/hcn40/hcn40-c-r9", "UM4/hcn40/hcn40-o-r8", "UM4/hcn40/hcn40-o-r9", 
        "UM4/smash7.3-fewH", "UM4/smash7.3-fhp2", "UM4/bakrxn",
        "UM4.bak/fdm-fmc/fdm-fmc-c", "UM4.bak/hcn40/hcn40-c-r8",
        "UM4.bak/hcn35/hcn35-o", "UM4.bak/hcn35/hcn35-c",
        "PAH/phenbenz/ph1bz1gues"]

# Folder filter.

dirs = [i for i in dirs if i.startswith("PAH/phenbenz")]

dirs = ["ecig-manual/ecigrhfr6"]

# If we are doing a certain set of calculations, then we automatically read them.
if args.qg: args.rg = True
if args.qt: args.rt = True
if args.rt: args.rg = True
if args.qf: args.rf = True

RUN = False
if args.qa or args.qg or args.qt or args.qf:
    work_queue.set_debug_flag('all')
    wq = work_queue.WorkQueue(port=args.p, exclusive=False, shutdown=False)
    wq.specify_keepalive_interval(8640000)
    wq.specify_name('nanoreactor')
    print('Work Queue named %s listening on %d' % (wq.name, wq.port))
    RUN = True

def getN(xyz, frm):
    # Return a nanoreactor object corresponding to the selected frame of this xyz.
    cwd = os.getcwd()
    dnm, xyz = os.path.split(xyz)
    os.chdir(dnm)
    M = Molecule(xyz)
    M[frm].write('.tmp.xyz')
    NR = Nanoreactor('.tmp.xyz', enhance=1.2, printlvl=-1, boring=[])
    os.chdir(cwd)
    return NR

def chkstr(sxyz, chgspn=False):
    #----
    #| Analyze the results of a string or IRC path.
    #----
    chkstr.noreact = 0
    chkstr.rearrange = 0
    dts = os.path.split(sxyz)[0]
    E = np.array([float(line.split("=")[-1].split()[0]) for line in open(sxyz) if "Reaction" in line])
    NanoSR = getN(os.path.join(sxyz), 0)
    NanoSR0 = getN(os.path.join(dts, 'string.xyz'), 0)
    NanoSP = getN(os.path.join(sxyz), -1)
    NanoSP0 = getN(os.path.join(dts, 'string.xyz'), -1)
    chkstr.equals = NanoEqual(NanoSR, NanoSR0) and NanoEqual(NanoSP, NanoSP0)
    vxyz = os.path.splitext(sxyz)[0]+"_ev.xyz"
    if not os.path.exists(vxyz):
        SMol = Molecule(sxyz)
        SMol.align()
        ArcMol = SMol.pathwise_rmsd()
        ArcMolCumul = np.insert(np.cumsum(ArcMol), 0, 0.0)
        TotArc = ArcMolCumul[-1]
        dx = 0.01 # We want a 0.01 Angstrom separation between points.
        npts = int(max(ArcMolCumul)/dx)
        if npts == 0: 
            print "\x1b[91mIRC generated a zero-length path\x1b[0m"
            return None
        ArcMolEqual = np.linspace(0, max(ArcMolCumul), npts)
        xyzold = np.array(SMol.xyzs)
        xyznew = np.zeros((npts, xyzold.shape[1], xyzold.shape[2]), dtype=float)
        for a in range(SMol.na):
            for i in range(3):
                xyznew[:,a,i] = np.interp(ArcMolEqual, ArcMolCumul, xyzold[:, a, i])
        Enew = np.interp(ArcMolEqual, ArcMolCumul, E)
        SMol.comms = ["Intrinsic Reaction Coordinate: Energy = % .4f kcal/mol" % i for i in Enew]
        SMol.comms[np.argmax(Enew)] += " (Transition State)"
        SMol.xyzs = list(xyznew)
        SMol.write(vxyz)
    T = Nanoreactor(vxyz, enhance=1.2, printlvl=-1, boring=[], delay=0, learntime=0, rxntime=10, metastability=1.0, pcorrectemit=0.0)
    RIso0 = [i['graph'] for i in T.TimeSeries.values() if i['signal'][0][1]]
    PIso0 = [i['graph'] for i in T.TimeSeries.values() if i['signal'][-1][1]]
    RIso1 = [(i['graph'], tuple(i['graph'].L())) for i in T.TimeSeries.values() if i['signal'][0][1]]
    PIso1 = [(i['graph'], tuple(i['graph'].L())) for i in T.TimeSeries.values() if i['signal'][-1][1]]
    NoReact = set(RIso1) == set(PIso1)
    Rearr = set(RIso0) == set(PIso0)
    ostr = ""
    ostr += ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for i in T.TimeSeries.values() if i['signal'][0][1]])
    ostr += " -> "
    ostr += ', '.join(["%s (%s)" % (i['graph'].ef(), commadash(i['graph'].L())) for i in T.TimeSeries.values() if i['signal'][-1][1]])
    print "%-60s" % ostr, "  \x1b[95mE_rxn, E_a (kcal/mol) = % 8.3f % 8.3f\x1b[0m" % (E[-1] - E[0], max(E))
    if NoReact:
        chkstr.noreact = 1
        return None
    elif Rearr:
        chkstr.rearrange = 1
        return None

    satoms = itertools.chain(*[i['graph'].L() for i in T.TimeSeries.values() if i['signal'][-1][1] and i['signal'][0][1]])
    ratoms = sorted(list(set(range(T.na)) - set(satoms)))
    RP = T.atom_select(ratoms)[0] + T.atom_select(ratoms)[-1]
    RP.align()
    RP[0].write(os.path.join(dts,"reactants.xyz"))
    RP[-1].write(os.path.join(dts,"products.xyz"))

    if chgspn:
        # if not os.path.exists(os.path.join(dts,"charges.irc.txt")):
        #     cwd = os.getcwd()
        #     os.chdir(dts)
        #     subprocess.call("AnalyzeQChemIRC.py qchem_irc2.out string.xyz", shell=True)
        #     os.chdir(cwd)
        charges = np.loadtxt(os.path.join(dts,"charges.irc.txt"))
        spins = np.loadtxt(os.path.join(dts,"spins.irc.txt"))
        np.savetxt(os.path.join(dts,'chgsel.irc.txt'), charges[:,ratoms], fmt='% .5f')
        np.savetxt(os.path.join(dts,'spnsel.irc.txt'), spins[:,ratoms], fmt='% .5f')
        totchgsel = np.sum(charges[:,ratoms], axis=1)
        totspnsel = np.sum(spins[:,ratoms], axis=1)
        totchg, chgpass = extract_int(totchgsel, 0.3, 1.0, label="charge")
        totspn, spnpass = extract_int(np.abs(totspnsel), 0.3, 1.0, label="spin-z")
    else:
        sm    = Molecule(sxyz)
        srch  = lambda s : np.array([float(re.search('(?<=%s )[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?' % s, c).group(0)) for c in sm.comms])
        totchgs  = srch('charge') # An array of the net charge.
        totspns  = srch('sz')    # An array of the net Z-spin.
        totchg, chgpass = extract_int(totchgs, 0.3, 1.0, label="charge")
        totspn, spnpass = extract_int(np.abs(totspns), 0.3, 1.0, label="spin-z")
        zero  = np.zeros((len(sm), sm.na), dtype=float)
        np.savetxt(os.path.join(dts,'chgsel.irc.txt'), zero, fmt='% .5f')
        np.savetxt(os.path.join(dts,'spnsel.irc.txt'), zero, fmt='% .5f')
    return E[-1]-E[0], max(E), totchg, totspn
chkstr.equals = 0
chkstr.noreact = 0
chkstr.rearrange = 0

def get_fldr(tag):
    if tag.startswith('GS.'):
        dnm, init = tag.replace('GS.','').split('___')
        dnm += "/gs%i" % manage_gs.counts[dnm]
    elif tag.startswith('AN.'):
        f, dnm = tag.replace('AN.','').split('___')
    elif tag.startswith('TS.'):
        dnm, init, lstr = tag.replace('TS.','').split('___')
    elif tag.startswith('FS.'):
        dnm, init = tag.replace('FS.','').split('___')
    return dnm

def manage_wq(wq, wait_time=10):
    """ Waits for tasks to finish in the Work Queue and executes follow-up functions if necessary. """
    for sec in range(wait_time):
        task = wq.wait(1)
        if task:
            exectime = task.cmd_execution_time/1000000
            if task.result != 0:
                oldid = task.id
                oldhost = task.hostname
                taskid = wq.submit(task)
                logger.warning("Task '%s' (task %i) failed on host %s (%i seconds), resubmitted:"
                               "taskid %i\n" % (get_fldr(task.tag), oldid, oldhost, exectime, taskid))
            else:
                if exectime > 60:
                    logger.info("Task '%s' (task %i) finished successfully on host %s (%i seconds)\n"
                                % (get_fldr(task.tag), task.id, task.hostname, exectime))
                # Feed results back into the calculation.
                if task.tag.startswith('GS.'):
                    #----
                    # Analyze and resubmit the growing string calculation if required.
                    #----
                    d0, init = task.tag.replace('GS.','').split('___')
                    manage_gs(d0, init)
                elif task.tag.startswith('AN.'):
                    #----
                    # Manage the path analysis calculation.
                    #----
                    f, dnm = task.tag.replace('AN.','').split('___')
                    manage_an(f, dnm)
                elif task.tag.startswith('TS.'):
                    #----
                    # Manage the transition state calculation.
                    #----
                    dts, init, lstr = task.tag.replace('TS.','').split('___')
                    manage_ts(dts, init, lstr, False)
                
                del task
        if hasattr(wq.stats, 'workers_full'):
            nbusy = wq.stats.workers_busy + wq.stats.workers_full
        else:
            nbusy = wq.stats.workers_busy
        logger.info("%s : %i/%i workers busy; %i/%i jobs complete\r" % 
                    (time.ctime(), nbusy, (wq.stats.total_workers_joined - wq.stats.total_workers_removed),
                     wq.stats.total_tasks_complete, wq.stats.total_tasks_dispatched))
        if time.time() - manage_wq.t0 > 900:
            manage_wq.t0 = time.time()
            logger.info('\n')
manage_wq.t0 = time.time()

def get_valid(dnm, rxnm):
    # The archive file containing the result.
    fresult = os.path.join(dnm, 'result.tar.bz2')
    # The log file.
    flog = os.path.join(dnm, rxnm + ".log")
    if os.path.exists(fresult) and os.path.exists(flog):
        # Skip over calculations with known errors.  Return 0, which signifies that the segment is invalid.
        # The initial and final points converged to the same state.
        if os.path.exists(flog) and any(['There is nothing more to be done' in line for line in open(flog).readlines()]): return 0
        # The number of electrons was inconsistent with the spin-z.
        if os.path.exists(flog) and any(['inconsistent with the spin' in line for line in open(flog).readlines()]): return 0
        # The electronic structure calculations failed.
        if os.path.exists(flog) and any(['Too many failures' in line for line in open(flog).readlines()]): return 0
        # There are only two atoms (H2) so the contact calculation failed.
        if os.path.exists(flog) and any(['contacts must be an n x 2' in line for line in open(flog).readlines()]): return 0
    return 1
    
def anstat(dnm, rxnm):
    flog = os.path.join(dnm, rxnm + ".log")
    if os.path.exists(flog) and any(['converged to the same point' in line for line in open(flog).readlines()]): return 'No barrier'
    if os.path.exists(flog) and any(['inconsistent with the spin' in line for line in open(flog).readlines()]): return 'Charge/spin inconsistency'
    else: return 'Unknown'
    
def get_frags(dnm, rxnm, depth, max_depth=1):
    # A recursive function to find all of the segments that a reaction has been split into.
    if depth == max_depth:
        # If we've reached the maximum recursion depth, don't look for any further splits,
        # but we won't know whether this segment is "valid" or not (default to True)
        return 0
    # The archive file containing the result.
    fresult = os.path.join(dnm, 'result.tar.bz2')
    if not os.path.exists(fresult): return 0
    # The log file.
    flog = os.path.join(dnm, rxnm + ".log")
    dcxyz = os.path.join(dnm, rxnm+"_dc.xyz")
    s2xyz = os.path.join(dnm, rxnm+"_stage2.xyz")
    # Extract the archive file if it exists.
    if (os.path.exists(fresult) and os.path.exists(flog)) and get_valid(dnm, rxnm) and not os.path.exists(s2xyz):
        if os.path.getsize(fresult) > 0:
            os.system("tar xjf %s --directory %s" % (fresult, dnm))
    allxyz = sorted(glob.glob(os.path.join(dnm, rxnm + "*.xyz")))
    xyz0 = [os.path.join(dnm, rxnm + ".xyz")]
    dir0 = [dnm]
    splxyzs = [i for i in allxyz if re.match(".*_split[0-9]+(-[0-9]+){%i}\.xyz" % depth,i)]
    ssdirs = []
    for splxyz in splxyzs:
        ssdir = os.path.join(os.path.split(splxyz)[0], re.sub("^reaction_[0-9]*(_[0-9]+)?_","",os.path.split(splxyz)[1]).replace(".xyz","").replace("split","split_"))
        ssdirs.append(ssdir)
    for splxyz, ssdir in zip(splxyzs,ssdirs):
        ans = get_frags(ssdir, os.path.splitext(os.path.split(splxyz)[-1])[0], depth+1, max_depth)
        if ans == 0: continue
        if ans != 1:
            ssdirs_, splxyzs_, = ans
            ssdirs += ssdirs_
            splxyzs += splxyzs_
    if depth == 0:
        return dir0+ssdirs, xyz0+splxyzs
    else:
        return ssdirs, splxyzs

def replace_comms(stxyz, spldc):
    M0 = Molecule(stxyz)
    M1 = Molecule(spldc)
    firstcomm = None
    lastcomm = None
    for i in range(len(M0)):
        if "charge" in M0.comms[i]:
            if firstcomm == None:
                firstcomm = M0.comms[i]
            lastcomm = M0.comms[i]
        elif lastcomm != None:
            M0.comms[i] = lastcomm
    for i in range(len(M0)):
        if "charge" in M0.comms[i]: break
        M0.comms[i] = firstcomm
    for i, j in enumerate([int(i) for i in np.linspace(0,len(M0)-1,len(M1))]):
        M1.comms[i] = M0.comms[j]
    M1.write(spldc)

def manage_an(f, dnm):
    fn = os.path.split(f)[-1]
    # Don't analyze too many repeats of the same reaction
    if (len(fn.split('_')) == 3 and int(fn.split('_')[2].replace('.xyz','')) >= 5) : return
    rxnm = os.path.splitext(fn)[0]
    flog = os.path.join(dnm, rxnm + ".log")
    fresult = os.path.join(dnm, 'result.tar.bz2')
    anprint = args.qa or (not args.rt)
    calctype = "Path Analysis"
    if not os.path.exists(fresult):
        #----
        # Queue up analysis job if result not present.
        #----
        manage_an.n_an += 1
        pidf = os.path.join(dnm,'.runpid')
        pidn = int(open(pidf).readlines()[0].strip()) if os.path.exists(pidf) else 101010
        if pidn != os.getpid() and pid_exists(pidn):
            if anprint: print_stat(calctype, "running", dnm)
        if args.qa:
            if not os.path.exists(dnm):
                os.makedirs(dnm)
            shutil.copy2(f, os.path.join(dnm, fn))
            if anprint: print_stat(calctype, "Launch", dnm)
            with open(os.path.join(dnm,'.runpid'),'w') as f: print >> f, os.getpid()
            tag = fn+'___'+dnm
            queue_up_src_dest(wq,"python AnalyzeReaction.py %s.xyz b3lyp 6-31g* > %s.log 2> %s.err" % (rxnm, rxnm, rxnm),
                              input_files=[(os.path.join(scriptd,"AnalyzeReaction.py"), "AnalyzeReaction.py"),
                                           (os.path.join(dnm, fn),fn)],
                              output_files=[(os.path.join(dnm,"result.tar.bz2"),"result.tar.bz2"),
                                            (os.path.join(dnm,"%s.log" % rxnm),"%s.log" % rxnm),
                                            (os.path.join(dnm,"%s.err" % rxnm),"%s.err" % rxnm)], 
                              tag="AN.%s" % tag, verbose=False)
    else:
        #----
        # Manage results of analysis job.
        #----
        if any(["Path analysis is finished!" in line for line in open(flog)]):
            if anprint: print_stat(calctype, "finished", dnm)
        else:
            if anprint: 
                if anstat(dnm, rxnm) == 'No barrier':
                    print_stat(calctype, "no barrier", dnm, ansi='\x1b[33m')
                else:
                    print_stat(calctype, "failed", dnm, msg="("+anstat(dnm, rxnm)+")")
        s2xyz = os.path.join(dnm, rxnm+"_stage2.xyz")
        if not os.path.exists(s2xyz):
            if os.path.getsize(fresult) > 0:
                os.system("tar xjf %s --directory %s" % (fresult, dnm))
        ans = get_frags(dnm, rxnm, 0)
        if ans == 0: return
        ssdirs, splxyzs = ans
        for tup in zip(ssdirs, splxyzs):
            ssdir = tup[0]
            splxyz = tup[1]
            stxyz = os.path.splitext(splxyz)[0] + ("_stage2.xyz" if "split" not in ssdir else ".xyz")
            if not os.path.exists(stxyz):
                stxyz = os.path.splitext(splxyz)[0] + "_stage1.xyz"
            spldc  = os.path.splitext(splxyz)[0] + "_dc.xyz"
            valid = get_valid(ssdir, splxyz) and os.path.exists(spldc)
            if not os.path.exists(splxyz):
                raise Exception('%s does not exist!' % splxyz)
            if not os.path.exists(ssdir):
                os.makedirs(ssdir)
            if valid:
                if not os.path.exists(os.path.join(ssdir, os.path.split(spldc)[-1])):
                    shutil.copy2(spldc, ssdir)
                if not os.path.exists(os.path.join(ssdir, os.path.split(splxyz)[-1])):
                    shutil.copy2(splxyz, ssdir)
                replace_comms(stxyz, spldc)
                manage_gs(ssdir, spldc)
                manage_fs(ssdir, spldc)
manage_an.n_an = 0

def manage_gs(d0, init):
    if not args.rg: return
    count = manage_gs.counts[d0]
    lstr = init
    M0 = Molecule(init)
    # Tiny and trivial reactions screw us over, e.g. H2
    if M0.na < 3: return
    cvg_grad = 0.002
    max_iter = 100
    pgrad_grow = False

    calctype = 'Growing String'
    status = 'Launch'
    pgrad = 1000
    do_stable = True
    pgrad_thre = 0.05
    TS_Done = False
    #gsprint = args.qg or (args.rg and not args.rt)
    gsprint = args.rg
    GS_Max = 30
    while True:
        status = 'Launch'
        niter = 0
        dgs = os.path.join(d0, "gs%i" % count)
        manage_gs.counts[d0] = count
        if not os.path.exists(dgs): break
        fout = os.path.join(dgs,'gs.log')
        arch = os.path.join(dgs,'gs_result.tar.bz2')
        pidf = os.path.join(dgs,'.runpid')
        pidn = int(open(pidf).readlines()[0].strip()) if os.path.exists(pidf) else 101010
        if os.path.exists(fout) and os.path.exists(arch):
            status = 'Unknown'
            # Error message in the event of a failed job
            errmsg = ""
            # Perpendicular gradient (conveged at 0.002)
            pgrad = 1000
            #------
            # Parse the growing string output file.
            #------
            for line in open(fout):
                if "MAX_PERP_GRAD_FOR_A_NODE" in line:
                    pgrad = float(line.split()[2])
                    niter += 1
                if "Reached Maximum iterations, exiting" in line:
                    status = 'Continue'
                    break
                if "You have finished the Growing String run" in line:
                    status = 'Converged'
                    break
                if "global name 're' is not defined" in line:
                    status = 'Failed'
                    errmsg = 'Syntax error'
                if "MAX reached in spline interpolation" in line:
                    status = 'Failed'
                    errmsg = 'Spline interpolation error'
                    break
                if "is inconsistent with the spin-z" in line:
                    status = 'Failed'
                    errmsg = 'Charge inconsistent with spin'
                    break
                if ("Failed to find HF/KS stable state" in line or "Too many attempts at finding a HF-stable state" in line):
                    status = 'Failed'
                    errmsg = 'Stability analysis failure'
                    do_stable = False
                    break
            #------
            # If the calculation did not fail, then extract the archive.
            #------
            if status != 'Failed':
                if os.path.getsize(os.path.join(dgs,"gs_result.tar.bz2")) > 0:
                    diff_time = 1
                    if os.path.exists(os.path.join(dgs,".extracted")):
                        diff_time = os.stat(os.path.join(dgs,"gs_result.tar.bz2")).st_mtime - os.stat(os.path.join(dgs,".extracted")).st_mtime
                    if diff_time > 0 or not os.path.exists(os.path.join(dgs, 'tsestimate.xyz')):
                        os.system("tar xjf %s --directory %s" % (os.path.join(dgs,"gs_result.tar.bz2"),dgs))
                        os.system("touch %s" % (os.path.join(dgs,".extracted")))
                    strs = sorted(glob.glob(os.path.join(dgs, "str*.xyz")))
                    lstr = strs[-1] if len(strs) > 0 else init
                    dcan = [i for i in os.listdir(dgs) if i not in [os.path.split(lstr)[1], 'Vfile.txt', 'tsestimate.xyz']]
                    for line in os.popen("tar -tf %s" % os.path.join(dgs, "gs_result.tar.bz2")):
                        if line.strip() in dcan:
                            os.remove(os.path.join(dgs, line.strip()))
                else:
                    status = 'Failed'
                    errmsg = 'Zero-size archive file'
            if status == 'Converged' and pgrad > cvg_grad:
                print "Puzzle in %s (converged but gradient %f)" % (dgs, pgrad)
                raw_input()
            if status == 'Unknown':
                if pgrad <= cvg_grad:
                    status = 'Converged'
                elif niter >= max_iter:
                    status = 'Continue'
            strs = sorted(glob.glob(os.path.join(dgs, "str*.xyz")))
            manage_gs.pgrads[d0].append(pgrad)
            if len(manage_gs.pgrads[d0]) >= 5 and manage_gs.pgrads[d0][-1] == max(manage_gs.pgrads[d0][-3:]):
                pgrad_grow = True
        #------
        # The calculation should not run in directories where other jobs are running, but it should not block itself either.
        #------
        if pidn != os.getpid() and pid_exists(pidn):
            if status not in ['Converged', 'Continue']:
                status = 'Running'
        if status == 'Continue':
            if gsprint: print_stat(calctype, "continue", dgs, msg="(%.3f grad after %i iterations)" % (pgrad, niter))
            lstr = strs[-1]
            if pgrad < pgrad_thre:
                dts = os.path.join(d0, "gs%i" % count + "_ts")
                TS_Done = TS_Done or manage_ts(dts, init, lstr, False)
        elif status == 'Running':
            if gsprint: print_stat(calctype, "running", dgs)
        elif status == 'Launch':
            break
        elif status == 'Converged':
            if gsprint: print_stat(calctype, "converged", dgs, msg="(%i iterations)" % niter)
            lstr = strs[-1]
            dts = os.path.join(d0, "gs%i" % count + "_ts")
            TS_Done = TS_Done or manage_ts(dts, init, lstr, True)
            break
        elif status == 'Failed':
            if gsprint: print_stat(calctype, "failed", dgs, msg="(%i iterations; %s)" % (niter, errmsg))
            if errmsg in ['Stability analysis failure', 'Syntax error']:
                if errmsg == 'Stability analysis failure' : do_stable = False
                if gsprint: print "\x1b[96m--== Will resubmit ==--\x1b[0m"
                for f in ['gs.log', 'gs_result.tar.bz2']:
                    absf = os.path.join(dgs, f)
                    movef(absf, absf+".bak")
                status = 'Launch'
            break
        elif status == 'Unknown':
            if gsprint: print_stat(calctype, "Unknown status", dgs, msg="(%i iterations)" % niter, ansi="\x1b[1;91m")
            Launch = False
            break
        else:
            raise RuntimeError('Unexpected status: %s' % status)
        count += 1

    #----
    # Launch the next iteration of growing string.
    #----
    if status == 'Launch':
        if pgrad < pgrad_thre:
            dts = os.path.join(d0, "gs%i" % (count-1) + "_ts")
            TS_Done = TS_Done or manage_ts(dts, init, lstr, True)
        #----
        # Look at the transition state even before growing string converges.
        #----
        # If the transition state calculation already succeeded, no need for growing string.
        if TS_Done:
            print_stat(calctype, "terminated", dgs, msg="(Found transition state)", ansi='\x1b[96m')
            return
        elif count >= GS_Max:
            print_stat(calctype, "terminated", dgs, msg="(Too many cycles)", ansi='\x1b[96m')
            return
        elif pgrad_grow:
            print_stat(calctype, "terminated", dgs, msg="(pgrad not decreasing)", ansi='\x1b[96m')
            return
        manage_gs.tot_gs += 1
        if args.qg:
            if not os.path.exists(dgs):
                os.makedirs(dgs)
            shutil.copy2(lstr, os.path.join(dgs, "initial.xyz"))
            print_stat(calctype, "Launch", dgs)
            with open(os.path.join(dgs,'.runpid'),'w') as f: print >> f, os.getpid()
            tag = d0+'___'+init
            queue_up_src_dest(wq, "python growing-string.py %s initial.xyz b3lyp 6-31g* > gs.log 2> gs.err" % ('-stab' if do_stable else ''),
                              input_files=[(os.path.join(scriptd,"growing-string.py"),"growing-string.py"),
                                           (os.path.join(scriptd,"gstring.exe"),"gstring.exe"),
                                           (os.path.join(scriptd,"qcgs.py"),"qcgs.py"),
                                           ("%s/initial.xyz" % dgs,"initial.xyz")],
                              output_files=[(os.path.join(dgs,"gs_result.tar.bz2"),"gs_result.tar.bz2"),
                                            (os.path.join(dgs,"gs.log"),"gs.log"),
                                            (os.path.join(dgs,"gs.err"),"gs.err")],
                              tag="GS.%s" % tag, verbose=False)
        else:
            if gsprint: print_stat(calctype, "ready", dgs)
manage_gs.tot_gs = 0
manage_gs.counts = defaultdict(int)
manage_gs.pgrads = defaultdict(list)

def manage_ts(dts, init, lstr, Launch):
    if dts in manage_ts.looked: return manage_ts.looked[dts]
    if not os.path.exists(dts) and (not Launch): return 0
    if not args.rt: return 0
    #----
    # Read in the initial growing string xyz for reaction info.
    #----
    M0 = Molecule(init)
    if M0.na < 3: return 0

    #----
    # Growing string directory contains string energies and transition state estimate.
    #----
    dgs = dts.replace("_ts","")
    if not os.path.exists(os.path.join(dgs, "tsestimate.xyz")): return 0
    tse = Molecule(os.path.join(dgs, "tsestimate.xyz"))[-1]
    tse.comms[0] = M0.comms[0]
    V = [float(line.strip()) for line in os.popen("tac %s 2> /dev/null | awk '(p && NF==0) {exit} (NF>0) {p=1; print $3}' | tac" % os.path.join(dgs, "Vfile.txt")).readlines()]

    #----
    # TS_Success indicates a successful transition state found
    #----
    TS_Success = False

    #----
    # TS_Finished indicates calculation finished (IRC endpoints consistent with string endpoints)
    #----
    TS_Finished = False

    chkstr.rearrange = 0
    chkstr.noreact = 0
    #----
    # Now begins the management of the TS calculation
    #----
    def write_result(xyz, string=False):
        """ Write results from an IRC XYZ file to a .png image. """
        # Temporarily doesn't work for growing string.
        if string: return 0
        if write_result.first:
            print
            write_result.first = 0
        if string:
            print "\x1b[1;95m GS:  \x1b[0m",
            chgspn = False
        else:
            print "\x1b[1;97m IRC: \x1b[0m",
            chgspn = True
        res = chkstr(xyz, chgspn)
        if res != None:
            delta = res[0]
            barrier = res[1]
            chg = res[2]
            spn = res[3]
            cwd = os.getcwd()
            os.chdir(dts)
            if (string and args.pic >= 2) or (not string and args.pic >= 1):
                subprocess.call("xyzob.py %s %s %s %s %.3f %.3f %s" % ("reactants.xyz", "products.xyz", "chgsel.irc.txt", "spnsel.irc.txt", 
                                                                       delta, barrier, "1" if string else ""), shell=True)
                print "\x1b[92mReaction saved\x1b[0m to %s/reaction.png%s" % (dts, " (warning: TS from string)" if string else "")
            os.chdir(cwd)
            if string: touch("%s/TS_From_String" % dts)
            else: rmf("%s/TS_From_String" % dts)
            return 1
        else:
            return 0
    write_result.first = 1

    tslog = os.path.join(dts, 'ts.log')
    gsxyz = os.path.join(dts, 'string.xyz')
    ircxyz = os.path.join(dts, 'string.irc.xyz')

    #----
    # If transition state doesn't exist, provide interim results from growing string.
    #----
    Exist = False
    if not os.path.exists(tslog):
        write_result(gsxyz, string=True)

    if dts in manage_ts.launched:
        Launch = False

    calctype = "TS calculation"
    IRC_Failure = False
    if os.path.exists(tslog) and os.path.exists(os.path.join(dts,'ts_result.tar.bz2')):
        Exist = True
        Launch = False
        if not os.path.exists(ircxyz) and os.path.getsize(os.path.join(dts,'ts_result.tar.bz2')) > 0:
            diff_time = 1
            if os.path.exists(os.path.join(dts,".extracted")):
                diff_time = os.stat(os.path.join(dts,"ts_result.tar.bz2")).st_mtime - os.stat(os.path.join(dts,".extracted")).st_mtime
            if diff_time > 0:
                os.system("tar xjf %s --directory %s string.irc.xyz charges.irc.txt spins.irc.txt 2> /dev/null" % (os.path.join(dts,"ts_result.tar.bz2"),dts))
                os.system("touch %s" % (os.path.join(dts,".extracted")))
            # dcan = [i for i in os.listdir(dts) if 'qchem' in i]
            # for line in os.popen("tar -tf %s" % os.path.join(dts, "ts_result.tar.bz2")):
            #     if line.strip() in dcan:
            #         os.remove(os.path.join(dts, line.strip()))

        if any(['IRC Failure' in line for line in open(tslog).readlines()]):
            IRC_Failure = True
            errmsg = []
            for line in open(tslog).readlines():
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
            print_stat(calctype, "Failure", dts, msg=("(%s)" % ', '.join(errmsg)) if len(errmsg) > 0 else '', ansi="\x1b[1;91m")
        elif not os.path.exists(ircxyz):
            #----
            # TS calculation finished but unfortunately did not provide an IRC coordinate
            #----
            print_stat(calctype, "Failure", dts, msg="(no IRC)", ansi="\x1b[1;91m")
            IRC_Failure = True
            errs = []
            if os.path.exists(tslog):
                for line in os.popen("grep -i 'error\|failure' %s | grep -v '===='" % tslog).readlines():
                    errs.append(line.replace('\n',''))
            if len(errs) > 0:
                print "--- Errors: ---"
                for line in errs: print line
                print "--- End Errors ---"
            write_result(gsxyz, string=True)
        else:
            write_result(gsxyz, string=True)
            if os.path.exists(ircxyz): make_monotonic(ircxyz)
            ircstat = write_result(ircxyz)
            TS_Success = ircstat
        if IRC_Failure and args.dt: # Delete some files from archives
            manage_ts.looked[dts] = 0
            bigfiles = [line.strip() for line in os.popen("tar -tvf %s | awk '($3>1000000) {print $NF}'" % os.path.join(dts, "ts_result.tar.bz2")).readlines()]
            if len(bigfiles) > 0:
                os.system("bunzip2 %s" % os.path.join(dts, "ts_result.tar.bz2"))
                os.system("tar --delete -f %s %s" % (os.path.join(dts, "ts_result.tar"), ' '.join(bigfiles)))
                os.system("bzip2 %s" % os.path.join(dts, "ts_result.tar"))

    running = 0

    if TS_Success:
        if chkstr.equals:
            TS_Finished = True
            print_stat(calctype, "=- Success! -=", dts, ansi="\x1b[1;92m")
        else:
            print_stat(calctype, "=- Success? -=", dts, ansi="\x1b[1;92m", msg="(Different molecules found)")
    else:
        pidf = os.path.join(dts,'.runpid')
        pidn = int(open(pidf).readlines()[0].strip()) if os.path.exists(pidf) else 101010
        if pidn != os.getpid() and pid_exists(pidn):
            running = 1
            Launch = 0
        # Force launch all TS
        if args.ft and (not Exist) and (dts not in manage_ts.launched): 
            Launch = 1
        if Launch:
            manage_ts.tot_ts += 1
            #----
            # Write the starting files for the TS calculation.
            #----
            M1 = Molecule(lstr)
            for i in range(len(M1)):
                M1.comms[i] = M0.comms[min(len(M0)-1, (len(M0)*i)/len(M1))] + "; E = % .3f kcal/mole" % (V[i])
            if max(V) > 200:
                print_stat(calctype, "skipping", dts, msg="(barrier too high = %.1f)" % max(V), ansi="\x1b[91m")
            elif args.qt:
                if not os.path.exists(dts):
                    os.makedirs(dts)
                M1.write(os.path.join(dts,'string.xyz'))
                tse.write(os.path.join(dts,"tsest.xyz"))
                print_stat(calctype, "Launch", dts)
                with open(os.path.join(dts,'.runpid'),'w') as f: print >> f, os.getpid()
                queue_up_src_dest(wq,"python transition-state.py tsest.xyz string.xyz &> ts.log",
                                  input_files=[(os.path.join(scriptd, "transition-state.py"),"transition-state.py"),
                                               ("%s/tsest.xyz" % dts,"tsest.xyz"),
                                               ("%s/string.xyz" % dts,"string.xyz")],
                                  output_files=[(os.path.join(dts,"ts_result.tar.bz2"),"ts_result.tar.bz2"),
                                                (os.path.join(dts,"ts.log"),"ts.log")], verbose=False,
                                  tag='TS.'+'___'.join([dts, init, lstr]))
            else:
                print_stat(calctype, "ready", dts)
        if Exist and not IRC_Failure:
            if chkstr.rearrange:
                print_stat(calctype, "Rearrangement", dts, ansi="\x1b[1;93m")
                if chkstr.equals: TS_Finished = True
            elif chkstr.noreact:
                print_stat(calctype, "No Reaction", dts, ansi="\x1b[1;93m")
                if chkstr.equals: TS_Finished = True
            elif running:
                print_stat(calctype, "running", dts)
        elif not Exist and not Launch:
            errmsg = ""
            if os.path.exists(tslog):
                for line in open(tslog):
                    if line.startswith("Exception: Calculation encountered a fatal error"):
                        errmsg += line.replace(')\n','').split('(')[1]
            if len(errmsg) > 0:
                manage_ts.looked[dts] = 0
            print_stat(calctype, "no data", dts, msg=("(%s)" % errmsg) if len(errmsg) > 0 else '', ansi="\x1b[91m")
                
    if Launch:
        manage_ts.launched.append(dts)
    if chkstr.noreact:
        os.system("touch %s/No_Reaction" % dts)
    if Exist: manage_ts.looked[dts] = 0
    if TS_Finished: manage_ts.looked[dts] = 1
    return TS_Finished
        
manage_ts.tot_ts = 0
manage_ts.looked = {}
manage_ts.launched = []

def manage_fs(d0, init):
    if not args.rf: return
    dfs = os.path.join(d0, "fs")
    # if os.path.exists(os.path.join(dfs, '.looked')) : return
    if not os.path.exists(dfs):
        os.makedirs(dfs)
    shutil.copy2(init, os.path.join(dfs, 'string.xyz'))
    #----
    # Read in the initial growing string xyz for reaction info.
    #----
    M0 = Molecule(init)
    if M0.na < 3: return
    #----
    # FS_Success indicates a successful transition state found
    #----
    FS_Success = False
    FS_Finished = False
    chkstr.rearrange = 0
    chkstr.noreact = 0
    def gather_error(log):
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
        return errmsg

    #----
    # now begins the management of the FS calculation
    #----
    def write_result(xyz):
        """ Write results from an IRC XYZ file to a .png image. """
        # Temporarily doesn't work for growing string.
        if write_result.first:
            print
            write_result.first = 0
        print "\x1b[1;97m IRC: \x1b[0m",
        chgspn = True
        res = chkstr(xyz, chgspn)
        if res != None:
            delta = res[0]
            barrier = res[1]
            chg = res[2]
            spn = res[3]
            cwd = os.getcwd()
            os.chdir(dfs)
            if args.pic >= 1:
                subprocess.call("xyzob.py %s %s %s %s %.3f %.3f" % ("reactants.xyz", "products.xyz", "chgsel.irc.txt", "spnsel.irc.txt", 
                                                                       delta, barrier), shell=True)
                print "\x1b[92mReaction saved\x1b[0m to %s/reaction.png" % (dfs)
            os.chdir(cwd)
            return 1
        else:
            return 0
    write_result.first = 1
    fslog = os.path.join(dfs, 'fs.log')
    ircxyz = os.path.join(dfs, 'string.irc.xyz')
    Exist = False
    calctype = "Freezing String"
    IRC_Failure = False
    Launch = True
    if os.path.exists(fslog):
        Exist = True
        Launch = False
        if not os.path.exists(ircxyz) and os.path.exists(os.path.join(dfs, 'fs_result.tar.bz2')) and os.path.getsize(os.path.join(dfs,'fs_result.tar.bz2')) > 0:
            diff_time = 1
            if os.path.exists(os.path.join(dfs,".extracted")):
                diff_time = os.stat(os.path.join(dfs,"fs_result.tar.bz2")).st_mtime - os.stat(os.path.join(dfs,".extracted")).st_mtime
            if diff_time > 0:
                os.system("tar xjf %s --directory %s string.irc.xyz charges.irc.txt spins.irc.txt 2> /dev/null" % (os.path.join(dfs,"fs_result.tar.bz2"),dfs))
                os.system("touch %s" % (os.path.join(dfs,".extracted")))
            # dcan = [i for i in os.listdir(dfs) if 'qchem' in i]
            # for line in os.popen("tar -tf %s" % os.path.join(dfs, "fs_result.tar.bz2")):
            #     if line.strip() in dcan:
            #         os.remove(os.path.join(dfs, line.strip()))
        if any([('IRC Failure' in line or 'fatal error' in line) for line in open(fslog).readlines()]):
            IRC_Failure = True
            errmsg = gather_error(fslog)
            print_stat(calctype, "Failure", dfs, msg=("(%s)" % ', '.join(errmsg)) if len(errmsg) > 0 else '', ansi="\x1b[1;91m")
        elif not os.path.exists(ircxyz):
            #----
            # TS calculation finished but unfortunately did not provide an IRC coordinate
            #----
            print_stat(calctype, "Failure", dfs, msg="(no IRC)", ansi="\x1b[1;91m")
            IRC_Failure = True
            errs = []
            if os.path.exists(fslog):
                for line in os.popen("grep -i 'error\|failure' %s | grep -v '===='" % fslog).readlines():
                    errs.append(line.replace('\n',''))
            if len(errs) > 0:
                print "--- Errors: ---"
                for line in errs: print line
                print "--- End Errors ---"
        else:
            # if os.path.exists(ircxyz): make_monotonic(ircxyz)
            ircstat = write_result(ircxyz)
            FS_Success = ircstat
    running = 0
    if FS_Success:
        if chkstr.equals:
            FS_Finished = True
            print_stat(calctype, "=- Success! -=", dfs, ansi="\x1b[1;92m", msg="(Same molecules)")
        else:
            print_stat(calctype, "=- Success? -=", dfs, ansi="\x1b[1;92m", msg="(Different molecules found)")
        Launch = 0
    elif Exist:
        pidf = os.path.join(dfs,'.runpid')
        pidn = int(open(pidf).readlines()[0].strip()) if os.path.exists(pidf) else 101010
        if pidn != os.getpid() and pid_exists(pidn):
            running = 1
            Launch = 0
    manage_fs.tot_fs += 1

    if Launch:
        if not os.path.exists(dfs):
            os.makedirs(dfs)
        ixyz = os.path.split(init)[1]
        if args.qf:
            print_stat(calctype, "Launch", dfs)
            with open(os.path.join(dfs,'.runpid'),'w') as f: print >> f, os.getpid()
            queue_up_src_dest(wq,"python freezing-string.py %s &> fs.log" % ixyz,
                              input_files=[(os.path.join(scriptd, "freezing-string.py"),"freezing-string.py"),
                                           (init, ixyz)],
                              output_files=[(os.path.join(dfs,"fs_result.tar.bz2"),"fs_result.tar.bz2"),
                                            (os.path.join(dfs,"fs.log"),"fs.log")], verbose=False,
                              tag='FS.'+'___'.join([dfs, ixyz]))
        else:
            print_stat(calctype, "ready", dfs)

    if Exist and not IRC_Failure:
        if chkstr.rearrange:
            print_stat(calctype, "Rearrangement", dfs, ansi="\x1b[1;93m", msg = "(Different molecules found)" if not chkstr.equals else "(Same molecules)")
            if chkstr.equals: FS_Finished = True
        elif chkstr.noreact:
            print_stat(calctype, "No Reaction", dfs, ansi="\x1b[1;93m", msg = "(Different molecules found)" if not chkstr.equals else "(Same molecules)")
            if chkstr.equals: FS_Finished = True
        elif running:
            print_stat(calctype, "running", dfs)

    if chkstr.noreact: touch(os.path.join(dfs, 'No_Reaction'))
    if Exist: touch(os.path.join(dfs, '.looked'))
    return FS_Finished

    #     manage_fs.launched.append(dfs)
    # if Exist: manage_fs.looked[dfs] = 0
    # if FS_Finished: manage_fs.looked[dfs] = 1
    # return FS_Finished
        
manage_fs.tot_fs = 0
manage_fs.looked = {}
manage_fs.launched = []

def main():
    fn = 0
    for dnm0 in dirs:
        refd = os.path.join(dnm0, "Refinement")
        if not os.path.exists(refd):
            os.makedirs(refd)
        fl = sorted(glob.glob(os.path.join(dnm0, "Consolidated", "reaction_*.xyz")))
        # Loop through all of the "reaction_*.xyz" files in the "Consolidated" directory.
        for f in fl:
            # Look for the corresponding directory in "Refinement".
            dnm = os.path.splitext(f)[0].replace("Consolidated", "Refinement")
            # if "smash7.3-Aug/Refinement/reaction_105" not in dnm: continue
            # if "360" not in dnm: continue
            manage_an(f, dnm)
    
            # rxnm = os.path.splitext(os.path.split(f)[-1])[0]
            # # There should be a "result.tar.bz2" from the previous analysis step.
            # do_frags(dnm, rxnm)
            # ans = get_frags(dnm, rxnm, 0)
            # if ans == 0: continue
            fn += 1
            if RUN and (fn%100 == 0): manage_wq(wq,wait_time=1)
        
    print "%i molecular dynamics paths are being analyzed" % manage_an.n_an
    print "%i growing string calculations" % manage_gs.tot_gs
    print "%i transition states from growing string" % manage_ts.tot_ts
    print "%i freezing string calculations" % manage_fs.tot_fs
    
    if RUN: 
        while True:
            manage_wq(wq,wait_time=5)
        print "All jobs are finished!"
    else:
        print "Dry run finished"

if __name__ == "__main__":
    main()
