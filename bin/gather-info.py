#!/usr/bin/env python

from __future__ import print_function
import os, sys, math, re
import numpy as np
from itertools import islice
from collections import defaultdict, OrderedDict

"""
In a folder that looks like this:
chunk_0000
chunk_0000/run.out
chunk_0000/scr/coors.xyz
chunk_0000/scr/grad.xyz
chunk_0000/scr/bond_order.list
chunk_0000/scr/charge.xls
chunk_0000/scr/spin.xls
chunk_0000/scr/vel.log
chunk_0001/...

Produce files like this:
gathered/coors.xyz
gathered/bond_order.list
gathered/charge-spin.txt
gathered/properties.txt
"""

fs2au = 41.3413733365614

def load_bondorder(boin, thre, traj_length):
    """
    Load a bondorder.list file.  

    This file format only lists bond orders above a threshold (typically 0.1)
    in each frame. Thus, the returned data takes the form of a sparse array.

    Copied from nanoreactor.py to lessen dependency

    Parameters
    ----------
    boin : str
        Name of the bond_order.list file
    thre : float
        Floating 
    traj_length : int
        Length of the trajectory
    
    Returns
    -------
    OrderedDict
        Dictionary that maps zero-indexed atom pairs (a1, a2) to numpy array
        containing the bond order between a1, a2 for each frame. 
        Keys only include a2 > a1
    """
    boMode = 0
    boFrame = -1
    boSparse = OrderedDict()
    keys = []
    for ln, line in enumerate(open(boin).readlines()):
        if boMode == 0:
            nbo = int(line.strip())
            boLine = ln
            boMode = 1
            boFrame += 1
        elif boMode == 1:
            if ln > boLine+1:
                s = line.split()
                a1 = int(s[0])
                a2 = int(s[1])
                bo = float(s[2])
                a1, a2 = sorted((a1, a2))
                if a1 != a2:
                    if (a1, a2) not in boSparse:
                        boSparse[(a1, a2)] = np.zeros(traj_length, dtype=float)
                        keys.append((a1, a2))
                    boSparse[(a1, a2)][boFrame] = bo
                if ln == boLine+nbo+1:
                    boMode = 0
    sortkeys = sorted(keys)
    boSparse_sorted = OrderedDict([(k, boSparse[k]) for k in sortkeys if np.max(boSparse[k]) > thre])

    return boSparse_sorted

class Chunk(object):
    def __init__(self, dnm, fnm='run.out', init=False):
        self.havedata = False
        self.dt = None
        self.dnm = dnm
        outfnm = os.path.join(dnm, fnm)
        self.frames = OrderedDict()
        # A dictionary like self.frames, except it can be "ahead" if needed (e.g. when microiterations are active)
        self.cumul = OrderedDict()
        latest_macro = 0
        self.na = None
        self.elem = None
        ssq = 0.0
        efict = 0.0
        scf_mode = False
        self.monitor = False
        action_type = "Normal"
        scf_cycles = 0
        adiis_cycles = 0
        scf_time = 0.0
        if os.path.exists(outfnm):
            for line in open(outfnm):
                if "CPMonitor" in line:
                    self.monitor = True
                if "Total atoms:" in line:
                    self.na = int(line.split()[2])
                if "time step:" in line:
                    if self.dt is not None:
                        raise RuntimeError("Read time step twice")
                    self.dt = float(line.split("time step:")[1].split()[0])
                if "SPIN S-SQUARED" in line:
                    ssq = float(line.split()[2])
                if "Electron Fictitious Kinetic Energy" in line:
                    efict = float(line.split()[-1])
                if "Start SCF Iterations" in line:
                    scf_mode = True
                    adiis_mode = False
                elif scf_mode:
                    if re.match('^[1-9].*[1-9]\.[0-9]+$', line.strip()):
                        s = line.split()
                        scf_cycles += 1
                        if adiis_mode:
                            adiis_cycles += 1
                        scf_time += float(s[-1])
                    elif "going to ADIIS" in line or "ADIIS will be performed until convergence threshold" in line:
                        adiis_mode = True
                    elif "more DIIS steps" in line:
                        adiis_mode = False
                    elif "FINAL ENERGY" in line:
                        scf_mode = False
                        adiis_mode = False
                if line.startswith("t = ") or line.startswith("=MD= t ="):
                    s = line.replace("=MD=","").split()
                    time_fs = round(float(s[2])/fs2au, 4)
                    stepn = time_fs / self.dt
                    macro = int(math.ceil(stepn))
                    latest_macro = max(latest_macro, macro)
                    latest_macro = macro
                    frame = OrderedDict([('frame', int(round(stepn))), ('time', time_fs), ('ssq', ssq), ('temp', float(s[14])), ('potential', float(s[6])), ('kinetic', float(s[10])), ('efict', efict), ('energy', float(s[18]))])
                    # Frames written to disk are integer multiples of the time step
                    if np.abs(stepn-round(stepn)) < 0.0001:
                        self.frames[int(round(stepn))] = frame
                        self.havedata = True
                    latest_frame = self.cumul.get(latest_macro, OrderedDict())
                    self.cumul[latest_macro] = latest_frame
                    if latest_frame.get('n_scf', 0) == 0:
                        latest_frame['bo_scf'] = scf_cycles
                    latest_frame['n_scf'] = latest_frame.get('n_scf', 0) + scf_cycles
                    latest_frame['n_adiis'] = latest_frame.get('n_adiis', 0) + adiis_cycles
                    latest_frame['n_grad'] = latest_frame.get('n_grad', 0) + 1
                    latest_frame['n_carpar'] = latest_frame.get('n_carpar', 0) + (efict > 0.0)
                    latest_frame['scftime'] = latest_frame.get('scftime', 0) + scf_time
                    # On the first step, "Time per MD step" is not present
                    efict = 0.0
                    if stepn == 0:
                        latest_frame['walltime'] = scf_time
                        latest_frame['bo_time'] = scf_time
                        if self.monitor:
                            latest_frame['recover'] = 0
                    # "2" is because CPMonitor action not printed on the first step
                    if len(self.frames) > 2 and self.monitor: 
                        if not monitor_action:
                            raise RuntimeError("Failed to parse CPMonitor action")
                    monitor_action = False
                    scf_cycles = 0
                    adiis_cycles = 0
                    scf_time = 0.0
                if "Time per MD step" in line:
                    latest_frame['walltime'] = latest_frame.get('walltime', 0.0) + float(line.split()[-2])
                    if latest_frame.get('bo_time', 0) == 0:
                        latest_frame['bo_time'] = float(line.split()[-2])
                if self.monitor and "Taking action" in line:
                    action_type = line.split()[-2]
                    if "Normal Mode" not in line:
                        latest_frame['recover'] = min(latest_frame.get('recover', 0) + 1, 1)
                    else:
                        latest_frame['recover'] = latest_frame.get('recover', 0)
                    monitor_action = True
        self.fseq = list(self.frames.keys())
        if self.havedata and (self.fseq != list(range(self.fseq[0], self.fseq[-1]+1))):
            raise RuntimeError("Sequence of frames is not contiguous")
        for fnum in list(self.frames.keys()):
            self.frames[fnum] = OrderedDict(list(self.frames[fnum].items()) + list(self.cumul[fnum].items()))

    def __repr__(self):
        return "MD Chunk: frames %i -> %i" % (self.fseq[0], self.fseq[-1])

    def writexyz(self, start=-1, end=-1, mode='a'):
        fxyz = open(os.path.join(self.dnm, 'scr', 'coors.xyz'))
        vxyz = open(os.path.join(self.dnm, 'scr', 'vel.log'))
        oxyz = open(os.path.join("gathered", "trajectory.xyz"), mode=mode)
        ovxyz = open(os.path.join("gathered", "velocity.xyz"), mode=mode)
        fdata = []
        fchg = open(os.path.join(self.dnm, 'scr', 'charge.xls'))
        fchg.readline()
        have_spin = False
        if os.path.exists(os.path.join(self.dnm, 'scr', 'spin.xls')):
            fspn = open(os.path.join(self.dnm, 'scr', 'spin.xls'))
            fspn.readline()
            have_spin = True
        opop = open(os.path.join("gathered", "charge-spin.txt"), mode=mode)
        have_bo = False
        if os.path.exists(os.path.join(self.dnm, 'scr', 'bond_order.list')):
            have_bo = True
            fbo = open(os.path.join(self.dnm, 'scr', 'bond_order.list'))
            obo = open(os.path.join("gathered", "bond_order.list"), mode=mode)
        if start == -1: start = self.fseq[0]
        if end == -1: end = self.fseq[-1]+1
        fkeep = list(range(start, end))
        while True:
            if len(fkeep) == 0: break
            xyzframe = list(islice(fxyz, self.na+2))
            velframe = list(islice(vxyz, self.na+2))
            chgframe = fchg.readline()
            if have_spin: spnframe = fspn.readline()
            if self.elem is None:
                self.elem = [line.split()[0] for line in xyzframe[2:]]
            if xyzframe == []: 
                break
            if have_bo:
                numbo = fbo.readline()
                # print(self.dnm, numbo)
                boframe = [numbo] + [fbo.readline() for i in range(int(numbo.strip())+1)]
            fnum = int(xyzframe[1].split()[1])
            if fnum in fkeep:
                fdata.append(self.frames[fnum])
                oxyz.writelines(xyzframe)
                ovxyz.writelines(velframe)
                chg = [float(i) for i in chgframe.split()]
                spn = [float(i) for i in spnframe.split()] if have_spin else [0.0 for i in chgframe.split()]
                popframe = ["%-5i\n" % self.na, xyzframe[1]]
                for i in range(self.na):
                    popframe.append("%-5s % 11.6f % 11.6f 0\n" % (self.elem[i],chg[i],spn[i]))
                opop.writelines(popframe)
                if have_bo: obo.writelines(boframe)
                print("\rWriting frame %i      " % fnum, end=' ')
            if fnum == fkeep[-1]: break
        fxyz.close()
        oxyz.close()
        ovxyz.close()
        opop.close()
        return fdata

def bo_frame_change(traj_length):
    fbo = os.path.join("gathered", "bond_order.list")
    if not os.path.exists(fbo): return
    boSparse_sorted = load_bondorder(fbo, 0.1, traj_length)
    dm_all = None
    for k, v in list(boSparse_sorted.items()):
        amask = np.ma.array(v, mask=(v==0.0))
        am_fut = amask[1:]
        am_now = amask[:-1]
        dm = np.ma.abs(am_fut - am_now)
        if dm_all is None:
            dm_all = dm.copy()
        else:
            dm_all = np.ma.vstack((dm_all, dm.copy()))
    maxVals = np.ma.max(dm_all, axis=0)
    maxArgs = np.ma.argmax(dm_all, axis=0)
    pairs = list(boSparse_sorted.keys())
    pair1 = np.array([pairs[i][0] for i in maxArgs])
    pair2 = np.array([pairs[i][1] for i in maxArgs])

    maxVals = np.append(maxVals, 0)
    pair1 = np.append(pair1, 0)
    pair2 = np.append(pair2, 0)

    return maxVals, pair1, pair2
            
def main():
    if not os.path.exists("gathered"): os.makedirs("gathered")
    cnum = 0
    chunks = []
    framedata = []
    while True:
        chunk = Chunk("chunk_%04i" % cnum)
        if not chunk.havedata: break
        chunks.append(chunk)
        print(chunk)
        cnum += 1
    print("Writing concatenated trajectory ...")
    last_chunk_mode = 'w'
    for i in range(len(chunks)-1):
        framedata += chunks[i].writexyz(start=chunks[i].fseq[0], end=chunks[i+1].fseq[0], mode='w' if i == 0 else 'a')
        last_chunk_mode = 'a'
    framedata += chunks[-1].writexyz(mode=last_chunk_mode)
    fdata_arr = np.array([[]+list(d.values()) for d in framedata])
    for i, d in enumerate(framedata):
        if i > 0 and keys != list(d.keys()):
            print(keys, list(d.keys()))
            raise RuntimeError("Dictionary keys do not match for frames %i and %i" % (i-1, i))
        keys = list(d.keys())

    # Remove irrelevant data arrays that are all zero
    keep = []
    for i, k in enumerate(keys):
        if (fdata_arr[:,i] != np.zeros_like(fdata_arr[:,i])).any():
            keep.append(i)
    keys = [keys[i] for i in keep]
    fdata_arr = fdata_arr[:, np.array(keep)]

    bo_change = True
    if os.path.exists(os.path.join("gathered", "bond_order.list")) and bo_change:
        print()
        print("Computing per-frame bond order changes ...")
        maxVal, maxPair1, maxPair2 = bo_frame_change(fdata_arr.shape[0])
        fdata_arr = np.hstack((fdata_arr, maxVal.reshape(-1, 1), maxPair1.reshape(-1, 1), maxPair2.reshape(-1, 1)))
        keys += ['bo_maxd', 'bo_a1', 'bo_a2']
        
    # 3-tuple of format string, header format, header title
    key_info = OrderedDict([('frame', ("Frame", "%7i", "%5s")),
                            ('time', ("Time(fs)", "%11.3f", "%11s")),
                            ('ssq', ("S-squared", "%11.6f", "%11s")),
                            ('temp', ("Temperature", "% 14.6f", "%14s")),
                            ('potential', ("Potential", "% 14.6f", "%14s")),
                            ('kinetic', ("Kinetic", "% 11.6f", "%11s")),
                            ('efict', ("Electron-KE", "% 14.6f", "%14s")),
                            ('energy', ("Total-Energy", "% 14.6f", "%14s")),
                            ('n_scf', ("N(SCF)", "%7i", "%7s")),
                            ('n_adiis', ("N(ADIIS)", "%7i", "%7s")),
                            ('n_grad', ("N(Grad)", "%7i", "%7s")),
                            ('n_carpar', ("N(CPMD)", "%7i", "%7s")),
                            ('scftime', ("SCFTime", "%8.2f", "%8s")),
                            ('walltime', ("WallTime", "%8.2f", "%8s")),
                            ('bo_scf', ("BOMD-NSCF", "%9i", "%9s")),
                            ('bo_time', ("BOMD-Time", "%9.2f", "%9s")),
                            ('recover', ("Recover", "%7i", "%7s")),
                            ('bo_maxd', ("BO-MaxD", "%7.3f", "%7s")),
                            ('bo_a1', ("BO-At1", "%6i", "%6s")),
                            ('bo_a2', ("BO-At2", "%6i", "%6s"))])
    for k in keys:
        if k not in key_info:
            raise RuntimeError('Please put %s in key_info' % k)
    # fmt = ' '.join([key_info[k][1] for k in key_info if k in keys])
    # header = ' '.join([(key_info[k][2] % key_info[k][0]) for k in key_info if k in keys])
    fmt = []
    header = []
    keyOrder = []
    for k in key_info:
        if k in keys:
            if 'recover' not in keys and k in ['bo_scf', 'bo_time']: continue
            keyOrder.append(keys.index(k))
            fmt.append(key_info[k][1])
            header.append(key_info[k][2] % key_info[k][0])
    fmt = ' '.join(fmt)
    header = ' '.join(header)
    keyOrder = np.array(keyOrder)
    np.savetxt(os.path.join("gathered", "properties.txt"), fdata_arr[:, keyOrder], fmt=fmt, header=header)

if __name__ == "__main__":
    main()
