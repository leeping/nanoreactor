#!/usr/bin/env python

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
                    latest_frame['n_scf'] = latest_frame.get('n_scf', 0) + scf_cycles
                    latest_frame['n_adiis'] = latest_frame.get('n_adiis', 0) + adiis_cycles
                    latest_frame['n_grad'] = latest_frame.get('n_grad', 0) + 1
                    latest_frame['n_carpar'] = latest_frame.get('n_carpar', 0) + (efict > 0.0)
                    latest_frame['scftime'] = latest_frame.get('scftime', 0) + scf_time
                    # On the first step, "Time per MD step" is not present
                    efict = 0.0
                    if stepn == 0:
                        latest_frame['walltime'] = scf_time
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
                if self.monitor and "Taking action" in line:
                    action_type = line.split()[-2]
                    if "Normal Mode" not in line:
                        latest_frame['recover'] = min(latest_frame.get('recover', 0) + 1, 1)
                    else:
                        latest_frame['recover'] = latest_frame.get('recover', 0)
                    monitor_action = True
        self.fseq = self.frames.keys()
        if self.havedata and (self.fseq != range(self.fseq[0], self.fseq[-1]+1)):
            raise RuntimeError("Sequence of frames is not contiguous")
        for fnum in self.frames.keys():
            self.frames[fnum] = OrderedDict(self.frames[fnum].items() + self.cumul[fnum].items())

    def __repr__(self):
        return "MD Chunk: frames %i -> %i" % (self.fseq[0], self.fseq[-1])

    def writexyz(self, start=-1, end=-1, mode='a'):
        fxyz = open(os.path.join(self.dnm, 'scr', 'coors.xyz'))
        oxyz = open(os.path.join("gathered", "trajectory.xyz"), mode=mode)
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
        fkeep = range(start, end)
        while True:
            if len(fkeep) == 0: break
            xyzframe = list(islice(fxyz, self.na+2))
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
                chg = [float(i) for i in chgframe.split()]
                spn = [float(i) for i in spnframe.split()] if have_spin else [0.0 for i in chgframe.split()]
                popframe = ["%-5i\n" % self.na, xyzframe[1]]
                for i in range(self.na):
                    popframe.append("%-5s % 11.6f % 11.6f 0\n" % (self.elem[i],chg[i],spn[i]))
                opop.writelines(popframe)
                if have_bo: obo.writelines(boframe)
                print "\rWriting frame %i      " % fnum,
            if fnum == fkeep[-1]: break
        fxyz.close()
        oxyz.close()
        opop.close()
        return fdata
            
def main():
    if not os.path.exists("gathered"): os.makedirs("gathered")
    cnum = 0
    chunks = []
    framedata = []
    while True:
        chunk = Chunk("chunk_%04i" % cnum)
        if not chunk.havedata: break
        chunks.append(chunk)
        print chunk
        cnum += 1
    for i in range(len(chunks)-1):
        framedata += chunks[i].writexyz(start=chunks[i].fseq[0], end=chunks[i+1].fseq[0], mode='w' if i == 0 else 'a')
    framedata += chunks[-1].writexyz(mode='a')
    fdata_arr = np.array([[]+d.values() for d in framedata])
    for i, d in enumerate(framedata):
        if i > 0 and keys != d.keys():
            print keys, d.keys()
            raise RuntimeError("Dictionary keys do not match for frames %i and %i" % (i-1, i))
        keys = d.keys()

    # Remove irrelevant data arrays that are all zero
    keep = []
    for i, k in enumerate(keys):
        if (fdata_arr[:,i] != np.zeros_like(fdata_arr[:,i])).any():
            keep.append(i)
    keys = [keys[i] for i in keep]
    fdata_arr = fdata_arr[:, np.array(keep)]

    # 3-tuple of format string, header format, header title
    key_info = {'frame' : ("Frame", "%7i", "%5s"),
                'time' : ("Time(fs)", "%11.3f", "%11s"),
                'ssq' : ("S-squared", "%11.6f", "%11s"),
                'temp' : ("Temperature", "% 14.6f", "%14s"),
                'potential' : ("Potential", "% 14.6f", "%14s"),
                'kinetic' : ("Kinetic", "% 11.6f", "%11s"),
                'efict' : ("Electron-KE", "% 14.6f", "%14s"),
                'energy' : ("Total-Energy", "% 14.6f", "%14s"),
                'n_scf' : ("N(SCF)", "%7i", "%7s"),
                'n_adiis' : ("N(ADIIS)", "%7i", "%7s"),
                'n_grad' : ("N(Grad)", "%7i", "%7s"),
                'n_carpar' : ("N(CPMD)", "%7i", "%7s"),
                'scftime' : ("SCFTime", "%8.2f", "%8s"),
                'walltime' : ("WallTime", "%8.2f", "%8s"),
                'recover' : ("Recover", "%7i", "%7s")}
    fmt = ' '.join([key_info[k][1] for k in keys])
    header = ' '.join([(key_info[k][2] % key_info[k][0]) for k in keys])
    np.savetxt(os.path.join("gathered", "properties.txt"), fdata_arr, fmt=fmt, header=header)

if __name__ == "__main__":
    main()
