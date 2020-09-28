#!/usr/bin/env python

import os, sys
import numpy as np
from itertools import islice
from collections import OrderedDict

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

LPW note 2019-03-04: This script was made obsolete by a significantly
revised version written in late February 2019.
"""

fs2au = 41.3413733365614

class Chunk(object):
    def __init__(self, dnm, fnm='run.out', init=False):
        self.havedata = False
        self.dt = None
        self.dnm = dnm
        outfnm = os.path.join(dnm, fnm)
        self.frames = OrderedDict()
        self.na = None
        self.elem = None
        ssq = 0.0
        efict = 0.0
        if os.path.exists(outfnm):
            for line in open(outfnm):
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
                if line.startswith("t = ") or line.startswith("=MD= t ="):
                    s = line.replace("=MD=","").split()
                    time_fs = round(float(s[2])/fs2au, 3)
                    stepn = time_fs / self.dt
                    frame = OrderedDict([('frame', int(round(stepn))), ('time', time_fs), ('ssq', ssq), ('temp', float(s[14])), ('potential', float(s[6])), ('kinetic', float(s[10])), ('efict', efict), ('energy', float(s[18]))])
                    # Frames written to disk are integer multiples of the time step
                    if np.abs(stepn-round(stepn)) < 0.001:
                        self.frames[int(round(stepn))] = frame
                        self.havedata = True
                    # ssq = float(line.split()[2]) # What is this?
                    efict = 0.0
        self.fseq = self.frames.keys()
        if self.havedata and (self.fseq != range(self.fseq[0], self.fseq[-1]+1)):
            raise RuntimeError("Sequence of frames is not contiguous")

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

    np.savetxt(os.path.join("gathered", "properties.txt"), fdata_arr, fmt="%7i %11.3f %11.6f % 14.6f % 14.6f % 11.6f % 14.6f % 14.6f",
               header="%5s %11s %11s %14s %14s %11s %14s %14s" % ("Frame", "Time(fs)", "S-squared", "Temperature", "Potential", "Kinetic", "Electron-KE", "Total-Energy"))

if __name__ == "__main__":
    main()
