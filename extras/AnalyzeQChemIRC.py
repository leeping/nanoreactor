#!/usr/bin/env python

import warnings
warnings.simplefilter('ignore')
from nanoreactor.molecule import *

def GetRMSD(mol, frame, xtrial):
    tmp = mol[frame]
    tmp.xyzs.append(xtrial)
    tmp.align()
    RMSD = np.sqrt(np.mean((tmp.xyzs[1] - tmp.xyzs[0]) ** 2))
    return RMSD

def monotonic(arr, start, end):
    # Make sure an array is monotonically decreasing from the start to the end.
    a0 = arr[start]
    i0 = start
    if end > start:
        i = start+1
        while i < end:
            if arr[i] < a0:
                arr[i0:i+1] = np.linspace(a0, arr[i], i-i0+1)
                a0 = arr[i]
                i0 = i
            i += 1
    if end < start:
        i = start-1
        while i >= end:
            if arr[i] < a0:
                arr[i:i0+1] = np.linspace(arr[i], a0, i0-i+1)
                a0 = arr[i]
                i0 = i
            i -= 1

def main():
    if len(sys.argv) < 3:
        print "To run this script: %s [qchem IRC output] [initial xyz path]" % __file__
        sys.exit()

    QCIRC = Molecule(sys.argv[1], errok=['SCF failed to converge', 'Maximum optimization cycles reached'])
    M = Molecule(sys.argv[2], errok=['SCF failed to converge'])
    FwdX = QCIRC.Irc['FwdX']
    BakX = QCIRC.Irc['BakX']
    FwdE = QCIRC.Irc['FwdE']
    BakE = QCIRC.Irc['BakE']
    FwdQ = QCIRC.Irc['FwdQ']
    BakQ = QCIRC.Irc['BakQ']
    FwdSz = QCIRC.Irc['FwdSz']
    BakSz = QCIRC.Irc['BakSz']
    print "Length of the IRC segments        = %6i %6i" % (len(FwdX), len(BakX))
    RMSD1 = GetRMSD(M, 0, FwdX[-1]) + GetRMSD(M, -1, BakX[-1])
    RMSD2 = GetRMSD(M, 0, BakX[-1]) + GetRMSD(M, -1, FwdX[-1])
    print "IRC RMSD to endpoints (fwd, bkwd) = %6.3f %6.3f" % (RMSD1, RMSD2)
    StrE = [float(c.split("=")[-1].split()[0]) for c in M.comms]
    if RMSD1 < RMSD2:
        O = M[0]
        M.xyzs = FwdX[::-1]
        M.xyzs += BakX[1:]
        E = FwdE[::-1] + BakE[1:]
        E = np.array(E)
        E -= E[0]
        E *= 627.51
        monotonic(E, len(FwdX)-1, len(E)-1)
        monotonic(E, len(FwdX)-1, 0)
        Q = FwdQ[::-1] + BakQ[1:]
        Sz = FwdSz[::-1] + BakSz[1:]
        M.comms = ["Intrinsic Reaction Coordinate: Energy = % .4f kcal/mol" % i for i in E]
        M.comms[len(FwdX)-1] += " (Transition State)"
        M.align_center()
        M.write(os.path.splitext(sys.argv[2])[0] + ".irc.xyz")
    elif RMSD1 >= RMSD2:
        O = M[0]
        M.xyzs = BakX[::-1]
        M.xyzs += FwdX[1:]
        E = BakE[::-1] + FwdE[1:]
        E = np.array(E)
        E -= E[0]
        E *= 627.51
        monotonic(E, len(BakX)-1, len(E)-1)
        monotonic(E, len(BakX)-1, 0)
        Q = BakQ[::-1] + FwdQ[1:]
        Sz = BakSz[::-1] + FwdSz[1:]
        M.comms = ["Intrinsic Reaction Coordinate: Energy = % .4f kcal/mol" % i for i in E]
        M.comms[len(BakX)-1] += " (Transition State)"
        M.align_center()
        M.write(os.path.splitext(sys.argv[2])[0] + ".irc.xyz")
    print "  IRC  reaction energy, barrier (kcal/mol)   = % 8.3f % 8.3f" % (E[-1] - E[0], max(E))
    print "String reaction energy, barrier (kcal/mol)   = % 8.3f % 8.3f" % (StrE[-1] - StrE[0], max(StrE))
    np.savetxt('charges.irc.txt', np.array(Q), fmt='% .5f')
    np.savetxt('spins.irc.txt', np.array(Sz), fmt='% .5f')

if __name__ == "__main__":
    main()
