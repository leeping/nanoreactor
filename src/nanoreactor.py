#!/usr/bin/env python

import warnings
# Suppress warnings from Molecule class.
warnings.simplefilter("ignore")
import os, sys, re
import networkx as nx
import numpy as np
import copy
import ast
from collections import OrderedDict, defaultdict, Counter
from chemistry import Elements, Radii
from copy import deepcopy
from molecule import Molecule, format_xyz_coord, extract_int
import contact
import itertools
import time
from viterbi import viterbi, _viterbi, viterbi_skl
from pkg_resources import parse_version
try:
    from sklearn.hmm import MultinomialHMM
    have_sklearn = 1
except: 
    print "Cannot import Hidden Markov Models: load Intel compiler environment or consider pip install scikit-learn"
    have_sklearn = 0

## Names of colors from VMD
ColorNames = ["blue", "red", "gray", "orange", "yellow", 
              "tan", "silver", "green", "white", "pink", 
              "cyan", "purple", "lime", "mauve", "ochre",
              "iceblue", "black", "yellow2", "yellow3", "green2",
              "green3", "cyan2", "cyan3", "blue2", "blue3", 
              "violet", "violet2", "magenta", "magenta2", "red2", 
              "red3", "orange2", "orange3"]

def wildmatch(fmatch, formulas):
    for fpat in formulas:
        if re.sub('\?', '', fpat) == re.sub('[0-9]', '', fmatch):
            return True
    return False

def subscripts(string):
    unicode_integers = {'0': 8320, '1': 8321, '2': 8322,
                        '3': 8323, '4': 8324, '5': 8325,
                        '6': 8326, '7': 8327, '8': 8328,
                        '9': 8329}
    ustr = unicode(string)
    for i in unicode_integers:
        ustr = ustr.replace(i, unichr(unicode_integers[i]))
    return ustr

def nodematch(node1,node2):
    # Matching two nodes of a graph.  Nodes are equivalent if the elements are the same
    return node1['e'] == node2['e']

def encode(l): 	
    """ 
    Run length encoding of a time series. 

    Turns [0, 0, 1, 1, 1, 0, 0, 0, 0] into: [(2, 0), (3, 1), (4, 0)]
    """
    return [[len(list(group)),name] for name, group in itertools.groupby(l)]

def decode(e):
    """
    Decode a run-length encoded list, returning the original time series (opposite of 'encode' function).

    Parameters
    ----------
    e : list
        Encoded list consisting of 2-tuples of (length, element)

    Returns
    -------
    list
        Decoded time series.
    """
    return sum([length * [item] for length,item in e],[]) 

def element(e, elem):
    begin = 0
    end = 0
    for i in e:
        end += i[0]
        if elem >= begin and elem < end:
            return i[1]
        begin = end

def append_e(e, i):
    if len(e) == 0:
        e = [[1,i]]
    elif e[-1][1] == i:
        e[-1][0] += 1
    else:
        e.append([1,i])
    return e

def segments(e):
    # Takes encoded input.
    begins = np.array([sum([k[0] for k in e][:j]) for j,i in enumerate(e) if i[1] == 1])
    lens = np.array([i[0] for i in e if i[1] == 1])
    return [(i, i+j) for i, j in zip(begins, lens)]

def commadash(l):
    # Formats a list like [27, 28, 29, 30, 31, 88, 89, 90, 91, 100, 136, 137, 138, 139]
    # into '27-31,88-91,100,136-139
    L = sorted(l)
    if len(L) == 0:
        return "(empty)"
    L.append(L[-1]+1)
    LL = [i in L for i in range(L[-1])]
    return ','.join('%i-%i' % (i[0],i[1]-1) if (i[1]-1 > i[0]) else '%i' % i[0] for i in segments(encode(LL)))

def uncommadash(s):
    # Does the opposite of commadash
    return list(itertools.chain(*[range(int(i.split('-')[0]), int(i.split('-')[1])+1) if len(i.split('-')) > 1 else [int(i)] for i in s.split(',')]))

def longest_segment(e):
    # Takes encoded input.
    begins = np.array([sum([k[0] for k in e][:j]) for j,i in enumerate(e) if i[1] == 1])
    lens = np.array([i[0] for i in e if i[1] == 1])
    if len(begins) == 0:
        return 0, 0
    longest_begin = begins[np.argmax(lens)]
    longest_end = longest_begin + np.max(lens)
    return longest_begin, longest_end

def longest_lifetime(e):
    b, e = longest_segment(e)
    return e - b

def exists_at_time(e, t):
    """
    Given a run-length encoded list, return whether this list is "1" at the specified time.
    """
    t0 = 0
    for l, x in e:
        t1 = t0 + l
        if t > t0 and t < t1:
            return x
        t0 = t1
    return x
        
def Rectify(signal, metastability, pcorrectemission):
    """
    Rectify a time series of true / false variables using Hidden Markov Model.
    Thanks to Robert McGibbon

    Note: In newer versions of sklearn, HMM will be deprecated.  At
    that time we may need to install a new package, or simply install
    sklearn version 0.15 or older.
    """
    if have_sklearn == 0:
        return None
    if metastability <= 0 or pcorrectemission <= 0:
        return None
    signal_ = np.array(decode(signal))
    # Inputs: 
    # Signal: A time series of true / false variables.
    # mestability: Probability that the underlying signal continues to be True(False) if the current value is True(False)
    # In principle possible to have two different values, but in practice we just use one.
    # Relatively strong rectification is 0.999 (stronger is closer to 1)
    # pcorrectemission: Probability that the measured signal is a correct prediction of the underlying signal
    # Relatively strong rectivation is 0.6 (stronger is lower)
    # Higher metastability + lower pcorrect = longer segments (more aggressive model).
    transmat = np.array([[metastability, 1-metastability], [1-metastability, metastability]])
    emission = np.array([[pcorrectemission, 1-pcorrectemission], [1-pcorrectemission, pcorrectemission]])
    hmm = MultinomialHMM(n_components=2, startprob=[0.5, 0.5], transmat=transmat)
    hmm.emissionprob_ = emission
    corrected = hmm.predict(signal_)
    return encode(corrected)

def FillGaps(signal, delay):
    # Erase short gaps of 'False' in a time series.
    # Input and output are encoded.
    filled = []
    for ichunk, chunk in enumerate(signal):
        if (ichunk > 0) and (ichunk < len(signal)-1) and (chunk[0] <= delay) and (chunk[1] == 0):
            filled.append([chunk[0], 1])
        else:
            filled.append(chunk)
    return encode(decode(filled))

def bondlist_tcl(bondlist):
    # Print out the list of bonds in a format that VMD can understand.
    Answer = ""
    for i, b in enumerate(bondlist):
        if i > 0:
            Answer += " "
        if len(b) == 0:
            Answer += "{}"
        elif len(b) == 1:
            Answer += "%i" % b[0]
        elif len(b) > 12:
            Answer += "{%s}" % ' '.join(["%i" % j for j in b[:12]])
        else:
            Answer += "{%s}" % ' '.join(["%i" % j for j in b])
    return Answer

def NanoEqual(Nano1, Nano2):
    GraphEqual = Counter([i['graph'] for i in Nano1.TimeSeries.values()]) == Counter([i['graph'] for i in Nano2.TimeSeries.values()])
    AtomEqual = Counter([tuple(i['graph'].L()) for i in Nano1.TimeSeries.values()]) == Counter([tuple(i['graph'].L()) for i in Nano2.TimeSeries.values()])
    return GraphEqual and AtomEqual

def make_monotonic(xyz, others=[]):
    M = Molecule(xyz)
    new_others = []
    StrE = [float(c.split("=")[-1].split()[0]) for c in M.comms]
    iTS = ["Transition State" in c for c in M.comms].index(1)
    E = StrE[iTS]
    selct = [iTS]
    for i in range(iTS, -1, -1):
        if StrE[i] < E:
            selct.append(i)
            E = StrE[i]
    E = StrE[iTS]
    for i in range(iTS, len(M), 1):
        if StrE[i] < E:
            selct.append(i)
            E = StrE[i]
    selct = sorted(selct)
    M[np.array(selct)].write(xyz)
    new_others = []
    for lst in others:
        new_others.append(np.array(lst)[np.array(selct)])
    return new_others
            
class MyG(nx.Graph):
    def __init__(self):
        super(MyG,self).__init__()
        self.Alive = True
    def __eq__(self, other):
        # This defines whether two MyG objects are "equal" to one another.
        if not self.Alive:
            return False
        if not other.Alive:
            return False
        return nx.is_isomorphic(self,other,node_match=nodematch)
    def __hash__(self):
        ''' The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. '''
        return 1
    def L(self):
        ''' Return a list of the sorted atom numbers in this graph. '''
        return sorted(self.nodes())
    def AStr(self):
        ''' Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . '''
        return ','.join(['%i' % i for i in self.L()])
    def e(self):
        ''' Return an array of the elements.  For instance ['H' 'C' 'C' 'H']. '''
        elems = nx.get_node_attributes(self,'e')
        return [elems[i] for i in self.L()]
    def ef(self):
        ''' Create an Empirical Formula '''
        Formula = list(self.e())
        return ''.join([('%s%i' % (k, Formula.count(k)) if Formula.count(k) > 1 else '%s' % k) for k in sorted(set(Formula))])
    def x(self):
        ''' Get a list of the coordinates. '''
        coors = nx.get_node_attributes(self,'x')
        return np.array([coors[i] for i in self.L()])
    def writexyz(self, fnm, center=False):
        ''' Return a list of strings corresponding to an XYZ file. '''
        out = []
        na = len(self.e())
        out.append("%5i" % na)
        out.append(self.ef())
        x = self.x()
        e = self.e()
        for i in range(na):
            out.append(format_xyz_coord(e[i], x[i]))
        if center:
            x -= x.mean(0)
        with open(fnm,'w') as f: f.writelines([i+'\n' for i in out])

class Nanoreactor(Molecule):
    def __init__(self, xyzin=None, qsin=None, ftype=None, stride=1, enhance=1.4, mindist=1.0, printlvl=0, boring=['all'], disallow=[], learntime=200, padtime=0, extract=False, frames=0, xyzout='out.xyz', metastability=0.999, pcorrectemit=0.6, saverxn=True, neutralize=False):
        #==========================#
        #   Load in the XYZ file   #
        #==========================#
        if printlvl >= 0: print "Loading molecule ...",
        if xyzin == None:
            raise Exception('Nanoreactor must be initialized with an .xyz file as the first argument')
        super(Nanoreactor,self).__init__(xyzin, ftype)
        if qsin != None and os.path.exists(qsin):
            if printlvl >= 0:
                print "Loading charge and spin populations...",
            QS = Molecule(qsin, ftype="xyz")
            self.Charges = np.array([xyz[:, 0] for xyz in QS.xyzs])
            self.Spins = np.array([xyz[:, 1] for xyz in QS.xyzs])
        else:
            self.Charges = np.array([[0 for i in range(self.na)] for j in range(len(self))])
            self.Spins = np.array([[0 for i in range(self.na)] for j in range(len(self))])
        if printlvl >= 0: print "Done"
        #==========================#
        #         Settings         #
        #==========================#
        # Enhancement factor for determining whether two atoms are bonded
        self.Fac = enhance
        # Set the number of frames.
        self.Frames = self.ns if frames == 0 else frames
        # Set the frame skip
        self.stride = stride
        # Switch for whether to extract molecules.
        self.extract = extract
        # Switch for whether to save chemical reactions.
        self.saverxn = saverxn
        # Switch for printing make-movie.tcl ; this is not necessary and may be deprecated soon
        self.Render = False
        # Boring molecules to be excluded from coloring (by empirical formula)
        # Note: Isomers formed later are still considered interesting.  This is a hack.
        self.BoringFormulas = set(boring)
        self.BoringIsomerIdxs = set()
        # Disallow certain molecules from being in the TimeSeries.
        self.DisallowedFormulas = set(disallow)
        if printlvl >= 1: print self.DisallowedFormulas, "is disallowed"
        # The print level (control the amount of printout)
        self.printlvl = printlvl
        # List of favorite colors for VMD coloring (excluding reds)
        self.CoolColors = [23, 32, 11, 19, 3, 13, 15, 27, 22, 6, 4, 12, 7, 9, 10, 28, 17, 21, 26, 24, 18]
        # Molecules that live for at least this long (x stride) will be colored in VMD.
        # Also, molecules that vanish for (less than) this amount of time will have their time series filled in.
        self.LearnTime = learntime
        if padtime != 0:
            self.PadTime = padtime
        else:
            self.PadTime = self.LearnTime/4
        # Hidden Markov Model settings, look at Rectify function
        self.metastability = metastability
        self.pcorrectemission = pcorrectemit
        # Whether to extract molecules to neutralize the system
        self.neutralize = neutralize
        #==========================#
        #   Initialize Variables   #
        #==========================#
        # If the number of frames is too big, reduce it here
        if self.Frames > self.ns:
            self.Frames = self.ns
        # Create an atom-wise list of covalent radii.
        R = np.array([(Radii[Elements.index(i)-1] if i in Elements else 0.0) for i in self.elem])
        # Dictionary of how long each molecule lives.
        self.TimeSeries = OrderedDict()
        # List of unique isomers.
        self.Isomers = []
        # Dictionary that aggregate isomers
        self.isomer_ef_iidx_dict = defaultdict(list)
        # Set of isomers that are RECORDED.
        self.Recorded = set()
        # A time-series of atom-wise isomer labels.
        self.IsoLabels = []
        self.IsoLocks = []
        # A time series of VMD-formatted bond specifications.
        self.BondLists = []
        # A time series of lists of bond lengths.
        self.BondLengths = []
        # A time series of the number of molecules.
        self.NumMols = []
        # Robert's testing stuff for the Viterbi algorithm.
        self.TestVit = False
        self.GidSets = []
        self.CodeBook = []
        # An iterator over all atom pairs, for example: [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        self.AtomIterator = np.array(list(itertools.combinations(range(self.na),2)))
        # A list of threshold distances for each atom pair for determining whether two atoms are bonded
        self.BondThresh = np.array([max(mindist,(R[i[0]] + R[i[1]]) * self.Fac) for i in self.AtomIterator])
        # Build graphs from the distance matrices (this is the bottleneck)
        if self.printlvl >= 0: print "Building graphs..."
        for i in range(0, self.Frames, self.stride):
            self.dxij = contact.atom_distances(np.array([self.xyzs[i]]),self.AtomIterator)
            if self.printlvl >= 0: print "\rFrame %-7i:" % i,
            self.AddFrame(i)
        if self.printlvl >= 0: print
        if self.printlvl >= 0: print "Done building graphs."
        # Determine whether the Output method creates a new xyz file.
        if xyzout.upper() != 'NONE':
            self.fout = xyzout
            self.bWrite = True
        else:
            self.fout = xyzin
            self.bWrite = False
        self.RectifyTimeSeries()
        # Create a mapping from isomer number to VMD color.
        self.ColorIdx = self.Analyze()

    def AddFrame(self, frame):
        RawG = self.MakeGraphFromXYZ(frame)
        MolGphs = [RawG.subgraph(c).copy() for c in nx.connected_components(RawG)]
        # MolGphs = nx.connected_component_subgraphs(RawG)
        ilabels = [0 for i in range(self.na)]
        nowgids = []
        efs = []
        bonds = [[] for i in range(self.na)]
        NumMols = 0
        for G in MolGphs:
            G.__class__ = MyG
            G.Alive = True
            NumMols += 1
            ef = G.ef()
            # iidx means Isomer Index.
            # compare to the Graph that has the same Empirical Formula
            for i in self.isomer_ef_iidx_dict[ef]:
                if self.Isomers[i] == G:
                    iidx = i
                    break
            else:
                iidx = len(self.Isomers)
                self.Isomers.append(G)
                self.isomer_ef_iidx_dict[ef].append(iidx)
                self.IsoLocks.append(False)
            if (ef in self.BoringFormulas or wildmatch(ef, self.BoringFormulas) or (frame == 0 and 'ALL' in [i.upper() for i in self.BoringFormulas])):
                self.BoringIsomerIdxs.add(iidx)
            # if G not in self.Isomers:
            gid = G.AStr()+":%i" % iidx
            nowgids.append(gid)
            efs.append(ef)
            if gid not in self.TimeSeries and ef not in self.DisallowedFormulas:
                self.TimeSeries[gid] = {'graph':G,'iidx':iidx,'raw_signal':encode([0 for i in range(frame)]),'lock':False}
            for j in G.nodes():
                ilabels[j] = iidx
        if self.printlvl >= 3:
            efuniq = []
            pops = []
            for i in set(efs):
                efuniq.append(i)
                pops.append(sum(np.array(efs)==i))
            MList = [(efuniq[i], pops[i]) for i in np.argsort(np.array(pops))[::-1]]
            # print "\r%10s : %-10s" % ("Molecule","Count")
            if frame == 0:
                fout = open('ef.txt','w')
            else:
                fout = open('ef.txt','a')
            sout = ', '.join(["%s" % ("%i(" % i[1] if i[1] > 1 else "") + subscripts(i[0]) + (")" if i[1] > 1 else "") for i in MList])
            print sout
            print >> fout, sout.encode('utf8')
            fout.close()
            # print >> fout, u"%s" % subscripts(sout)
            # for i in MList:
            #     print "%10s : %-10i" % (i[0],i[1])

        # build data for Viterbi algorithm.
        if self.TestVit:
            GidSet = set(nowgids)
            try:
                code = self.GidSets.index(GidSet)
            except:
                code = len(self.GidSets)
                self.GidSets.append(GidSet)
            self.CodeBook.append(code)

        for gid in self.TimeSeries:
            self.TimeSeries[gid]['raw_signal'] = append_e(self.TimeSeries[gid]['raw_signal'], 1 if gid in nowgids else 0)
            if not self.TimeSeries[gid]['lock']:
                # LPW note on Sep 24: I think it's unfair to require a
                # continuous live-time before locking the graph ID.
                # We could look at the total live-time instead. 
                # LPW 2019-03-04: Actually making the change.
                if sum([i[0] for i in self.TimeSeries[gid]['raw_signal'] if i[1]]) == self.LearnTime:
                # if self.TimeSeries[gid]['raw_signal'][-1] == [self.LearnTime,True]:
                    if self.printlvl >= 1:
                        print "Locking  gid %s - its timeseries is" % gid, self.TimeSeries[gid]['raw_signal']
                    self.TimeSeries[gid]['lock'] = True
                    self.IsoLocks[self.TimeSeries[gid]['iidx']] = True
                if self.TimeSeries[gid]['raw_signal'][-1] == [self.LearnTime,False]:
                    if self.IsoLocks[self.TimeSeries[gid]['iidx']] == False:
                        if self.printlvl >= 2:
                            print "Deleting graph from isomer index"
                        self.Isomers[self.TimeSeries[gid]['iidx']].Alive = False
                    if self.printlvl >= 2:
                        print "Deleting gid %s - its timeseries is" % gid, self.TimeSeries[gid]['raw_signal']
                    del self.TimeSeries[gid]
        #print "There are %i isomers" % len(self.Isomers)

        self.NumMols.append(NumMols)
        self.IsoLabels.append(ilabels)

    def RectifyTimeSeries(self):
        if self.TestVit:
            print "Testing Viterbi algorithm..."
            t1 = time.time()
            cvit = viterbi(self.CodeBook, metastability=self.metastability, p_correct=self.pcorrectemission)
            print 'c  ', cvit
            t2 = time.time()
            _vit = _viterbi(self.CodeBook, metastability=self.metastability, p_correct=self.pcorrectemission)
            print 'py ', _vit
            t3 = time.time()
            svit = viterbi_skl(self.CodeBook, metastability=self.metastability, p_correct=self.pcorrectemission)
            print 'skl', svit
            t4 = time.time()
            print 'c:', t2-t1, 'py:', t3-t2, 'skl:', t4-t3
            vit = cvit[1]

        if self.printlvl >= 0: print "Rectifying time series..."
        tsnum = 0
        for gid, ts in self.TimeSeries.items():
            RawSignal = ts['raw_signal']
            FilledSignal = FillGaps(ts['raw_signal'],self.LearnTime)
            if self.TestVit:
                ViterbiSignal = encode([gid in self.GidSets[i] for i in vit])
            if len(encode(FilledSignal)) == 1:
                RectifiedSignal = FilledSignal
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    RectifiedSignal = Rectify(ts['raw_signal'],self.metastability,self.pcorrectemission)
            if RectifiedSignal == None:
                self.TimeSeries[gid]['signal'] = FilledSignal
            else:
                self.TimeSeries[gid]['signal'] = RectifiedSignal
            tsnum += 1
            if self.printlvl >= 0: print "\r%i of %i done" % (tsnum, len(self.TimeSeries)),
            if self.printlvl >= 2:
                print "Raw Signal:", RawSignal
                print "Filled    :", FilledSignal
                print "Rectified :", RectifiedSignal
                if self.TestVit:
                    print "Viterbi :", ViterbiSignal
        if self.printlvl >= 0: print
        return

    def Analyze(self):
        #ValidIso = [I for I in self.Isomers if I.Alive]
        MaxLife = [max([longest_lifetime(self.TimeSeries[g]['signal']) for g in self.TimeSeries if self.TimeSeries[g]['iidx'] == u] + [0]) for u in range(len(self.Isomers))]
        BornTimes = []
        Indices = []
        for Life, Gids in zip(MaxLife, [[g for g in self.TimeSeries if self.TimeSeries[g]['iidx'] == u] for u in range(len(self.Isomers))]):
            Born = False
            for gid in Gids:
                b, e = longest_segment(self.TimeSeries[gid]['signal'])
                L = e - b
                if L == Life:
                    BornTimes.append(b)
                    Born = True
                    Indices.append(np.array([int(i) for i in gid.split(':')[0].split(',')]))
                    break
            if not Born:
                BornTimes.append(None)
                Indices.append(None)
        ColorIdx = [8 for i in range(len(self.Isomers))]
        CoolNum = 0
        ColorNum = 0
        self.CoolGraphs = []
        #for i in np.argsort(MaxLife)[::-1]:
        #print MaxLife
        #print BornTimes
        for i in np.argsort(BornTimes):
            I = self.Isomers[i]
            #I = ValidIso[i]
            if self.printlvl >= 1 and MaxLife[i] > 0: print "Isomer %3i %10s : Max Life %i" % (i, "("+I.ef()+")", MaxLife[i]),
            # if I.ef() in self.BoringFormulas:
            if i in self.BoringIsomerIdxs:
                self.Recorded.add(i)
                ColorIdx[i] = 8
                if self.printlvl >= 1 and MaxLife[i] > 0: print "(Boring!)"
            elif MaxLife[i] == 0:
                if self.printlvl >= 0: print "\r",
            elif MaxLife[i] < self.LearnTime:
                ColorIdx[i] = 1
                if self.printlvl >= 1: print "(Short-lived)"
            else:
                if self.printlvl == 0: print "Isomer %3i %10s : Max Life %i" % (i, "("+I.ef()+")", MaxLife[i]),
                # if I.ef() == 'H':
                #     ColorIdx[i] = 8
                #     if self.printlvl >= 0: print "(Interesting; hydrogen atom)"
                # else:
                ColorIdx[i] = self.CoolColors[ColorNum % len(self.CoolColors)]
                if self.printlvl >= 0: print "(Interesting; color %s)" % ColorNames[self.CoolColors[ColorNum % len(self.CoolColors)]],
                ColorNum += 1
                self.Recorded.add(i)
                if self.extract:
                    #MidFrame = int(BornTimes[i] + MaxLife[i]/2)
                    FrameSel = np.arange(BornTimes[i],BornTimes[i]+MaxLife[i])
                    Slice = self.atom_select(Indices[i])[FrameSel]
                    #Slice.comms = ["Extracted from %s: formula %s atoms %s isomer_id %i frame %s charge % .3f sz % .3f sz^2 % .3f" % (I.ef(), commadash(Indices[i]), i, str(frame), sum(self.Charges[frame][Indices[i]]), sum(self.Spins[frame][Indices[i]]), sum([j**2 for j in self.Spins[frame][Indices[i]]])) for frame in FrameSel]
                    Slice.comms = ["Product: formula %s atoms %s frame %s charge %+.3f sz %+.3f sz^2 %.3f" % (I.ef(), commadash(Indices[i]), str(frame), sum(self.Charges[frame][Indices[i]]), sum(self.Spins[frame][Indices[i]]), sum([j**2 for j in self.Spins[frame][Indices[i]]])) for frame in FrameSel]
                    Slice.align_center()
                    Slice.write("extract_%03i.xyz" % CoolNum)
                    self.CoolGraphs.append(I)
                    if self.printlvl >= 0: print " - saving to extract_%03i.xyz" % CoolNum, 
                #self.xyzs[BornTimes[i] + i/2]
                if self.printlvl >= 0: print
                CoolNum += 1
        return ColorIdx

    def PullAidx(self, frame, atoms, ThreshTime, fwd=True):
        Incr = 10
        GoodTime = 0
        while True:
            Valid = False
            Gids = []
            Isos = []
            Answer = set([])
            # The frame is strictly NOT ALLOWED if any atoms belong in the DisallowedFormulas.
            # This is because if an atom belongs to the DisallowedFormulas, the output set can be smaller than the input set.
            if not any([self.Isomers[self.IsoLabels[frame][a]].ef() in self.DisallowedFormulas for a in atoms]):
                for gid,ts in self.TimeSeries.items():
                    if ts['iidx'] not in self.Recorded: continue
                    if element(ts['signal'], frame) and any([i in ts['graph'].L() for i in atoms]):
                        # if element(ts['signal'], frame) != element(ts['raw_signal'], frame): 
                        #     if self.printlvl >= 2:
                        #         print "The filtered signal doesn't match the raw signal in this frame"
                        #     break
                        NewSet = set(ts['graph'].L())
                        if len(Answer.intersection(NewSet)) > 0:
                            if self.printlvl >= 2:
                                print "Double-counting has occurred"
                            break
                        Answer.update(NewSet)
                        Gids.append(gid)
                        Isos.append(ts['iidx'])
                if Answer.issuperset(set(atoms)):
                    Valid = True
                    GoodTime += Incr
                    if GoodTime >= ThreshTime:
                        return sorted(list(Answer)), Gids, frame, sorted(Isos)
                else:
                    if self.printlvl >= 2:
                        print "The accumulated atoms are not a superset of the input atoms - accumulated = %s, input = %s" % (commadash(Answer),commadash(atoms))
            else:
                if self.printlvl >= 2:
                    plist = [self.Isomers[self.IsoLabels[frame][a]].ef() in self.DisallowedFormulas for a in atoms]
                    print "Found some atoms in DisallowedFormulas:", plist
            if not Valid:
                GoodTime = 0
            if fwd:
                if self.printlvl >= 0: print "\rScanning forward, now on frame %i" % frame,
                frame += Incr
            else:
                if self.printlvl >= 0: print "\rScanning backward, now on frame %i" % frame,
                frame -= Incr
            if frame < 0 or frame >= self.Frames:
                return None, None, None, None

    def WriteChargeSpinLabels(self, selection):
        # LPW 2019-03-04 Increasing threshold to 0.25 de-clutters visualization
        Threshold = 0.25
        ChargeLabels = [[] for i in selection]
        #print self.Recorded
        RecSeries = {}
        for gid,ts in self.TimeSeries.items(): # Loop over graph IDs and time series
            if ts['iidx'] in self.Recorded:
                RecSeries[gid] = ts

        for gid,ts in RecSeries.items():
            idx = np.array(ts['graph'].L())
            decoded = decode(ts['signal'])
            for i in selection:
                if decoded[i]:
                    ChgArr = np.array(self.Charges[i][idx])
                    SumChg = sum(ChgArr)
                    #AtomLabel = idx[np.argmax(ChgArr*np.sign(SumChg))]
                    AtomLabel = idx
                    MaxIdx = idx[np.argmax(ChgArr*np.sign(SumChg))]
                    if abs(SumChg) >= Threshold:
                        AtomChgTuple = (AtomLabel, "%+.2f" % SumChg, MaxIdx)
                        ChargeLabels[i].append(AtomChgTuple)

        qout = open('charge.dat','w')
        for Label in ChargeLabels:
            for AtomChgTuple in Label:
                #print >> qout, "{%s} %s" % (' '.join(["%i" % i for i in AtomChgTuple[0]]), AtomChgTuple[1])
                #print >> qout, "%i %s" % (AtomChgTuple[0], AtomChgTuple[1]),
                # This generates a string like 100 101 102! 103, 0.25; which indicates the atoms in a molecule, the net charge on that molecule, and the atom
                # with the greatest charge marked by an exclamation point
                print >> qout, "%s, %s;" % (' '.join(["%i%s" % (i, "!" if i == AtomChgTuple[2] else "") for i in AtomChgTuple[0]]), AtomChgTuple[1]),
            print >> qout

        Threshold = 0.25
        sout = open('spin.dat','w')
        for i in selection:
            for a in range(self.na):
                if abs(self.Spins[i][a]) >= Threshold:
                    print >> sout, "%i %+.2f" % (a, self.Spins[i][a]),
            print >> sout
        qout.close()

    def GetNeutralizing(self, atoms, framesel):
        """
        Given a list of atoms and a time series, find another list of atoms corresponding to:
        1) complete molecules that exist throughout the entire time series
        2) is the closest to the original set of atoms by some measure 
           (we'll use the maximum value of the closest contact time series)
        3) neutralizes the overall system

        Parameters
        ----------
        atoms : list
            List of atoms 
        framesel : list
            List of frames

        Returns
        -------
        counter_atoms : list
            List of atoms that neutralizes the input list
        """
        # Get the current charge on the atom selection
        SelChg = np.mean(np.sum(self.Charges[framesel][:, atoms], axis=1))
        SelSpn = np.mean(np.sum(self.Spins[framesel][:, atoms], axis=1))
        if self.printlvl >= 1: print "Attempting to neutralize atoms %s (charge %+.3f)" % (commadash(framesel), SelChg)
        Selxyz = np.array(self.atom_select(atoms)[framesel].xyzs)
        # Candidate molecules to serve as counterions
        Candidate_aidx = []
        Candidate_gids = []
        Candidate_mindist = []
        Candidate_maxdist = []
        Candidate_ichg = []
        # Loop over all molecules in the time series
        for gid, ts in self.TimeSeries.items():
            # Check to make sure that this molecule exists for ALL frames in the frame selection
            # and does not overlap with ANY atoms in our atom selection.
            if (exists_at_time(ts['signal'], framesel[0]) and 
                exists_at_time(ts['signal'], framesel[-1]) and
                len(set(ts['graph'].L()).intersection(set(atoms))) == 0 and
                list(set(decode(ts['signal'])[framesel[0]:framesel[-1]])) == [1]):
                # List of atoms in this molecule
                aidx = ts['graph'].L()
                MolChg = np.mean(np.sum(self.Charges[framesel][:, aidx], axis=1))
                # This signifies that the molecule has nonzero charge with opposite sign
                if np.abs(MolChg) > 0.25 and MolChg * SelChg < 0:
                    Molxyz = np.array(self.atom_select(aidx)[framesel].xyzs)
                    # Get the squared distance matrix for every charged molecule with opposite sign
                    contacts2 = np.zeros((Selxyz.shape[1], Molxyz.shape[1], len(framesel)))
                    for a in range(Selxyz.shape[1]):
                        for b in range(Molxyz.shape[1]):
                            contacts2[a, b, :] = np.sum((Selxyz[:, a, :] - Molxyz[:, b, :])**2, axis=1)
                    # Build a closest contact time series
                    cct = []
                    for t in range(contacts2.shape[2]):
                        cct.append(np.sqrt(np.min(contacts2[:, :, t])))
                    cct = np.array(cct)
                    # Record the closest that the molecule got, and how far it drifted away
                    Candidate_aidx.append(aidx)
                    Candidate_gids.append(gid)
                    Candidate_mindist.append(min(cct))
                    Candidate_maxdist.append(max(cct))
                    Candidate_ichg.append(MolChg)
        counter_atoms = []
        counter_gids = []
        added_sets = [[]]
        added_chgs = [SelChg]
        added_spns = [SelSpn]
        added_isos = [[]]
        # Now loop over all candidates in order of increasing closest contact distance
        for c in np.argsort(Candidate_maxdist):
            aidx = Candidate_aidx[c]
            if self.printlvl >= 1: print "Trying to neutralize with atoms %s (charge %+.3f distance %.3f - %.3f)" % (commadash(aidx), Candidate_ichg[c], Candidate_mindist[c], Candidate_maxdist[c])
            counter_atoms = sorted(list(set(counter_atoms + aidx)))
            counter_gids = sorted(list(set(counter_gids + [Candidate_gids[c]])))
            NewSelChg = np.mean(np.sum(self.Charges[framesel][:, atoms+counter_atoms], axis=1))
            NewSelSpn = np.mean(np.sum(self.Spins[framesel][:, atoms+counter_atoms], axis=1))
            # Check for charge and spin consistency.
            nproton = sum([Elements.index(i) for i in [self.elem[j] for j in (atoms+counter_atoms)]])
            nelectron = int(nproton + round(NewSelChg))
            spn = int(round(NewSelSpn))
            # The number of electrons should be odd iff the spin is odd.
            if ((nelectron-spn)/2)*2 != (nelectron-spn):
                if self.printlvl >= 1: print "Charge and spin \x1b[91mnot consistent\x1b[0m (charge %+.3f -> %+.3f) ; added atoms %s" % (SelChg, NewSelChg, commadash(counter_atoms))
            elif int(round(NewSelChg)) == 0: 
                if self.printlvl >= 1: print "Successfully \x1b[92mneutralized\x1b[0m the reaction (charge %+.3f -> %+.3f) ; added atoms %s" % (SelChg, NewSelChg, commadash(counter_atoms))
                added_sets.append(counter_atoms[:])
                added_isos.append([int(i.split(':')[1]) for i in counter_gids])
                added_chgs.append(NewSelChg)
                added_spns.append(NewSelSpn)
            elif int(round(NewSelChg)) * SelChg < 0:
                if self.printlvl >= 1: print "\x1b[91mOvershot\x1b[0m the reaction (charge %+.3f -> %+.3f) ; added atoms %s" % (SelChg, NewSelChg, commadash(counter_atoms))
                added_sets.append(counter_atoms[:])
                added_isos.append([int(i.split(':')[1]) for i in counter_gids])
                added_chgs.append(NewSelChg)
                added_spns.append(NewSelSpn)
                break
            else:
                if self.printlvl >= 1: print "Reaction is \x1b[93mnot neutralized\x1b[0m yet (charge %+.3f -> %+.3f) ; added atoms %s" % (SelChg, NewSelChg, commadash(counter_atoms))
        selct = np.argmin(np.abs(np.array(added_chgs)))
        print "Result of neutralization: charge \x1b[94m%+.3f\x1b[0m -> \x1b[92m%+.3f\x1b[0m, added atoms %s" % (SelChg, added_chgs[selct], commadash(added_sets[selct]))
        print added_isos[selct]
        return added_sets[selct], added_isos[selct]

    def GetReactions(self):
        # This piece of code first pulls up our recognized reaction product trajectory segments.
        # It scans forward / backward past the ends of the segment in search of a chemical reaction.
        # By scanning past the ends of the segment, we look at all of the molecules that the atoms have evolved into.
        # The new molecules' atom IDs are all stored.  If the topology (denoted by gids) has changed 
        # but the atom IDs have stayed the same, then an isomerization has taken place and the trajectory is stored.
        # If the atom IDs have expanded, it means that more atoms are involved in the chemical reaction,
        # and an attempt is made to incorporate them into a contiguous chemical reaction.
        if not self.saverxn:
            return
        if self.printlvl >= 0: print "Attempting to extract chemical reactions:"
        RxnNum = 0
        RxnList = [] # A list of 2-tuples consisting of [set(atoms), set(frames)]
        RxRepeats = defaultdict(int) # A dictionary that keeps track of how many instances of a particular reaction has occurred
        # I'm going to sort the chemical reactions in order of their first frame.
        Slices = []        # A list of lists of Molecule objects to be saved to disk
        Firsts = []        # A list of lists of first frames
        Lasts  = []        # A list of lists of last frames
        SliceIndices = []  # A list of atom indices corresponding to each saved trajectory
        SliceFrames = []   # A list of frames corresponding to each saved trajectory
        BufferTime = 0
        PadTime = self.PadTime
        for gid, ts in self.TimeSeries.items():
            iidx = ts['iidx']
            I = self.Isomers[iidx]
            S = segments(FillGaps(ts['signal'],self.LearnTime))
            if self.printlvl >= 1: print "Original and Condensed Segments:", segments(ts['signal']), S
            aidx = ts['graph'].L()
            if iidx not in self.BoringIsomerIdxs and len(S) > 0:
                if self.printlvl >= 1: print "Molecular formula:", I.ef(), "atoms:", commadash(aidx), "frames:", S
                for s in S:
                    if (s[1]-s[0]) > self.LearnTime:
                        Begin  = s[0] + PadTime
                        End    = s[1] - PadTime
                        Before = max(0, s[0] - + PadTime)
                        After  = min(self.Frames-1, s[1] + PadTime)
                        if Before < Begin:
                            #if self.printlvl >= 1: print "Looking backward to frame %i," % Before,
                            Reactant_Atoms = aidx
                            New_Reactants = None
                            EndPoint = False
                            Reacted = True
                            minf = Before
                            maxf = Begin
                            iso0 = [ts['iidx']]
                            gid0 = [gid]
                            while True:
                                fidx = Begin if EndPoint else Before
                                here = Before if EndPoint else Begin
                                New_Reactants, gid1, newf, iso1 = self.PullAidx(fidx,Reactant_Atoms,PadTime,fwd=EndPoint)
                                if New_Reactants == None:
                                    if self.printlvl >= 1: print " - scanned past start of trajectory!"
                                    Reacted = False
                                    break
                                if newf < minf:
                                    minf = newf
                                elif newf > maxf:
                                    maxf = newf
                                if self.printlvl >= 1: print "Whole molecules in frame %i have atoms %s ; pulled molecules from frame %i and got %s" % (here, commadash(Reactant_Atoms), newf, commadash(New_Reactants)),
                                if len(gid1) == 1 and gid1[0] == gid:
                                    if self.printlvl >= 1: print " - no reaction took place!"
                                    Reacted = False
                                    break
                                if Reactant_Atoms == New_Reactants : 
                                    if self.printlvl >= 1: print " - success!"
                                    break
                                else: 
                                    if self.printlvl >= 1: print " - expanding the system"
                                    Reactant_Atoms = New_Reactants[:]
                                    iso0 = iso1[:]
                                    gid0 = gid1[:]
                                EndPoint = not EndPoint
                                here = newf
                            if Reacted:
                                Before = max(0, minf - BufferTime) # Insert a small number of frames at the start where we have the reactants.
                                Begin = min(self.Frames, maxf + BufferTime)
                                FrameSel = np.arange(Before, Begin)
                                # Check for repeats.
                                Repeat = False
                                Overwrite = -1
                                Annotate = -1
                                # At this stage we may have some 'spectator molecules'.  Time to get rid of them.
                                for spec in set(gid0).intersection(set(gid1)):
                                    II = self.TimeSeries[spec]['graph']
                                    if self.printlvl >= 1: print "Removing spectator molecule, graph id %s (%s)" % (spec, II.ef())
                                    for a in II.L():
                                        Reactant_Atoms.remove(a)
                                    iso0.remove(self.TimeSeries[spec]['iidx'])
                                    iso1.remove(self.TimeSeries[spec]['iidx'])
                                    gid0.remove(spec)
                                    gid1.remove(spec)
                                if len(Reactant_Atoms) == 0: 
                                    if self.printlvl >= 1: print "There are no atoms left!"
                                    break
                                for Rnum, Rxn in enumerate(RxnList):
                                    if len(Rxn[1].intersection(set(FrameSel))) >= 0.5*len(Rxn[1]):
                                        if set(Reactant_Atoms).issubset(Rxn[0]):
                                            if self.printlvl >= 1: print "This reaction is a repeat/subset of one we already have!"
                                            Repeat = True
                                            break
                                if not Repeat:
                                    for Rnum, Rxn in enumerate(RxnList):
                                        if (Rxn[2] == iso0 and Rxn[3] == iso1) or (Rxn[2] == iso1 and Rxn[3] == iso0):
                                            if self.printlvl >= 1: print "This reaction is \x1b[93manother instance\x1b[0m of reaction number %i!" % Rnum
                                            Repeat = True
                                            Annotate = Rnum
                                            break
                                if not Repeat:
                                    for Rnum, Rxn in enumerate(RxnList):
                                        if len(Rxn[1].intersection(set(FrameSel))) >= 0.5*len(Rxn[1]):
                                            if set(Reactant_Atoms).issuperset(Rxn[0]):
                                                if self.printlvl >= 1: print "This reaction is a \x1b[94msuperset\x1b[0m of one we already have - overwriting!"
                                                Repeat = True
                                                Overwrite = Rnum
                                                break
                                save = False
                                if not Repeat:
                                    save = True
                                    RxnList.append((set(Reactant_Atoms),set(FrameSel),iso0,iso1))
                                    RxnNum += 1
                                elif Overwrite >= 0:
                                    save = True
                                    RxnList[Overwrite] = (set(Reactant_Atoms),set(FrameSel),iso0,iso1)
                                elif Annotate >= 0:
                                    save = True
                                    RxRepeats[Annotate] += 1
                                if save:
                                    if self.neutralize:
                                        Neutral_Atoms, Neutral_Isos = self.GetNeutralizing(Reactant_Atoms, FrameSel)
                                        Reactant_Atoms += Neutral_Atoms
                                        iso0 += Neutral_Isos
                                        iso1 += Neutral_Isos
                                    Reactant_Atoms = sorted(list(set(Reactant_Atoms)))
                                    Slice = self.atom_select(Reactant_Atoms)[FrameSel]
                                    rx0 = '+'.join([("%i" % iso0.count(i) if iso0.count(i) > 1 else "") + self.Isomers[i].ef() for i in sorted(set(iso0))])
                                    rx1 = '+'.join([("%i" % iso1.count(i) if iso1.count(i) > 1 else "") + self.Isomers[i].ef() for i in sorted(set(iso1))])
                                    evector="%s -> %s" % (rx0 if EndPoint else rx1, rx1 if EndPoint else rx0)
                                    tx0 = str([commadash(sorted([Reactant_Atoms.index(i) for i in self.TimeSeries[g]['graph'].L()])) for g in gid0]).replace(" ","")
                                    tx1 = str([commadash(sorted([Reactant_Atoms.index(i) for i in self.TimeSeries[g]['graph'].L()])) for g in gid1]).replace(" ","")
                                    tvector="%s -> %s" % (tx0 if EndPoint else tx1, tx1 if EndPoint else tx0)
                                    Slice.comms = ["Reaction: formula %s atoms %s frame %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, str(frame), sum(self.Charges[frame][Reactant_Atoms]),
                                                                                                                               sum(self.Spins[frame][Reactant_Atoms]), sum([j**2 for j in self.Spins[frame][Reactant_Atoms]])) for frame in FrameSel]
                                    Slice.align_center()
                                    if not Repeat:
                                        Slices.append([Slice])
                                        Firsts.append([FrameSel[0]])
                                        Lasts.append([FrameSel[-1]])
                                        SliceIndices.append([Reactant_Atoms])
                                        SliceFrames.append([FrameSel])
                                    elif Overwrite >= 0:
                                        Slices[Overwrite] = [Slice]
                                        Firsts[Overwrite] = [FrameSel[0]]
                                        Lasts[Overwrite] = [FrameSel[-1]]
                                        SliceIndices[Overwrite] = [Reactant_Atoms]
                                        SliceFrames[Overwrite] = [FrameSel]
                                    elif Annotate >= 0:
                                        Slices[Annotate].append(Slice)
                                        Firsts[Annotate].append(FrameSel[0])
                                        Lasts[Annotate].append(FrameSel[-1])
                                        SliceIndices[Annotate].append(Reactant_Atoms)
                                        SliceFrames[Annotate].append(FrameSel)
                                    if self.printlvl >= 0: print "\rReaction found: formula %s atoms %s frames %i through %i" % (evector, commadash(Reactant_Atoms), FrameSel[0], FrameSel[-1])
                                    
                        if After > End:
                            #if self.printlvl >= 1: print "Looking forward to frame %i," % After,
                            Reactant_Atoms = aidx
                            New_Reactants = None
                            EndPoint = False
                            Reacted = True
                            minf = End
                            maxf = After
                            iso0 = [ts['iidx']]
                            gid0 = [gid]
                            while True:
                                fidx = End if EndPoint else After
                                here = After if EndPoint else End
                                New_Reactants, gid1, newf, iso1 = self.PullAidx(fidx,Reactant_Atoms,PadTime,fwd=not EndPoint)
                                if New_Reactants == None:
                                    if self.printlvl >= 1: print " - scanned past end of trajectory!"
                                    Reacted = False
                                    break
                                if newf > maxf:
                                    maxf = newf
                                elif newf < minf:
                                    minf = newf
                                if self.printlvl >= 1: print "Whole molecules in frame %i have atoms %s ; pulled molecules from frame %i and got %s" % (here, commadash(Reactant_Atoms), newf, commadash(New_Reactants)),
                                if len(gid1) == 1 and gid1[0] == gid:
                                    if self.printlvl >= 1: print " - no reaction took place!"
                                    Reacted = False
                                    break
                                if Reactant_Atoms == New_Reactants : 
                                    if self.printlvl >= 1: print " - success!"
                                    break
                                else: 
                                    if self.printlvl >= 1: print " - expanding the system"
                                    Reactant_Atoms = New_Reactants[:]
                                    iso0 = iso1[:]
                                    gid0 = gid1[:]
                                EndPoint = not EndPoint
                                here = newf
                            if Reacted:
                                After = min(self.Frames, maxf + BufferTime)
                                End = max(0, minf - BufferTime)
                                FrameSel = np.arange(End, After)
                                # Check for repeats.
                                Repeat = False
                                Overwrite = -1
                                Annotate = -1
                                # At this stage we may have some 'spectator molecules'.  Time to get rid of them.
                                for spec in set(gid0).intersection(set(gid1)):
                                    II = self.TimeSeries[spec]['graph']
                                    if self.printlvl >= 1: print "Removing spectator molecule, graph id %s (%s)" % (spec, II.ef())
                                    for a in II.L():
                                        Reactant_Atoms.remove(a)
                                    iso0.remove(self.TimeSeries[spec]['iidx'])
                                    iso1.remove(self.TimeSeries[spec]['iidx'])
                                    gid0.remove(spec)
                                    gid1.remove(spec)
                                if len(Reactant_Atoms) == 0: 
                                    if self.printlvl >= 1: print "There are no atoms left!"
                                    break
                                for Rnum, Rxn in enumerate(RxnList):
                                    if len(Rxn[1].intersection(set(FrameSel))) >= 0.5*len(Rxn[1]):
                                        if set(Reactant_Atoms).issubset(Rxn[0]):
                                            if self.printlvl >= 1: print "This reaction is a repeat/subset of one we already have!"
                                            Repeat = True
                                            break
                                if not Repeat:
                                    for Rnum, Rxn in enumerate(RxnList):
                                        if (Rxn[2] == iso0 and Rxn[3] == iso1) or (Rxn[2] == iso1 and Rxn[3] == iso0):
                                            if self.printlvl >= 1: print "This reaction is \x1b[93manother instance\x1b[0m of reaction number %i!" % Rnum
                                            Repeat = True
                                            Annotate = Rnum
                                            break
                                if not Repeat:
                                    for Rnum, Rxn in enumerate(RxnList):
                                        if len(Rxn[1].intersection(set(FrameSel))) >= 0.5*len(Rxn[1]):
                                            if set(Reactant_Atoms).issuperset(Rxn[0]):
                                                if self.printlvl >= 1: print "This reaction is a \x1b[94msuperset\x1b[0m of one we already have - overwriting!"
                                                Repeat = True
                                                Overwrite = Rnum
                                                break
                                save = False
                                if not Repeat:
                                    save = True
                                    RxnList.append((set(Reactant_Atoms),set(FrameSel),iso0,iso1))
                                    RxnNum += 1
                                elif Overwrite >= 0:
                                    save = True
                                    RxnList[Overwrite] = (set(Reactant_Atoms),set(FrameSel),iso0,iso1)
                                elif Annotate >= 0:
                                    save = True
                                    RxRepeats[Annotate] += 1
                                if save:
                                    if self.neutralize:
                                        Neutral_Atoms, Neutral_Isos = self.GetNeutralizing(Reactant_Atoms, FrameSel)
                                        Reactant_Atoms += Neutral_Atoms
                                        iso0 += Neutral_Isos
                                        iso1 += Neutral_Isos
                                    Reactant_Atoms = sorted(list(set(Reactant_Atoms)))
                                    Slice = self.atom_select(Reactant_Atoms)[FrameSel]
                                    rx0 = '+'.join([("%i" % iso0.count(i) if iso0.count(i) > 1 else "") + self.Isomers[i].ef() for i in sorted(set(iso0))])
                                    rx1 = '+'.join([("%i" % iso1.count(i) if iso1.count(i) > 1 else "") + self.Isomers[i].ef() for i in sorted(set(iso1))])
                                    evector="%s -> %s" % (rx1 if EndPoint else rx0, rx0 if EndPoint else rx1)
                                    tx0 = str([commadash(sorted([Reactant_Atoms.index(i) for i in self.TimeSeries[g]['graph'].L()])) for g in gid0]).replace(" ","")
                                    tx1 = str([commadash(sorted([Reactant_Atoms.index(i) for i in self.TimeSeries[g]['graph'].L()])) for g in gid1]).replace(" ","")
                                    tvector="%s -> %s" % (tx1 if EndPoint else tx0, tx0 if EndPoint else tx1)
                                    Slice.comms = ["Reaction: formula %s atoms %s frame %s charge %+.3f sz %+.3f sz^2 %.3f" % (evector, tvector, str(frame), sum(self.Charges[frame][Reactant_Atoms]),
                                                                                                                               sum(self.Spins[frame][Reactant_Atoms]), sum([j**2 for j in self.Spins[frame][Reactant_Atoms]])) for frame in FrameSel]
                                    Slice.align_center()
                                    if not Repeat:
                                        Slices.append([Slice])
                                        Firsts.append([FrameSel[0]])
                                        Lasts.append([FrameSel[-1]])
                                        SliceIndices.append([Reactant_Atoms])
                                        SliceFrames.append([FrameSel])
                                    elif Overwrite >= 0:
                                        Slices[Overwrite] = [Slice]
                                        Firsts[Overwrite] = [FrameSel[0]]
                                        Lasts[Overwrite] = [FrameSel[-1]]
                                        SliceIndices[Overwrite] = [Reactant_Atoms]
                                        SliceFrames[Overwrite] = [FrameSel]
                                    elif Annotate >= 0:
                                        Slices[Annotate].append(Slice)
                                        Firsts[Annotate].append(FrameSel[0])
                                        Lasts[Annotate].append(FrameSel[-1])
                                        SliceIndices[Annotate].append(Reactant_Atoms)
                                        SliceFrames[Annotate].append(FrameSel)
                                    if self.printlvl >= 0: print "\rReaction found: formula %s atoms %s frames %i through %i" % (evector, commadash(Reactant_Atoms), FrameSel[0], FrameSel[-1])
        print
        RxnSrl = 0                                                    # Serial number of the reaction for writing to disk
        haverxn = {}
        # Look for existing reaction.xyz so we can preserve existing reactions from previous runs.
        for fnm in os.listdir('.'):
            if fnm.startswith('reaction_') and fnm.endswith('.xyz'):
                srl = int(os.path.splitext(fnm)[0].split("_")[1])
                maxinst = 0
                for fnm1 in os.listdir('.'):
                    if fnm1.startswith('reaction_%03i' % srl) and fnm1.endswith('.xyz'):
                        if len(os.path.splitext(fnm1)[0].split("_")) > 2:
                            maxinst = max(maxinst, int(os.path.splitext(fnm1)[0].split("_")[2]))
                if srl >= RxnSrl:
                    RxnSrl = srl+1
                rcomms = [l for l in open(fnm).readlines() if l.startswith('Reaction:')]
                rframes = set([int(l.split("frame")[1].split()[0]) for l in rcomms])
                L = ast.literal_eval(rcomms[0].split('->')[1].split()[-1])
                aset = set(list(itertools.chain(*[uncommadash(i) for i in L])))
                haverxn[fnm] = [aset, rframes, srl, maxinst]

        def rxn_lookup(rxnnum, inst):
            # Read frames and atoms from the reaction about to be saved.
            rxn = Slices[rxnnum][inst]
            rframes = set([int(l.split("frame")[1].split()[0]) for l in rxn.comms])
            L = ast.literal_eval(rxn.comms[0].split('->')[1].split()[-1])
            aset = set(list(itertools.chain(*[uncommadash(i) for i in L])))
            for rxn0 in haverxn:
                aset0, rframes0, srl0, maxinst = haverxn[rxn0]
                overlap = (float(len(rframes.intersection(rframes0))) / max(len(rframes), len(rframes0)))
                if aset == aset0 and overlap > 0.9:
                    if self.printlvl >= 0: print "Reaction in frames %i -> %i overlaps with %s (%.1f%% frames)" % (Firsts[rxnnum][inst], Lasts[rxnnum][inst], rxn0, 100*overlap)
                    return srl0, maxinst, rxn0
            return -1, -1, None
                
        for RxnNum in np.argsort(np.array([min(i) for i in Firsts])): # Reactions sorted by the first frame of occurrence
            InstSrl = 0                                               # Instance number of the reaction for writing to disk
            RxnSrl_ = RxnSrl                                          # The temporary reaction serial number (will be replaced if reaction exists on disk)
            for Inst in np.argsort(Firsts[RxnNum]):                   # Instances of a given reaction, again sorted by first frame of occurrence
                HaveRxn, HaveInst, HaveFnm = rxn_lookup(RxnNum, Inst) # Whether this reaction already exists in the list of reaction_123.xyz, and the maximum instance of this reaction.
                if HaveRxn > -1:
                    RxnSrl_ = HaveRxn
                if HaveInst >= InstSrl:                               # Any more instances of this reaction will need to be written with reaction_123_003.xyz (003 incremented from 002 for example)
                    InstSrl = HaveInst+1
                if InstSrl == 0:
                    outfnm = "reaction_%03i.xyz" % RxnSrl_
                else:
                    outfnm = "reaction_%03i_%i.xyz" % (RxnSrl_, InstSrl)
                if HaveRxn == -1:
                    # LPW: All of the repeated reaction.xyz files are getting annoying!
                    if InstSrl < 10:
                        if self.printlvl >= 0: print "\x1b[1;92mSaving\x1b[0m frames %i -> %i to %s" % (Firsts[RxnNum][Inst], Lasts[RxnNum][Inst], outfnm)
                        Slices[RxnNum][Inst].center()
                        Slices[RxnNum][Inst].write(outfnm)
                    elif self.printlvl >= 0: print "\x1b[1;93mNot Saving\x1b[0m frames %i -> %i (instance %i)" % (Firsts[RxnNum][Inst], Lasts[RxnNum][Inst], InstSrl)
                    InstSrl += 1
                if HaveFnm != None:
                    outfnm = HaveFnm
                if os.path.exists(outfnm) and not os.path.exists(outfnm.replace('.xyz','.pop')):
                    # Build a molecule object containing the corresponding charges and spin
                    SlicePop = deepcopy(Slices[RxnNum][Inst])
                    ThisIdx = SliceIndices[RxnNum][Inst]
                    ThisFrm = SliceFrames[RxnNum][Inst]
                    # Grab charges from the global trajectory
                    ThisChg = [self.Charges[iframe][ThisIdx] for iframe in ThisFrm]
                    ThisSpn = [self.Spins[iframe][ThisIdx] for iframe in ThisFrm]
                    # Assign charges to positions (for convenience)
                    for iframe in range(len(SlicePop)):
                        for iatom in range(SlicePop.na):
                            SlicePop.xyzs[iframe][iatom][0] = ThisChg[iframe][iatom]
                            SlicePop.xyzs[iframe][iatom][1] = ThisSpn[iframe][iatom]
                            SlicePop.xyzs[iframe][iatom][2] = 0.0
                    # Write charges and spins to x and y coordinates in a duplicate .xyz file
                    SlicePop.write(outfnm.replace('.xyz','.pop'),ftype='xyz')
            if RxnSrl_ == RxnSrl: RxnSrl += 1

        if self.printlvl >= 0: print
        return

    def PrintColors(self):
        ColorNow = [-1 for i in range(self.na)]
        # Print header stuff
        header = """axes location Off
display rendermode GLSL
display backgroundgradient on
display projection Orthographic
display depthcue off
display nearclip set 0.010000
material change opacity Ghost 0.000000
#material change opacity Transparent 0.500000

"""
        print >> self.moviefile, header
        for a in range(self.na):
            if a > 0:
                print >> self.moviefile, "mol addrep 0"
            print >> self.moviefile, "mol modselect %i 0 index %i" % (a, a)
            print >> self.moviefile, "mol modstyle %i 0 VDW 0.50 27.0" % (a)
            print >> self.moviefile, "mol modmaterial %i 0 Transparent" % (a)

        extra = """
mol addrep 0
mol modselect %i 0 not name \\"H.*\\"
mol modstyle %i 0 DynamicBonds 1.800000 0.050000 21.000000
mol addrep 0
mol modselect %i 0 all
mol modstyle %i 0 DynamicBonds 1.100000 0.050000 21.000000
mol addrep 0
mol modstyle %i 0 VDW 0.150000 27.000000
""" % (self.na, self.na, self.na + 1, self.na + 1, self.na + 2)
        
        print >> self.moviefile, extra
        renderf = 0
        for f in range(0, self.Frames, self.stride):
            ColorByAtom = [self.ColorIdx[j] for j in self.IsoLabels[renderf]]
            print >> self.moviefile, "animate goto %i" % f
            print >> self.moviefile, "display resetview"
            print >> self.moviefile, "display height 4.0"
            print >> self.moviefile, "rotate y by %.1f" % ((0.1 * renderf) % 360)
            for an, color in enumerate(ColorByAtom):
                print >> self.colortab, color,
                if ColorNow[an] != color:
                    ColorNow[an] = color
                    print >> self.moviefile, "mol modcolor %i 0 ColorID %i" % (an, color)
                    if color == 8:
                        print >> self.moviefile, "mol modmaterial %i 0 Ghost" % an
                    else:
                        print >> self.moviefile, "mol modmaterial %i 0 Transparent" % an
            print >> self.colortab
            if self.Render:
                print >> self.moviefile, "render snapshot frame%04i.tga" % renderf
            renderf += 1
        
    def MakeGraphFromXYZ(self, sn, window=10):
        G = MyG()
        bonds = [[] for i in range(self.na)]
        lengths = []
        for i, a in enumerate(self.elem):
            G.add_node(i)
            if parse_version(nx.__version__) >= parse_version('2.0'):
                if 'atomname' in self.Data:
                    nx.set_node_attributes(G,{i:self.atomname[i]}, name='n')
                nx.set_node_attributes(G,{i:a}, name='e')
                nx.set_node_attributes(G,{i:self.xyzs[sn][i]}, name='x')
            else:
                if 'atomname' in self.Data:
                    nx.set_node_attributes(G,'n',{i:self.atomname[i]})
                nx.set_node_attributes(G,'e',{i:a})
                nx.set_node_attributes(G,'x',{i:self.xyzs[sn][i]})
        bond_bool = self.dxij[0] < self.BondThresh
        for i, a in enumerate(bond_bool):
            if not a: continue
            (ii, jj) = self.AtomIterator[i]
            bonds[ii].append(jj)
            bonds[jj].append(ii)
            G.add_edge(ii, jj)
            lengths.append(self.dxij[0][i])
        self.BondLists.append(bondlist_tcl(bonds))
        self.BondLengths.append(lengths)
        return G

    def Output(self):
        # Print final data to file.
        self.moviefile = open('make-movie.tcl','w')
        self.colortab = open('color.dat','w')
        self.PrintColors()
        self.GetReactions()
        with open('bonds.dat','w') as bondtab: bondtab.write('\n'.join(self.BondLists)+'\n')
        if self.bWrite:
            if os.path.exists(self.fout):
                user_input = raw_input("%s exists, enter a new file name or hit Enter to overwrite >>> " % self.fout)
                if len(user_input.strip()) > 0:
                    self.fout = user_input.strip()
            print "Writing", self.fout
            try:
                self.write(self.fout, selection=range(0,self.Frames,self.stride))
            except:
                print "File write failed, check what you typed in."
                self.fout = self.fnm
        self.WriteChargeSpinLabels(range(0,self.Frames,self.stride))

