#!/usr/bin/env python
from __future__ import print_function
import os, sys, re
import networkx as nx
import numpy as np
import copy
from collections import namedtuple, OrderedDict, defaultdict, Counter
from .chemistry import Elements, Radii
from copy import deepcopy
from .molecule import AtomContact, BuildLatticeFromLengthsAngles, Molecule, format_xyz_coord
#from . import contact
import itertools
import time
from pkg_resources import parse_version
from scipy.signal import butter, freqz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.switch_backend('agg')

## Names of colors from VMD
ColorNames = ["blue", "red", "gray", "orange", "yellow", 
              "tan", "silver", "green", "none", "pink", 
              "cyan", "purple", "lime", "mauve", "ochre",
              "iceblue", "black", "yellow2", "yellow3", "green2",
              "green3", "cyan2", "cyan3", "blue2", "blue3", 
              "violet", "violet2", "magenta", "magenta2", "red2", 
              "red3", "orange2", "orange3"]

def wildmatch(fmatch, formulas):
    for fpat in formulas:
        if '?' in fpat and (re.sub('\?', '', fpat) == re.sub('[0-9]', '', fmatch)):
            return True
        elif fpat == fmatch: return True
    return False

def subscripts(string):
    unicode_integers = {'0': 8320, '1': 8321, '2': 8322,
                        '3': 8323, '4': 8324, '5': 8325,
                        '6': 8326, '7': 8327, '8': 8328,
                        '9': 8329}
    ustr = str(string)
    for i in unicode_integers:
        ustr = ustr.replace(i, chr(unicode_integers[i]))
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
    # Formats a list like [26, 27, 28, 29, 30, 87, 88, 89, 90, 99, 135, 136, 137, 138]
    # into a string as: '27-31,88-91,100,136-139'
    # Note: The string uses one-based indices whereas the list uses zero-based indices
    L = sorted(l)
    if len(L) == 0:
        return "(empty)"
    L.append(L[-1]+1)
    LL = [i in L for i in range(L[-1])]
    return ','.join('%i-%i' % (i[0]+1,i[1]) if (i[1]-1 > i[0]) else '%i' % (i[0]+1) for i in segments(encode(LL)))

def uncommadash(s):
    # Takes a string like '27-31,88-91,100,136-139'
    # and turns it into a list like [26, 27, 28, 29, 30, 87, 88, 89, 90, 99, 135, 136, 137, 138]
    # Note: The string uses one-based indices whereas the list uses zero-based indices
    L = []
    try:
        for w in s.split(','):
            ws = w.split('-')
            a = int(ws[0])-1
            if len(ws) == 1:
                b = int(ws[0])
            elif len(ws) == 2:
                b = int(ws[1])
            else:
                print("Dash-separated list cannot exceed length 2")
                raise
            if a < 0 or b <= 0 or b <= a:
                if a < 0 or b <= 0:
                    print("Items in list cannot be zero or negative: %d %d" % (a, b))
                else:
                    print("Second number cannot be smaller than first: %d %d" % (a, b))
                raise
            newL = list(range(a,b))
            if any([i in L for i in newL]):
                print("Duplicate entries found in list")
                raise
            L += newL
        if sorted(L) != L:
            print("List is out of order")
            raise
    except:
        print("Invalid string for converting to list of numbers: %s" % s)
        raise RuntimeError
    return L

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
    def __eq__(self, other):
        # This defines whether two MyG objects are "equal" to one another.
        return nx.is_isomorphic(self,other,node_match=nodematch)
    def __hash__(self):
        ''' The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. '''
        return 1
    def L(self):
        ''' Return a list of the sorted atom numbers in this graph. '''
        return sorted(self.nodes())
    def AStr(self):
        ''' Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . '''
        return commadash(self.L())
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
    def make_whole(self, a, b, c):
        ''' Make the molecule whole in a rectilinear box '''
        # x = nx.get_node_attributes(self,'x')
        # print(x)
        x = self.x().copy()
        x0 = x[0]
        dx = x - x0
        # Apply the minimum image convention 
        x[:,0] += a * (dx[:,0] < -a/2)
        x[:,0] -= a * (dx[:,0] >  a/2)
        x[:,1] += b * (dx[:,1] < -b/2)
        x[:,1] -= b * (dx[:,1] >  b/2)
        x[:,2] += c * (dx[:,2] < -c/2)
        x[:,2] -= c * (dx[:,2] >  c/2)
        xm = np.mean(x, axis=0)
        if xm[0] > a: x[:,0] -= a
        elif xm[0] < 0: x[:,0] += a
        if xm[1] > b: x[:,1] -= b
        elif xm[1] < 0: x[:,1] += b
        if xm[2] > c: x[:,2] -= c
        elif xm[2] < 0: x[:,2] += c
        xdict = dict([(i, xi) for i, xi in zip(self.L(), x)])
        if parse_version(nx.__version__) >= parse_version('2.0'):
            nx.set_node_attributes(self, xdict, 'x')
        else:
            nx.set_node_attributes(self, 'x', xdict)

def low_pass_smoothing(all_raw_time_series, sigma, dt_fs, reflect=True):
    """ 
    Low-pass smoothing function for bond order or interatomic distance time series.
    
    Parameters
    ----------
    all_raw_time_series : np.ndarray
        2-D array consisting of time series to be filtered. First dimension is the number
        of time series to be filtered at once; second dimension is the length of each series
    sigma : float
        Filter roll-off frequency expressed in wavenumbers.
    dt_fs : float
        Time step of time series expressed in femtoseconds.
    reflect : bool, default=True
        Add a reflection of the time series to itself prior to filtering, in order to remove
        "jump" effects near the initial and final points.

    Returns
    -------
    filtered : np.ndarray
        Signal after lowpass filter has been applied, in the same shape as all_raw_time_series
    freqx : np.ndarray
        Frequency axis for plotting spectra in units of cm^-1
    ft_original : np.ndarray
        Fourier-transform of doubled signal truncated to original signal length
    ft_filtered : np.ndarray
        Fourier-transform of doubled signal with filter applied, truncated to original signal length
    """
    # Length of trajectory
    traj_length = all_raw_time_series.shape[1]
    # Conversion from sampling rate (inverse timestep in units of fs^-1) to wavenumber
    conversion = 33355.0/dt_fs
    # Order of the Butterworth filter
    order = 6
    
    # Calculate frequency cutoff in units of the Nyquist frequency (half of the sampling rate.)
    # First convert to units of the sampling rate (inverse timestep) then multiply by 2:
    # 
    # sigma      1 cm       dt_fs * fs        
    # ----- * ---------- * ------------ * 2 = low_cutoff
    #  cm     33500 * fs        dt            
    #
    # 1) Cutoff frequency in cm^-1
    # 2) Cutoff frequency in fs^-1
    # 3) Cutoff frequency in units of sampling rate dt^-1
    # 4) Cutoff frequency in units of Nyquist frequency
    low_cutoff = float(sigma)/conversion * 2.0
    
    # Create Butterworth filter coefficients
    b, a = butter(order, low_cutoff, btype='low')

    reflect = True
    if reflect:
        # Create the doubled time series
        reflected = np.fliplr(all_raw_time_series)
        # remove the enpoints of the reflected
        reflected = np.delete(reflected, -1, axis=1)
        reflected = np.delete(reflected, 0, axis=1)
        # attach the signal end-to-end with its reflection. The purpose of this is is get rid of the tailing issue
        doubled_series = np.hstack((all_raw_time_series, reflected))
        new_len = len(doubled_series[0]) # length of doubled end-to-end time series
        # list of elements that make up the reflected portion (to be used to remove the reflected portion later on)
        removal_list = list(range(traj_length, new_len))
        # create the frequency portion (for plotting)
        w, h = freqz(b, a, worN=new_len, whole=True)
        # fast fourier transform
        ft = np.fft.fft(doubled_series)
        ft_filtered = ft*abs(h) if low_cutoff > 0.0 else ft
        filtered = np.fft.ifft(ft_filtered)
        # Delete the data from the reflection addition
        filtered = np.delete(filtered, removal_list, axis=1)
        ft_original = np.delete(ft, removal_list, axis=1)
        ft_filtered = np.delete(ft_filtered, removal_list, axis=1)
        w = np.delete(w, removal_list)
        freqx = w*conversion/(2*np.pi)
    else:
        # create the frequency portion (for plotting)
        w, h = freqz(b, a, worN=traj_length, whole=True)
        abs_of_h = abs(h) # butterworth filter
        # fast fourier transform
        ft_original = np.fft.fft(all_raw_time_series)
        # multiply the FT by the filter to filter out higher frequencies
        ft_filtered = ft_original*abs_of_h if low_cutoff > 0.0 else ft_original
        filtered = np.fft.ifft(ft_filtered)
        freqx = w*conversion/(2*np.pi)
        
    return abs(filtered), freqx, ft_original, ft_filtered

def load_bondorder(boin, thre, traj_length):
    """
    Load a bondorder.list file.  

    This file format only lists bond orders above a threshold (typically 0.1)
    in each frame. Thus, the returned data takes the form of a sparse array.

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

    # LPW 2020-03-24 commented out for now
    # dm_all = None
    # for k, v in boSparse_sorted.items():
    #     amask = np.ma.array(v, mask=(v==0.0))
    #     am_fut = amask[1:]
    #     am_now = amask[:-1]
    #     dm = np.ma.abs(am_fut - am_now)
    #     if dm_all is None:
    #         dm_all = dm.copy()
    #     else:
    #         dm_all = np.ma.vstack((dm_all, dm.copy()))
    # maxVals = np.ma.max(dm_all, axis=0)
    # maxArgs = np.ma.argmax(dm_all, axis=0)
    # pairs = list(boSparse_sorted.keys())
    # for i in range(dm_all.shape[1]):
    #     print "%5i % .4f %i %i" % (i, maxVals[i], pairs[maxArgs[i]][0], pairs[maxArgs[i]][1])
    return boSparse_sorted

def formulaSum(efList):
    """ Takes a list of empirical formulas such as ['H2O', 'H2O', 'CH4'] and returns '2H2O+CH4'. """
    count = Counter(efList)
    words = []
    for v in sorted(list(set(count.values())))[::-1]:
        for k, v1 in list(count.items()):
            if v == v1:
                if v == 1:
                    words.append(k)
                else:
                    words.append('%i%s' % (v, k))
    return '+'.join(words)

class Nanoreactor(Molecule):
    def __init__(self, xyzin=None, qsin=None, properties='properties.txt', dt_fs=0.0, boin='bond_order.list', bothre=0.0,
                 enhance=1.4, mindist=1.0, printlvl=0, known=['all'], exclude=[], learntime=100.0, cutoff=100.0, padtime=0, save_molecules=False, frames=0, saverxn=True,
                 neutralize=False, radii=[], align=False, pbc=0.0, plot=False):
        #==========================#
        #         Settings         #
        #==========================#
        # Enhancement factor for determining whether two atoms are bonded
        self.Fac = enhance
        # Switch for whether to save molecules to disk.
        self.save_molecules = save_molecules
        # Switch for printing make-movie.tcl ; this is not necessary and may be deprecated soon
        self.Render = False
        # Known molecules to be excluded from coloring (by empirical formula)
        # Note: Isomers formed later are still considered interesting.  This is a hack.
        self.KnownFormulas = set(known)
        # Exclude certain molecules from being counted in any reaction event
        self.ExcludedFormulas = set(exclude)
        if printlvl >= 1 and len(self.ExcludedFormulas) > 0: print(self.ExcludedFormulas, "is excluded from being part of any reaction")
        # The print level (control the amount of printout)
        self.printlvl = printlvl
        # List of favorite colors for VMD coloring (excluding reds)
        self.CoolColors = [23, 32, 11, 19, 3, 13, 15, 27, 22, 6, 4, 12, 7, 9, 10, 28, 17, 21, 26, 24, 18]
        # Molecules that live for at least this long will be colored in VMD.
        # Also, molecules that vanish for (less than) this amount of time will have their time series filled in.
        self.LearnTime = learntime
        if padtime != 0:
            self.PadTime = padtime
        else:
            self.PadTime = self.LearnTime
        # Whether to extract molecules to neutralize the system
        self.neutralize = neutralize
        # Bond order threshold
        self.boThre = bothre
        # Keep time series that come within this factor of the threshold
        self.sparsePad = 1.2
        # Whether to align molecules / reactions prior to output
        self.align = align
        
        #==========================#
        #   Load in the XYZ file   #
        #==========================#
        if xyzin == None:
            raise Exception('Nanoreactor must be initialized with an .xyz file as the first argument')
        self.timing(super(Nanoreactor, self).__init__, "Loading molecule", xyzin)
        # Rudimentary periodic boundary condition support; cubic box only.
        # Later we can support more flexible PBCs by passing a trajectory format that supports them
        # using code already in molecule.py
        if pbc > 0.0:
            self.boxes = [BuildLatticeFromLengthsAngles(pbc, pbc, pbc, 90.0, 90.0, 90.0) for i in range(len(self))]
            
        #===============================#
        #   Load charge and spin data   #
        #===============================#
        if qsin != None and os.path.exists(qsin):
            QS = self.timing(Molecule, "Loading charge and spin populations", qsin, ftype="xyz")
            QSarr = np.array(QS.xyzs)
            self.Charges = QSarr[:, :, 0]
            self.Spins = QSarr[:, :, 1]
            self.have_pop = True
        else:
            self.Charges = np.array([[0 for i in range(self.na)] for j in range(len(self))])
            self.Spins = np.array([[0 for i in range(self.na)] for j in range(len(self))])
            self.have_pop = False
            
        #==========================#
        #   Load bond order data   #
        #==========================#
        self.boHave = False
        if boin != None and os.path.exists(boin) and bothre > 0.0:
            self.boHave = True
            self.boSparse = self.timing(load_bondorder, "Loading pairwise bond orders", boin, bothre/self.sparsePad, len(self))
        elif bothre > 0.0:
            raise RuntimeError('To use bond order threshold, must provide bond order list via "boin" argument')
        
        #=====================#
        #   Load properties   #
        #=====================#
        if os.path.exists(properties):
            if dt_fs != 0.0:
                raise RuntimeError("%s exists, don't provide a time step" % properties)
            props = np.loadtxt('properties.txt')
            keys = open('properties.txt').readline().split()[1:]
            self.propDict = OrderedDict([(keys[i], props[:,i]) for i in range(len(keys))])
            self.dt_fs = self.propDict['Time(fs)'][1]-self.propDict['Time(fs)'][0]
        else:
            if dt_fs == 0.0:
                raise RuntimeError("%s doesn't exist, provide a nonzero time step" % properties)
            self.dt_fs = dt_fs
        self.LearnTime = int(self.LearnTime / self.dt_fs)
        self.PadTime = int(self.PadTime / self.dt_fs)
        self.freqCutoff = cutoff
            
        if self.printlvl >= 0: print("Done loading files")
        print("The simulation timestep is %.1f fs" % self.dt_fs)
        print("Identification time for molecules is %.1f fs" % self.LearnTime)
        if self.freqCutoff == 0.0:
            print("Skipping lowpass filter on time series")
        else:
            print("Lowpass filter cutoff is %.1f cm^-1 (%.1f fs)" % (self.freqCutoff, 33355.0/self.freqCutoff))
        print("Padding each reaction event with %.1f fs" % self.PadTime)
        
        #==========================#
        #   Initialize Variables   #
        #==========================#
        # An iterator over all atom pairs, for example: [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        self.AtomIterator = np.array(list(itertools.combinations(list(range(self.na)),2)))
        # Set of isomers that are RECORDED.
        self.Recorded = set()
        # A time-series of atom-wise isomer labels.
        self.IsoLabels = []
        if self.boHave:
            self.boFiltered = self.timing(self.tsFilter, "Filtering bond order time series", self.boSparse, self.boThre, self.freqCutoff, 'bo', 'plot_bo.pdf' if plot else None)
        else:
            # Replace default radii with custom radii.
            for i in range(0, len(radii), 2):
                atom_symbol = radii[i]
                custom_rad = float(radii[i+1])
                Radii[Elements.index(atom_symbol)-1] = custom_rad
                print("Custom covalent radius for %2s : %.3f" % (atom_symbol, custom_rad))
            # Measure interatomic distances.
            self.dxSparse, self.dxThre = self.timing(self.measureDistances, "Measuring interatomic distances", self.sparsePad, mindist)
            self.dxFiltered = self.timing(self.tsFilter, "Filtering distance time series", self.dxSparse, self.dxThre, self.freqCutoff, 'dx', 'plot_dx.pdf' if plot else None)
        # self.global_graphs : A list of all possible ways the atoms are connected in the whole system, consisting of a list of 2-tuples.
        # self.gg_frames : An OrderedDict that maps start_time : (index in self.global_graphs, end_time)
        # self.BondLists : A time series of VMD-formatted bond specifications for each frame in the trajectory.
        self.global_graphs, self.gg_frames, self.BondLists = self.timing(self.makeGlobalGraphs, "Making global graphs",
                                                                         self.boFiltered if self.boHave else self.dxFiltered,
                                                                         self.boThre if self.boHave else self.dxThre,
                                                                         'bo' if self.boHave else 'dx')
        #========================#
        #| Make molecule graphs #|
        #========================#
        # self.Isomers : List of MyG() graph objects for all of the isomers found in the system.
        #                "Isomer index" refers to the position of the isomer in this list.
        # self.MolIDs  : List of molecule IDs for all molecules found in the system such as '117-121,138:123'
        #                (i.e. list of atoms in 'comma-dash' format:isomer index).
        #                "Molecule index" refers to the position of the molecule in this list.
        # self.TimeSeries : Ordered dictionary that maps molecule ID to time series data for that molecule.
        # self.traj_iidx : Array that maps (frame, atom) to the isomer index of that atom
        # self.traj_midx : Array that maps (frame, atom) to the molecule index of that atom
        # self.traj_stable : Array that maps (frame, atom) to whether the molecule containing this atom is currently stable
        # self.known_iidx : List of isomers that are 'known', i.e. matching user-provided empirical formulas and excluded from coloring
        self.Isomers, self.MolIDs, self.TimeSeries, self.traj_iidx, self.traj_midx, self.traj_stable, self.known_iidx = self.timing(self.makeMoleculeGraphs, "Making molecule graphs")
        if hasattr(self, 'boxes'): self.timing(self.makeWhole, "Making molecules whole")
        self.IsomerData, self.traj_color = self.timing(self.analyzeIsomers, "Analyzing isomers")

        #========================#
        #| Find reaction events #|
        #========================#
        # self.EventIDs : List of reaction event IDs such as '5812-5921:10,133-135,150-151,220'
        #                 (i.e. starting and ending frames, inclusive of endpoints:list of atoms)
        # self.Events : Ordered dictionary that maps reaction event IDs to relevant information such as:
        #               the list of molecule IDs that are involved and a "chemical equation" string
        self.EventIDs, self.Events = self.timing(self.findReactionEvents, "Finding reaction events")

    def timing(self, func, msg, *args, **kwargs):
        """
        Wrapper function that prints out timing information for a function or method.
        (intended to be called by the constructor)
        
        Parameters
        ----------
        func : function or method
            The function that's being wrapped
        msg : string
            String to be printed before the function call
        *args, **kwargs : 
            Positional and keyword arguments expected by the function
        """
        if self.printlvl >= 0:
            print(msg + " ...", end=' ')
            t0 = time.time()
        ret = func(*args, **kwargs)
        if self.printlvl >= 0:
            print("%.3f s" % (time.time()-t0))
        return ret

    def measureDistances(self, pad, mindist=1.0):
        """
        Measure interatomic distances.  Only keep timeseries whose minimum values
        are below the distance threshold times (pad).  Distance threshold is determined
        by the sum of covalent radii of the atom pair times (Fac).
        (intended to be called by the constructor)

        Parameters
        ----------
        pad : float
            Keep timeseries whose minimum value is (pad) times the distance threshold.
        mindist : float
            Below this distance all atom pairs are considered to be bonded.
        
        Returns
        -------
        dxSparse : OrderedDict 
            Dictionary that maps zero-indexed atom pairs (a1, a2) to numpy array
            containing interatomic distances between a1, a2 for each frame. 
            Keys only include a2 > a1.
        dxThre : OrderedDict 
            Dictionary that maps zero-indexed atom pairs (a1, a2) to distance
            threshold between a1, a2 for each frame.
        """
        # Create an atom-wise list of covalent radii.
        R = np.array([(Radii[Elements.index(i)-1] if i in Elements else 0.0) for i in self.elem])
        # A list of threshold distances for each atom pair for determining whether two atoms are bonded
        self.BondThresh = np.array([max(mindist,(R[i[0]] + R[i[1]]) * self.Fac) for i in self.AtomIterator])
        i = 0
        # Atom pair batch size for computing interatomic distance.
        # The maximum array size is batch_size * traj_length
        # (Avoids n_atom * n_atom * traj_length array)
        batch = 10000
        dxSparse = OrderedDict()
        dxThre = OrderedDict()
        # Build graphs from the distance matrices
        while i < len(self.AtomIterator):
            if self.printlvl >= 2: print("%i/%i" % (i, len(self.AtomIterator)))
            j = min(i+batch, len(self.AtomIterator))
            if hasattr(self, 'boxes'):
                boxes = np.array([[self.boxes[s].a, self.boxes[s].b, self.boxes[s].c] for s in range(len(self))])
                dxij = AtomContact(np.array(self.xyzs), self.AtomIterator[i:j], box=boxes)
            else:
                dxij = AtomContact(np.array(self.xyzs), self.AtomIterator[i:j])
            dxmin = np.min(dxij, axis=0)
            thre = self.BondThresh[i:j]
            for k in np.where(dxmin < (thre*pad))[0]:
                dxSparse[tuple(self.AtomIterator[i+k])] = dxij[:, k].copy()
                dxThre[tuple(self.AtomIterator[i+k])] = self.BondThresh[i+k]
            i += batch
        return dxSparse, dxThre
            
    def tsFilter(self, tsData, tsThre, freqCut, mode, plotFile=None):
        """
        Apply a low-pass filter to bond order or interatomic distance time series.
        (intended to be called by the constructor)

        Parameters
        ----------
        tsData : OrderedDict 
            Dictionary that maps zero-indexed atom pairs (a1, a2) to numpy array
            containing numerical time series data between a1, a2 for each frame. 
        tsThre : float or OrderedDict
            If OrderedDict, map zero-indexed atom pairs (a1, a2) to threshold value.
            If float, use a single threshold value for all time series.
        freqCut : float
            Roll-off frequency in wavenumbers. 
        mode : str
            Pass either 'bo' or 'dx' for bond order or interatomic distance respectively.
        plotfile : str
            File name ending in .pdf for saving plots of raw and filtered time series.
        
        Returns
        -------
        OrderedDict
            Same structure as tsData, with filter applied
        """
        tsPairs = list(tsData.keys())
        tsArr = np.array(list(tsData.values()))
        tsArr_lp, freqx, ft, ft_lp = low_pass_smoothing(tsArr, freqCut, self.dt_fs)
        # The bulk of this function is actually for plotting
        if plotFile is not None:
            if mode == 'dx':
                title = 'Time series of interatomic distances; lowpass filter ' + r'%i cm$^{-1}$' % freqCut
                y1label = 'Distance (Angstrom)'
                y1lim = [0, 5]
                y1ticks = [1, 2, 3, 4]
                y2fac = 0.1
            elif mode == 'bo':
                title = 'Time series of bond order; lowpass filter %i cm^-1' % freqCut
                y1label = 'Bond order'
                y1lim = [0, 1]
                y1ticks = [0.2, 0.4, 0.6, 0.8]
                y2fac = 0.001
            else:
                raise RuntimeError('mode %s not recognized' % mode)
            fout = PdfPages(plotFile)
    
            histograms = OrderedDict()
            for i in range(0, len(tsArr)):
                if self.printlvl >= 2 and i%100 == 0: print("Plotting timeseries %i/%i" % (i, len(tsArr)))
                fign = i%10
                if fign == 0:
                    fig = plt.figure()
                    fig.set_size_inches(8, 8)
                # Plot raw and filtered time series on left panels
                ax1 = fig.add_axes([0.07, 0.87-0.09*fign, 0.37, 0.09])
                times = np.arange(len(self))*self.dt_fs
                ax1.plot(times, tsArr[i], color='#1e90ff', linewidth=0.75)
                ax1.plot(times, tsArr_lp[i], color='#ff6302', linewidth=0.75)
                if isinstance(tsThre, OrderedDict):
                    thre = tsThre[tsPairs[i]]
                else:
                    thre = tsThre
                # Dotted horizontal line showing threshold
                ax1.axhline(thre, color='k', linestyle='--', linewidth=0.75)
                ax1.set_xlim([min(times), max(times)])
                # If using bond orders, set separate limits for each time series
                if mode == 'bo':
                    if max(tsArr[i]) > 3.0:
                        raise RuntimeError('Did not expect BO above 3.0')
                    elif max(tsArr[i]) > 2.0:
                        y1lim = [0, 3]
                        y1ticks = [1, 2]
                    elif max(tsArr[i]) > 1.0:
                        y1lim = [0, 2]
                        y1ticks = [0.5, 1, 1.5]
                    else:
                        y1lim = [0, 1]
                        y1ticks = [0.2, 0.4, 0.6, 0.8]
                ax1.set_ylim(y1lim)
                ax1.set_yticks(y1ticks)
                # Plot spectrum of the BO time series on the right panels
                ax2 = fig.add_axes([0.58, 0.87-0.09*fign, 0.36, 0.09])
                ax2.plot(freqx, np.abs(ft[i]**2), color='#1e90ff', linewidth=0.5)
                ax2.plot(freqx, np.abs(ft_lp[i]**2), color='#ff6302', linewidth=0.5)
                # Dotted vertical line showing frequency cutoff
                if freqCut > 0.0:
                    ax2.axvline(freqCut, color='k', linestyle='--', linewidth=0.75)
                ai, aj = tsPairs[i]
                ax2.text(0.9, 0.8, '%s-%s %i-%i' % (self.elem[ai], self.elem[aj], ai+1, aj+1),
                         horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)
                highFreq = False
                if highFreq:
                    ax2.set_xlim([0, 5000])
                    ax2.set_ylim([0, y2fac*0.01*(ft[i].shape[0])**2])
                else:
                    ax2.set_xlim([0, freqCut*4 if freqCut > 0.0 else 1000])
                    ax2.set_ylim([0, y2fac*(ft[i].shape[0])**2])
                ax2.set_yticks([])

                ax2b = ax2.twinx()
                ax2b.plot(freqx, np.abs(ft_lp[i]**2)/np.abs(ft[i]**2), color='r', linestyle='--', linewidth=0.75)
                ax2b.axhline(0.5, color='k', linestyle='--', linewidth=0.75)
                if highFreq:
                    ax2b.set_xlim([0, 5000])
                else:
                    ax2b.set_xlim([0, freqCut*4 if freqCut > 0.0 else 1000])
                ax2b.set_yscale('log')
                ax2b.set_ylim([0.001, 10])
                ax2b.set_yticks([0.01, 0.1, 1])
                # ax2b.set_yticks([])
                
                # Plot histogram of the bond order / distance for this atom pair
                ax3 = fig.add_axes([0.44, 0.87-0.09*fign, 0.06, 0.09])
                # Construct histograms of bond order or distance data for each element pair
                # Do this only once for each element pair because it's time-consuming.
                tsElem = tuple(sorted([self.elem[tsPairs[i][0]], self.elem[tsPairs[i][1]]]))
                if tsElem not in histograms:
                    tsSame = np.array([k for k, j in enumerate(tsPairs) if
                                       ((self.elem[j[0]], self.elem[j[1]]) == tsElem or
                                        (self.elem[j[1]], self.elem[j[0]]) == tsElem)])
                    tsHist = (tsArr[tsSame,:]+(tsArr[tsSame,:]==0.0)*1e3).flatten()
                    if mode == 'bo':
                        tsMax = int(np.max(tsArr[tsSame,:]))+1
                        heights, bins = np.histogram(tsHist, bins=72, range=[0, tsMax], density=True)
                    elif mode == 'dx':
                        heights, bins = np.histogram(tsHist, bins=72, weights=tsHist**-2, range=y1lim, density=True)
                    histograms[tsElem] = (bins[:-1]+(bins[1]-bins[0])/2, heights)
                    
                # To plot a pre-computed histogram, use bin midpoints as x data and heights as y data
                histx, histy = histograms[tsElem]
                nbin = len(np.where(histx<y1lim[1])[0])
                ax3.hist(histx[:nbin], bins=nbin, weights=histy[:nbin], log=True, range=y1lim, color='#6BE140',
                         edgecolor='#6BE140', linewidth=0, orientation='horizontal')
                ax3.set_ylim(y1lim)
                ax3.get_xaxis().set_visible(False)
                ax3.get_yaxis().set_visible(False)
                ax3.axhline(thre, color='k', linestyle='--', linewidth=0.75)
                ax3.text(1.1, thre/ax3.get_ylim()[1], '%.2f' % thre, horizontalalignment='left',
                         verticalalignment='center', transform=ax3.transAxes)
                # Final formatting
                if fign == 9 or i == tsArr.shape[0]-1:
                    fig.suptitle(title, y=0.985)
                    ax1.set_xlabel('Time (fs)')
                    ax2.set_xlabel('Frequency (cm^-1)')
                    fout.savefig(fig, dpi=600)
                    plt.close(fig)
                else:
                    if fign == 5:
                        ax1.set_ylabel(y1label)
                        ax2.set_ylabel('Intensity')
                    ax1.tick_params(axis='x', direction='in', length=2)
                    ax2.tick_params(axis='x', direction='in', length=2)
                    ax1.set_xticklabels([])
                    ax2.set_xticklabels([])
            fout.close()
        return OrderedDict([(k, tsArr_lp[i, :]) for i, k in enumerate(tsPairs)])

    def makeGlobalGraphs(self, tsData, tsThre, mode):
        """
        Create list of global pairwise connectivity graphs.
        (intended to be called by the constructor)

        Parameters
        ----------
        tsData : OrderedDict 
            Dictionary that maps zero-indexed atom pairs (a1, a2) to numpy array
            containing numerical time series data between a1, a2 for each frame. 
        tsThre : float or OrderedDict
            If OrderedDict, map zero-indexed atom pairs (a1, a2) to threshold value.
            If float, use a single threshold value for all time series.
        mode : str
            If pass 'dx', bonded when distances are lower than threshold
            If pass 'bo', bonded when bond orders are greater than threshold
        
        Returns
        -------
        global_pairGraphs : list
            Each element in global_pairGraphs is a list of 2-tuples (bonds) corresponding to
            one set of atomic connectivities in the whole system
        gg_frames : OrderedDict
            Mapping of frame numbers to 2-tuple containing (corresponding entry in global_pairGraphs, next frame)
        """
        tsPairs = np.array(list(tsData.keys()))
        tsArr = np.array(list(tsData.values()))
        
        # Convert tsThre to array if needed
        if type(tsThre) is OrderedDict:
            tsThre = np.array(list(tsThre.values()))[:, np.newaxis]
        elif type(tsThre) is not float:
            raise RuntimeError('tsThre wrong type')

        # Make trajectory of bonded atom pairs
        if mode == 'dx':
            bonded = tsArr < tsThre
        elif mode == 'bo':
            bonded = tsArr > tsThre
        else:
            raise RuntimeError('mode may only be dx or bo')

        # List of global connectivity graphs. The idea is that we should only need to
        # store and analyze the distinct connectivity graphs, which should be smaller
        # in number than the # of frames in the whole trajectory.
        globalGraphs = []
        lastGraph = None
        # OrderedDict that maps (the frame number where a global graph first appears) to (global graph index)
        gg_frames = OrderedDict()
        gg_idx = 0
        for i in range(len(self)):
            # List of atom pairs indices that are bonded (pointing to atom pairs in tsPairs).
            gg = tuple(np.where(bonded[:, i])[0])
            # Skip if the graph hasn't changed since the last frame (should happen often)
            if (gg == lastGraph):
                continue
            # Loop over the existing global graphs that we have
            for igg in range(len(globalGraphs)):
                # The global graph that just appeared is a repeat of a previous one
                if globalGraphs[igg] == gg:
                    gg_idx = igg
                    if self.printlvl >= 2:
                        print("frame %i repeats global graph %i" % (i, gg_idx), end=' ')
                    gg_frames[i] = gg_idx
                    break
            else:
                # The global graph that just appeared has not been seen previously
                gg_idx = len(globalGraphs)
                if self.printlvl >= 2:
                    print("frame %i found new global graph: %i" % (i, gg_idx), end=' ')
                globalGraphs.append(gg)
                gg_frames[i] = gg_idx
            if i > 0 and self.printlvl >= 2:
                added = ['%i-%i' % (tsPairs[i, 0]+1, tsPairs[i, 1]+1) for i in sorted(list(set(gg) - set(lastGraph)))]
                removed = ['%i-%i' % (tsPairs[i, 0]+1, tsPairs[i, 1]+1) for i in sorted(list(set(lastGraph) - set(gg)))]
                if len(added) > 0: print("added", ','.join(added), end=' ')
                if len(removed) > 0: print("removed", ','.join(removed), end=' ')
                print()
            lastGraph = gg

        # Make list of global graphs that actually consists of lists of atom pairs
        # (rather than atom pair indices in tsPairs)
        tsPairs = [tuple(a) for a in tsPairs]
        global_pairGraphs = [[tsPairs[i] for i in g] for g in globalGraphs]
        # Now gg_frames should map (the frame number where the global graph first appears)
        # to a 2-tuple: (the index to the global graph, the frame number where the next global graph appears)
        for i, (k, v) in enumerate(gg_frames.items()):
            if i == len(gg_frames)-1:
                gg_frames[k] = (v, len(self))
            else:
                gg_frames[k] = (v, list(gg_frames.keys())[i+1])
                
        # Build the BondLists, a "trajectory of bonded pairs" for writing bonds.dat for visualization.
        BondLists = []
        for currFrame, (ggId, nextFrame) in list(gg_frames.items()):
            bonds = [[] for i in range(self.na)]
            for (ii, jj) in global_pairGraphs[ggId]:
                bonds[ii].append(jj)
                bonds[jj].append(ii)
            bondTcl = bondlist_tcl(bonds)
            for i in range(nextFrame-currFrame):
                BondLists.append(bondTcl)
                
        return global_pairGraphs, gg_frames, BondLists

    def makeMoleculeGraphs(self):
        """
        Create molecule graphs from global graphs.
        (intended to be called by the constructor)

        Returns
        -------
        Isomers : list
            List of unique isomers (molecular graphs). Two graphs are isomorphic if the atomic symbols and connectivities match.
            The position of an isomer within this list is called the "isomer index".
            Each element in this list is a MyG instance containing the atomic symbols, connectivity, and atom numbers
            (the atom number are not so relevant because they are not part of the isomorphism)

        MolIDs : list
            List of unique molecules. These are more specific than Isomers because they are further distinguished using atom numbers.

        TimeSeries : OrderedDict
            Dictionary that maps molecule IDs to relevant data for that molecule.
            Keys within TimeSeries are:
            ----
            graph : MyG
                Graph of the molecule ID, including atomic symbols, connectivity, and atom numbers
            iidx : int
                Isomer index of the graph
            midx : int
                Molecule index of the graph
            raw_signal : np.ndarray
                Existence time series containing zeros and ones
            stable_times : OrderedDict
                Dictionary that maps start_time to interval_length for segments of the existence
                time series in excess of LearnTime
            ----
            The position of a Molecule ID in this OrderedDict is called the "molecule index"

        traj_iidx : np.ndarray
            Array that maps (frame, atom number) to isomer index

        traj_midx : np.ndarray
            Array that maps (frame, atom number) to molecule index

        traj_stable : np.ndarray
            Array that specifies whether (frame, atom number) is part of a stable molecule

        known_iidx : set
            These isomer indices are deemed "known" because they match a user-supplied filter, and will not
            be highlighted in the output visualization
        """
        # Initialize variables
        Isomers = []
        TimeSeries = OrderedDict()
        known_iidx = set()
        traj_midx = np.zeros((len(self), self.na), dtype='int')
        traj_iidx = np.zeros((len(self), self.na), dtype='int')
        traj_stable = np.zeros((len(self), self.na), dtype='int')
        
        # Sanity checking: Each entry in traj_midx should be set once and only once when looping over molecule IDs
        traj_midx -= 1
        
        # Dictionary that aggregate isomers, used for optimization
        isomer_ef_iidx_dict = defaultdict(list)
        
        for igg, globalGraph in enumerate(self.global_graphs):
            # Get "alive times" for this global graph
            aliveIntvls = []
            for ggFrame, (ggId, nextFrame) in list(self.gg_frames.items()):
                if igg == ggId:
                    aliveIntvls.append((ggFrame, nextFrame))
            # Build the NetworkX graph object for this global graph and split it into
            # connected subgraphs (molecules)
            RawG = MyG()
            for i, a in enumerate(self.elem):
                RawG.add_node(i)
                if parse_version(nx.__version__) >= parse_version('2.0'):
                    nx.set_node_attributes(RawG,{i:a}, name='e')
                else:
                    nx.set_node_attributes(RawG,'e',{i:a})
            for (ii, jj) in globalGraph:
                RawG.add_edge(ii, jj)
            MolGphs = [RawG.subgraph(c).copy() for c in nx.connected_components(RawG)]
            for G in MolGphs:
                G.__class__ = MyG
                ef = G.ef()
                # iidx means Isomer Index. Compare to the Graph that has the same Empirical Formula
                for i in isomer_ef_iidx_dict[ef]:
                    if Isomers[i] == G:
                        iidx = i
                        break
                else:
                    iidx = len(Isomers)
                    Isomers.append(G)
                    isomer_ef_iidx_dict[ef].append(iidx)
                # Check if the empirical formula of this graph matches the known formulas provided by user
                if (ef in self.KnownFormulas or wildmatch(ef, self.KnownFormulas) or (igg == 0 and 'ALL' in [i.upper() for i in self.KnownFormulas])):
                    known_iidx.add(iidx)
                # Create the molecule ID and check if it's in the dictionary
                molID = G.AStr()+":%i" % iidx
                if molID not in TimeSeries:
                    raw_signal = np.zeros(len(self), dtype=int)
                    # This line is very important: it creates the entry in the TimeSeries dictionary
                    # that is the main repository of information for molecular graphs.
                    TimeSeries[molID] = OrderedDict([('graph', G), ('iidx', iidx), ('midx', len(TimeSeries)),
                                                     ('raw_signal', raw_signal), ('stable_times', OrderedDict())])
                else:
                    raw_signal = TimeSeries[molID]['raw_signal']
                for i, j in aliveIntvls: raw_signal[i:j] = 1

        MolIDs = list(TimeSeries.keys())

        for i, (molID, ts) in enumerate(TimeSeries.items()):
            # Create self.TimeSeries[molID]["stable_times"], an OrderedDict that maps starting points of stable segments to their lengths.
            # Populate self.traj_midx and self.traj_iidx, trajectory-like objects that store the molecule ID and isomer ID of each atom
            frame = 0
            atoms = np.array(ts['graph'].L())
            for intvl, on_off in encode(ts['raw_signal']):
                if on_off:
                    if (traj_midx[frame:frame+intvl, atoms] != -1).any(): raise RuntimeError('traj_midx is twice assigned')
                    traj_midx[frame:frame+intvl, atoms] = ts['midx']
                    traj_iidx[frame:frame+intvl, atoms] = ts['iidx']
                    if intvl > self.LearnTime:
                        ts['stable_times'][frame] = intvl
                        traj_stable[frame:frame+intvl, atoms] = 1
                frame += intvl
        if (traj_midx == -1).any(): raise RuntimeError('traj_midx is not fully assigned')
        
        if self.printlvl >= 2:
            for molID in TimeSeries:
                ts = TimeSeries[molID]
                atom_str, iidx_str = molID.split(':')
                ef_str = ts['graph'].ef()
                ts_str = ''.join([("/\u203E%i\u203E\\" % t[0]) if t[1] else ('_%i_' % t[0]) for t in encode(ts['raw_signal'])])
                print(("molecule index %i formula %s iidx %s atoms %s series %s" % (MolIDs.index(molID), ef_str, iidx_str, atom_str, ts_str)).encode('utf-8'))

        return Isomers, MolIDs, TimeSeries, traj_iidx, traj_midx, traj_stable, known_iidx

    def makeWhole(self):
        if not hasattr(self, 'boxes'): return
        for molID, ts in list(self.TimeSeries.items()):
            frame = 0
            G = ts['graph']
            atoms = np.array(G.L())
            for intvl, on_off in encode(ts['raw_signal']):
                if on_off:
                    for s in range(frame, frame+intvl):
                        xdict = dict([(i, self.xyzs[s][i]) for i in atoms])
                        if parse_version(nx.__version__) >= parse_version('2.0'):
                            nx.set_node_attributes(G, xdict, 'x')
                        else:
                            nx.set_node_attributes(G, 'x', xdict)
                        G.make_whole(self.boxes[s].a, self.boxes[s].b, self.boxes[s].c)
                        self.xyzs[s][atoms] = np.array(G.x())
                frame += intvl
    
    def allStable(self, frame, atoms, direction):
        """
        Given a frame, set of atoms, and a direction, scan the trajectory 
        along the direction until all of the atoms are stable, then return
        the frame number and the atoms that "complete" the molecules at that frame
        (intended to be called by completeEvent() function)

        Parameters
        ----------
        frame : int
            Frame number to start scanning from (this frame itself is not scanned)
        atoms : np.ndarray
            Atoms to check stability for
        direction : int
            Either +1 or -1, the direction to scan in

        Returns
        -------
        frame : int
            First frame that contains all stable atoms in the scan direction
        stable_mol_atoms : np.ndarray
            The atom indices that complete the stable molecules at the returned frame
        """
        if direction not in [1, -1]:
            raise RuntimeError('Direction must be +1 or -1')
        # Scan forward or backward until a new stable frame is found
        while True:
            frame += direction
            # print "Scanning %s" % ("forward" if direction > 0 else "backward"), frame
            if frame <= 0:
                frame = 0
                # print "Scanned past start"
                break
            if frame >= len(self)-1:
                # print "Scanned past end"
                frame = len(self)-1
                break
            if self.traj_stable[frame, atoms].all():
                break
        # The set of molecule indices that the atoms belong to at the new stable frame
        stable_midx = set(self.traj_midx[frame, atoms])
        # All of the atoms belonging to the above set of molecules
        stable_mol_atoms = sorted(list(itertools.chain(*[self.TimeSeries[self.MolIDs[i]]['graph'].L() for i in stable_midx])))
        return frame, np.array(stable_mol_atoms)

    def completeEvent(self, frame, atoms, direction):
        """
        The lingo of "completing" a reaction event refers to the following:

        1) Start at the last frame that a molecule is stable
        2) Follow the atoms forward in time until they all belong to stable molecules again
        3) Find the atoms that "complete" these stable molecules at that point in time, which
           may be a superset of the original atoms we started with
        4) Go back to step 1 with the expanded set of atoms and search backwards until
           all atoms belong to stable molecules
        5) Repeat until set of atoms no longer expands

        (intended to be called by findReactionEvents())
    
        Parameters
        ----------
        frame : int
            First or last frame that a molecule is stable, 
            the starting point for completing a reaction event
        atoms : np.ndarray
            The atoms belonging to this molecule
        direction : int, +1 or -1
            The direction to search in (+1 if it's the last frame,
            -1 if it's the first frame)

        Returns
        -------
        rxMin : int
            The initial frame of this reaction event (i.e. latest frame
            that all reactant molecules are stable)
        rxMax : int
            The last frame of this reaction event (i.e. the earliest frame
            that all product molecules are stable)
        refAtoms : np.ndarray
            All of the atoms that complete this reaction event
        """
        if direction not in [1, -1]:
            raise RuntimeError('Direction must be +1 or -1')
        refAtoms = atoms.copy()
        rxMin = rxMax = frame
        while True:
            rxFrame, rxAtoms = self.allStable(rxMax if direction > 0 else rxMin, refAtoms, direction)
            rxMin = min(rxMin, rxFrame)
            rxMax = max(rxMax, rxFrame)
            if len(refAtoms) == len(rxAtoms) and (rxAtoms == refAtoms).all():
                break
            # When reversing direction, take a step back to include the frame already scanned over
            if direction > 0: rxMin += 1
            else: rxMax -= 1
            refAtoms = rxAtoms.copy()
            direction *= -1
        if rxMin == rxMax:
            raise RuntimeError("completeEvent called for a frame that doesn't contain a reaction event")
        return rxMin, rxMax, refAtoms

    def getMolIDs(self, frame, atoms):
        """
        Return the molecule IDs for a set of atoms at a particular frame.
        This function assumes you are passing a list of atoms that correspond
        to complete and stable molecules at the chosen frame.
        (intended to be called by makeEvent() function)

        Parameters
        ----------
        frame : int
            The frame number where molecule IDs are queried
        atoms : np.ndarray
            The list of atoms for performing the query
        
        Returns
        -------
        molIDs : list
            A list of molecule ID strings
        """
        molidxs = set(self.traj_midx[frame, atoms])
        molIDs = [self.MolIDs[i] for i in sorted(list(molidxs))]
        molatoms = sorted(list(itertools.chain(*[self.TimeSeries[molID]['graph'].L() for molID in molIDs])))
        if not self.traj_stable[frame, atoms].all():
            print(frame, atoms, self.traj_stable[frame, atoms])
            raise RuntimeError("getMolIDs called using frames and atoms that do not have stable molecules")
        if (np.array(molatoms) != atoms).any():
            raise RuntimeError("getMolIDs called using frames and atoms that do not correspond to complete molecules")
        return molIDs

    def padFrames(self, rxMin, rxMax, atoms):
        """
        Pad the reaction event using up to "PadTime" 
        number of frames conditional on all molecules
        being stable and no additional reactions occurring.
        (intended to be called by makeEvent() function)

        Parameters
        ----------
        rxMin : int
            The first frame of the reaction event
        rxMax : int
            The final frame of the reaction event
        atoms : np.ndarray
            The atoms involved in the reaction event
        
        Returns
        -------
        int, int
            The "padded" first and final frames of the reaction event
        """
        midxMin = set(self.traj_midx[rxMin, atoms])
        midxMax = set(self.traj_midx[rxMax, atoms])
        # Scan back in time until the "molecules of these atoms" are no longer stable
        # or the molecule indices have changed.
        for minPad in range(self.PadTime+1):
            if rxMin-minPad == 0: break
            if not self.traj_stable[rxMin-minPad, atoms].all() or set(self.traj_midx[rxMin-minPad, atoms]) != midxMin:
                minPad -= 1
                break
        # Scan forward in time, as above
        for maxPad in range(self.PadTime+1):
            if rxMax+maxPad == 0: break
            if not self.traj_stable[rxMax+maxPad, atoms].all() or set(self.traj_midx[rxMax+maxPad, atoms]) != midxMax:
                maxPad -= 1
                break
        return rxMin-minPad, rxMax+maxPad

    def makeEvent(self, frame1, frame2, atoms):
        """
        Given the frames and atoms that make up a reaction event,
        create the "object" that represents the reaction event itself
        (currently an OrderedDict).  This function also adds sanity
        checks to see whether a reaction event really should be kept.
        (intended to be called by findReactionEvents() function)

        Parameters
        ----------
        frame1 : int
            The first frame of the reaction event (before padding)
        frame2 : int
            The last frame of the reaction event (before padding)
        atoms : np.ndarray
            The atoms that participate in this reaction event

        Returns
        -------
        EventID : str or None
            String that represents the reaction event as:
            frame1-frame2:commadash(atoms) e.g. 3814-3915:33,196-199
            (If not a valid reaction event, None will be returned)
        Event: OrderedDict or None
            Dictionary containing relevant data for the reaction event.
            Keys within Event are:
            ----
            molIDs : tuple
                2-tuple containing lists of molecule IDs of reactants and products
            frames : tuple
                2-tuple containing first and last frame of the reaction event
                These may be "padded" w/r.t. frame1 and frame2
            equation : str
                Chemical equation for the reaction event, such as 'H3N+CH4+HO->H4N+CH4O'
            atoms : np.ndarray
                Atoms that participate in this reaction event
                These may be reduced w/r.t. input atoms due to spectator removal
                (Future: Neutralization of reaction event should also occur here)
            ----
            (If not a valid reaction event, None will be returned)
        """
        if frame1 == 0:
            if self.printlvl >= 2: print("Scanned past start")
            return None, None
        if frame2 == len(self)-1:
            if self.printlvl >= 2: print("Scanned past end")
            return None, None
        molid1 = self.getMolIDs(frame1, atoms)
        molid2 = self.getMolIDs(frame2, atoms)
        common = set(molid1).intersection(set(molid2))
        if len(common) > 0:
            if self.printlvl >= 2: print("** Common molIDs: **", sorted(list(common)))
        # Remove spectator molecules from the reaction event. This makes an assumption that
        # we are ignoring molecules that "catalyze" the reaction. On the other hand, 
        # this method isn't guaranteed to find all molecules that catalyze the reaction,
        # so maybe that should be implemented in a separate function.
        molid1 = [m for m in molid1 if m not in common]
        molid2 = [m for m in molid2 if m not in common]
        if len(molid1) == 0:
            if self.printlvl >= 2: print("No molecules left after removing spectators")
            return None, None
        # Check if any of the molecules are included in the excluded formulas; if so, do not keep it
        for molID in molid1 + molid2:
            ef = self.TimeSeries[molID]['graph'].ef()
            if (ef in self.ExcludedFormulas or wildmatch(ef, self.ExcludedFormulas)):
                if self.printlvl >= 2: print("Reaction event includes excluded molecules")
                return None, None
        # Re-create the list of atoms which may be reduced now that spectators are removed
        atoms = np.array(sorted(list(itertools.chain(*[self.TimeSeries[molID]['graph'].L() for molID in molid1]))))
        frame1Pad, frame2Pad = self.padFrames(frame1, frame2, atoms)
        if self.getMolIDs(frame1Pad, atoms) != molid1 or self.getMolIDs(frame2Pad, atoms) != molid2:
            print(atoms, molid1, self.getMolIDs(frame1Pad, atoms), molid2, self.getMolIDs(frame2Pad, atoms), end=' ')
            raise RuntimeError('padFrames malfunction')
        if self.neutralize:
            neu_molID, success = self.getNeutralizing(frame1, frame2, atoms)
            molid1 += neu_molID
            molid2 += neu_molID
            atoms = np.array(sorted(list(itertools.chain(*[self.TimeSeries[molID]['graph'].L() for molID in molid1]))))
        formula1 = formulaSum([self.TimeSeries[molID]['graph'].ef() for molID in molid1])
        formula2 = formulaSum([self.TimeSeries[molID]['graph'].ef() for molID in molid2])
        # Create the reaction event object
        Event = OrderedDict([('molIDs', (molid1, molid2)), ('frames', (frame1Pad, frame2Pad)),
                             ('equation', "%s->%s" % (formula1, formula2)), ('atoms', atoms.copy())])
        EventID = '%i-%i:%s' % (frame1Pad, frame2Pad, commadash(atoms))
        if self.printlvl >= 2:
            print("Event ID:", EventID, "%s -> %s" % (formula1, formula2), end=' ')
            print("Frames: %i-%i" % (frame1Pad, frame2Pad), "Molecule IDs: %s -> %s" % (molid1, molid2))
        return EventID, Event
                        
    def findReactionEvents(self):
        """
        Top level function for finding reaction events.
        (intended to be called by the constructor after makeMoleculeGraphs())
        
        Returns
        -------
        EventIDs : list
            List of strings that represent reaction events as:
            frame1-frame2:commadash(atoms) e.g. 3814-3915:33,196-199
        Events: OrderedDict
            Dictionary mapping event IDs to relevant data for the reaction event.
            Keys within Events[EventID] are:
            ----
            molIDs : tuple
                2-tuple containing lists of molecule IDs of reactants and products
            frames : tuple
                2-tuple containing first and last frame of the reaction event
                These may be "padded" w/r.t. frame1 and frame2
            equation : str
                Chemical equation for the reaction event, such as 'H3N+CH4+HO->H4N+CH4O'
            atoms : np.ndarray
                Atoms that participate in this reaction event
                These may be reduced w/r.t. input atoms due to spectator removal
            ----
        """
        
        # The search for reaction events start at the edges of stable intervals
        unsortedEvents = {}
        for molNum, (molID, ts) in enumerate(self.TimeSeries.items()):
            if self.printlvl >= 2:
                print("=============")
                print("Molecule", molNum, molID, ts['graph'].ef())
            atoms = np.array(ts['graph'].L())
            for fstart, intvl in list(ts['stable_times'].items()):
                fend = fstart + intvl - 1
                if self.printlvl >= 2: print("Start Intvl End", fstart, intvl, fend)
                if fstart > 0:
                    # Look for reaction event that led to formation of this molecule
                    rstart, rend, ratoms = self.completeEvent(fstart, atoms, -1)
                    evid, event = self.makeEvent(rstart, rend, ratoms)
                    if evid is not None and evid not in unsortedEvents:
                        unsortedEvents[evid] = event
                if fend < len(self)-1:
                    # Look for reaction event that led to destruction of this molecule
                    rstart, rend, ratoms = self.completeEvent(fend, atoms, 1)
                    evid, event = self.makeEvent(rstart, rend, ratoms)
                    if evid is not None and evid not in unsortedEvents:
                        unsortedEvents[evid] = event

        sortkeys = []
        lookup = {}
        for evid in list(unsortedEvents.keys()):
            frameWord, atomWord = evid.split(':')
            atomList = uncommadash(atomWord)
            key = (int(frameWord.split('-')[0]), int(frameWord.split('-')[1])) + tuple(atomList)
            if key in sortkeys:
                print(key, evid, lookup[key])
                raise RuntimeError('key is duplicated in sortkeys')
            sortkeys.append(key)
            lookup[key] = evid

        EventIDs = []
        Events = OrderedDict()
        for key in sorted(sortkeys):
            evid = lookup[key]
            EventIDs.append(evid)
            Events[evid] = unsortedEvents[evid]

        if self.printlvl >= 2:
            for iev, evid in enumerate(EventIDs):
                print(iev, evid, Events[evid]['equation'], Events[evid]['molIDs'])
        elif self.printlvl >= 1:
            for iev, evid in enumerate(EventIDs):
                Event = Events[evid]
                print("Reaction event found (%i/%i) : frames %i-%i : %s" % (iev+1, len(Events), Event['frames'][0], Event['frames'][1], Event['equation']))

        return EventIDs, Events

    def analyzeIsomers(self):
        """
        Print information regarding isomers in the simulation.
        (intended to be called by the constructor)

        This may be called at any point after makeMoleculeGraphs()

        Returns
        -------
        IsomerData : OrderedDict
            Maps the isomer indices to relevant information for that isomer.  
            Keys within IsomerData are:
            ----
            graph : MyG
                Graph of the isomer ID, including atomic symbols, connectivity, and atom numbers 
                (atom numbers are irrelevant as they are molecule-specific)
            midx : list
                Molecule indices that have this isomer ID
            firstFound : int
                The frame in which this isomer first appeared
            stableIndices : list
                Atom indices of the stable appearances of this isomer
            stableIntervals : tuple
                List of first frames / interval lengths of the stable appearances of this isomer
            flag : str
                String descriptor of the status of this isomer, which is one of the following:
                "Found" : an interesting isomer that was found,
                "Known" : a known molecule matching user-provided empirical formula (and not highlighted)
                "Excluded" : molecule to be excluded from all reaction events,
                "Transient" : an isomer that exists for too short of a time to be assigned its own color
                              and to be used in reaction event finding
            color : int
                Color index in VMD for visualization.
            ----
        traj_color : np.ndarray
            Trajectory-like object containing the color of each atom for each frame in the trajectory
            (for VMD visualization)
        """
        # Dictionary containing relevant information for isomers
        IsomerData = OrderedDict()
        # Whether to list all isomers in the first frame as "known"
        knownFirst = 'all' in [i.lower() for i in self.KnownFormulas]
        
        for iidx, G in enumerate(self.Isomers):
            if G.ef() in self.ExcludedFormulas:
                flag = "Excluded"
            elif G.ef() in self.KnownFormulas:
                flag = "Known"
            elif wildmatch(G.ef(), self.KnownFormulas):
                flag = "Known"
            else:
                flag = "Transient"
            # The data for each isomer is contained in an OrderedDict
            IsomerData[iidx] = OrderedDict([('graph', G), ('midx', []), ('firstFound', len(self)), ('stableIndices', []),
                                            ('stableIntervals', []), ('flag', flag), ('color', 1)])

        # For each molecule, allocate information into the isomer data
        nFound = 0
        for midx, (molID, ts) in enumerate(self.TimeSeries.items()):
            iidx = ts['iidx']
            IData = IsomerData[iidx]
            IData['midx'].append(midx)
            frame = 0
            for intvl, on_off in encode(ts['raw_signal']):
                if on_off:
                    IData['firstFound'] = min(IData['firstFound'], frame)
                    if frame == 0 and knownFirst:
                        IData['flag'] = "Known"
                frame += intvl
            for frame, intvl in list(ts['stable_times'].items()):
                IData['stableIntervals'].append((frame, intvl))
                IData['stableIndices'].append(ts['graph'].L())
                if IData['flag'] == "Transient":
                    IData['flag'] = "Found"
                    nFound += 1

        # Sort isomers according to the first frame that any instance of the isomer occurs.
        # Molecules that are transient are printed at the end.
        sortKeys = []
        for iidx, IData in list(IsomerData.items()):
            firstStable = min([s[0] for s in IData['stableIntervals']]) if IData['stableIntervals'] else len(self)
            sortKeys.append((firstStable, iidx))

        # Print summary table.
        if self.printlvl >= 0:
            print()
            print("%10s %20s %10s %10s %10s %10s %10s %10s %20s" % ("Index", "Formula", "Instances", "firstFound", "firstStabl", "maxStable", "meanStable", "Flag", "Color"))
            print("="*120)
        ColorNum = 0
        nsave = 0
        for key in sorted(sortKeys):
            iidx = key[1]
            IData = IsomerData[iidx]
            if IData['flag'] == "Found":
                IData['color'] = self.CoolColors[ColorNum % len(self.CoolColors)]
                ColorNum += 1
            elif IData['flag'] in ["Known", "Excluded"]:
                IData['color'] = 8
            if self.printlvl >= 0:
                firstStable = min([s[0] for s in IData['stableIntervals']]) if IData['stableIntervals'] else len(self)
                maxInterval = max([s[1] for s in IData['stableIntervals']]) if IData['stableIntervals'] else 0
                avgLife = np.mean([i[1] for i in IData['stableIntervals']]) if IData['stableIntervals'] else 0
                print("%10i %20s %10i %10i %10i %10i %10.2f %10s %20s" % (iidx, IData['graph'].ef(), len(IData['midx']), IData['firstFound'], firstStable, maxInterval,
                                                                          avgLife, IData['flag'], "%s (%i)" % (ColorNames[IData['color']], IData['color'])))

        # Save longest stable intervals to the 'isomers' folder
        if self.save_molecules:
            for key in sorted(sortKeys):
                iidx = key[1]
                IData = IsomerData[iidx]
                if IData['flag'] == "Found":
                    iintvl = np.argmax([s[1] for s in IData['stableIntervals']])
                    frame, intvl = IData['stableIntervals'][iintvl]
                    atoms = IData['stableIndices'][iintvl]
                    traj_slice = self.atom_select(atoms)[frame:frame+intvl]
                    odir = 'molecules'
                    if not os.path.exists(odir): os.makedirs(odir)
                    fout = 'molecule_%03i.xyz' % nsave
                    formula = IData['graph'].ef()
                    if self.printlvl >= 2: print("Writing iidx %i (%i/%i stable), frames %i-%i to file %s : %s" % (iidx, nsave+1, nFound, frame, frame+intvl, fout, formula))
                    elif self.printlvl >= 1: print("Writing isomer %i/%i, frames %i-%i to file %s : %s" % (nsave+1, nFound, frame, frame+intvl, fout, formula))
                    traj_slice.comms = ["%s atoms %s frame %i charge %+.3f sz %+.3f sz^2 %.3f"
                                        % (formula, commadash(atoms), f, sum(self.Charges[f][atoms]),
                                           sum(self.Spins[f][atoms]), sum([j**2 for j in self.Spins[f][atoms]])) for f in range (frame, frame+intvl)]
                    if self.align:
                        traj_slice.center()
                        traj_slice.align()
                    traj_slice.write(os.path.join(odir, fout))
                    if self.have_pop:
                        # Write .xyz-like file containing Mulliken charge and spin populations
                        # in the first and second columns
                        traj_slice_pop = deepcopy(traj_slice)
                        pop_arr = np.zeros((len(traj_slice_pop), len(atoms), 3), dtype=float)
                        pop_arr[:, :, 0] = self.Charges[frame:frame+intvl, atoms]
                        pop_arr[:, :, 1] = self.Spins[frame:frame+intvl, atoms]
                        traj_slice_pop.xyzs = list(pop_arr)
                        traj_slice_pop.write(os.path.join(odir, fout.replace('.xyz', '.pop')), ftype='xyz')
                    nsave += 1
                    
        # Return trajectory of VMD color indices for visualization.
        traj_color = np.zeros((len(self), self.na), dtype='int')
        traj_color -= 1
        for midx, (molID, ts) in enumerate(self.TimeSeries.items()):
            color = IsomerData[ts['iidx']]['color']
            atoms = np.array(ts['graph'].L())
            frame = 0
            for intvl, on_off in encode(ts['raw_signal']):
                if on_off:
                    if (traj_color[frame:frame+intvl, atoms] != -1).any(): raise RuntimeError('traj_color is twice assigned')
                    traj_color[frame:frame+intvl, atoms] = color
                frame += intvl
        if (traj_color == -1).any(): raise RuntimeError('traj_color is not fully assigned')
        return IsomerData, traj_color

    def writeReactionEvents(self):
        """
        Write stored reaction events to files.
        (intended to be called by Output() function)

        To keep things tidy, reaction events will be written to a 'reactions' subdirectory.
        
        Each reaction event will be written to a numbered output file such as reaction_000.xyz, 
        and if it's a repeat of an existing reaction (i.e. reactant and product isomer IDs 
        are the same), the file name will be appended to reaction_000_01.xyz and so on.
        
        Mulliken populations will be written to reaction_000.pop, which are XYZ-formatted
        but have charges and spins written to the columns for x- and y-coordinates.
        """
        # Keep a list of output isomer indices for the purpose of seeing which
        # reaction events duplicate ones already written
        output_iidx = []
        for iev, (evid, event) in enumerate(self.Events.items()):
            r_iidx = sorted([self.TimeSeries[i]['iidx'] for i in event['molIDs'][0]])
            p_iidx = sorted([self.TimeSeries[i]['iidx'] for i in event['molIDs'][1]])
            for iout, (r_out, p_out) in enumerate(output_iidx):
                if (r_iidx == r_out and p_iidx == p_out) or (r_iidx == p_out and p_iidx == r_out):
                    if self.printlvl >= 2: print("reaction ID %s repeats output id %i" % (evid, iout))
                    event['output_id'] = iout
                    break
            else:
                event['output_id'] = len(output_iidx)
                if self.printlvl >= 2: print("reaction ID %s assigned output id %i" % (evid, len(output_iidx)))
                output_iidx.append((r_iidx, p_iidx))

        odir = 'reactions'
        if not os.path.exists(odir): os.makedirs(odir)
        # This is a double loop, first over the unique reactant/product isomer indices,
        # then over all of the reaction events that match these isomer indices.
        for iout in range(len(output_iidx)):
            repeat = 0
            subd = os.path.join (odir, 'reaction_%03i' % iout)
            for iev, (evid, event) in enumerate(self.Events.items()):
                if not os.path.exists(subd): os.makedirs(subd)
                if event['output_id'] == iout:
                    # Determine the file name
                    if repeat == 0:
                        fout = 'reaction_%03i.xyz' % iout
                    else:
                        fout = 'reaction_%03i_%02i.xyz' % (iout, repeat)
                    # Figure out the first and last frame by parsing the event ID
                    fstart, fend = self.Events[evid]['frames']#[int(i) for i in evid.split(':')[0].split('-')]
                    atoms = self.Events[evid]['atoms']#np.array(uncommadash(evid.split(':')[1]))
                    traj_slice = self.atom_select(atoms)[fstart:fend+1]
                    if self.printlvl >= 2: print("Writing event ID %s, number %i, frames %i-%i to file %s" % (evid, iev, fstart, fend, fout))
                    elif self.printlvl >= 1: print("Writing event number %i/%i, frames %i-%i to file %s : %s" % (iev+1, len(self.Events), fstart, fend, fout, self.Events[evid]['equation']))
                    a = event['atoms']
                    traj_slice.comms = ["%s atoms %s frame %i charge %+.3f sz %+.3f sz^2 %.3f"
                                        % (event['equation'], commadash(a), f, sum(self.Charges[f][a]),
                                           sum(self.Spins[f][a]), sum([j**2 for j in self.Spins[f][a]])) for f in range (fstart, fend+1)]
                    if self.align:
                        traj_slice.center()
                        traj_slice.align()
                    traj_slice.write(os.path.join(subd, fout))
                    if self.have_pop:
                        # Write .xyz-like file containing Mulliken charge and spin populations
                        # in the first and second columns
                        traj_slice_pop = deepcopy(traj_slice)
                        pop_arr = np.zeros((len(traj_slice_pop), len(atoms), 3), dtype=float)
                        pop_arr[:, :, 0] = self.Charges[fstart:fend+1, atoms]
                        pop_arr[:, :, 1] = self.Spins[fstart:fend+1, atoms]
                        traj_slice_pop.xyzs = list(pop_arr)
                        traj_slice_pop.write(os.path.join(subd,fout.replace('.xyz', '.pop')), ftype='xyz')
                    repeat += 1

    def WriteChargeSpinLabels(self):
        """
        Write charge and spin labels to charge.dat and spin.dat for use in VMD visualization.
        """
        # LPW 2019-03-04 Increasing threshold to 0.25 de-clutters visualization
        Threshold = 0.25
        ChargeLabels = [[] for i in range(self.ns)]
        #print self.Recorded
        RecSeries = {}
        for molID,ts in list(self.TimeSeries.items()): # Loop over graph IDs and time series
            if ts['iidx'] in self.Recorded:
                RecSeries[molID] = ts

        for molID,ts in list(RecSeries.items()):
            idx = np.array(ts['graph'].L())
            decoded = decode(ts['raw_signal'])
            for i in range(self.ns):
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
                # This generates a string like 100 101 102! 103, 0.25; which indicates the atoms in a molecule, the net charge on that molecule, and the atom
                # with the greatest charge marked by an exclamation point
                print("%s, %s;" % (' '.join(["%i%s" % (i, "!" if i == AtomChgTuple[2] else "") for i in AtomChgTuple[0]]), AtomChgTuple[1]), end=' ', file=qout)
            print(file=qout)

        Threshold = 0.25
        sout = open('spin.dat','w')
        for i in range(self.ns):
            for a in range(self.na):
                if abs(self.Spins[i][a]) >= Threshold:
                    print("%i %+.2f" % (a, self.Spins[i][a]), end=' ', file=sout)
            print(file=sout)
        qout.close()

    def getNeutralizing(self, frame1, frame2, atoms, tol=0.25):
        """
        Given two frames and a list of atoms, find a list of atoms that:
        
        1) belong to molecules that exist from frame1:frame2
           and don't intersect with the provided atoms
        2) neutralizes the overall system
        3) is close to the original set of atoms in space
        4) keeps charge and spin consistent

        Parameters
        ----------
        frame1 : int
            First frame of the interval
        frame2 : int
            Last frame of the interval (inclusive of endpoints)
        atoms : list
            List of atoms to be neutralized
        tol : float
            Neutralization tolerance

        Returns
        -------
        counter_atoms : list
            List of atoms that satisfies the above conditions
        """
        frames = np.arange(frame1, frame2+1)
        # Get the current charge / spin on the atom selection
        chg = np.mean(np.sum(self.Charges[frame1:frame2+1, atoms], axis=1))
        spn = np.mean(np.sum(self.Spins[frame1:frame2+1, atoms], axis=1))
        # Don't add any molecules if the molecule is already neutralized
        if np.abs(chg) < tol:
            return [],[]
        if self.printlvl >= 2: print("Attempting to neutralize atoms %s (charge %+.3f spin %+.3f)" % (commadash(atoms), chg, spn))
        xyz = np.array(self.atom_select(atoms)[frames].xyzs)
        
        # Ordered dictionary of candidate molecules to be added to the list
        Candidates = OrderedDict()
        Candidate = namedtuple('Candidate', ['mindx', 'maxdx', 'chg', 'spn'])
        # Loop over all molecules in the time series
        for molID, ts in list(self.TimeSeries.items()):
            # Check to make sure that this molecule exists for ALL frames in the frame selection
            # and does not overlap with ANY atoms in our atom selection.
            if set(ts['graph'].L()).intersection(set(atoms)): continue
            if not ts['raw_signal'][frames].all(): continue
            # List of atoms in this molecule
            c_atoms = ts['graph'].L()
            c_chg = np.mean(np.sum(self.Charges[frame1:frame2+1, c_atoms], axis=1))
            c_spn = np.mean(np.sum(self.Spins[frame1:frame2+1, c_atoms], axis=1))
            # The molecule must have a large enough charge/spin of the correct sign to neutralize the original atoms
            if np.abs(c_chg) > tol/2 and c_chg * chg < 0:
                c_xyz = np.array(self.atom_select(c_atoms)[frames].xyzs)
                # Get the squared distance matrix for every charged molecule with opposite sign
                sq_dmat = np.zeros((xyz.shape[1], c_xyz.shape[1], len(frames)))
                for a in range(xyz.shape[1]):
                    for b in range(c_xyz.shape[1]):
                        sq_dmat[a, b, :] = np.sum((xyz[:, a, :] - c_xyz[:, b, :])**2, axis=1)
                # Build a closest contact time series
                c_contact = np.array(np.min(sq_dmat, axis=(0, 1)))**0.5
                # Record the closest that the molecule got, and how far it drifted away
                Candidates[molID] = Candidate(min(c_contact), max(c_contact), c_chg, c_spn)

        # Now loop over all candidates in order of increasing closest contact distance
        molID_sorted_maxdist = [list(Candidates.keys())[i] for i in np.argsort([c.maxdx for c in list(Candidates.values())])]
        keep_molID = []
        success = False
        curr_chg = chg
        curr_spn = spn
        limit = 3
        for molID in molID_sorted_maxdist:
            valid = False
            formula = self.TimeSeries[molID]['graph'].ef()
            c_atoms = self.TimeSeries[molID]['graph'].L()
            keep_atoms = list(itertools.chain(*[self.TimeSeries[m]['graph'].L() for m in keep_molID]))
            new_atoms = sorted(c_atoms + keep_atoms + list(atoms))
            new_chg = np.mean(np.sum(self.Charges[frame1:frame2+1, new_atoms], axis=1))
            new_spn = np.mean(np.sum(self.Spins[frame1:frame2+1, new_atoms], axis=1))
            # Check for charge and spin consistency.
            nprot = sum([Elements.index(i) for i in [self.elem[j] for j in new_atoms]])
            nelec = int(nprot + round(new_chg))
            nspin = int(round(new_spn))
            # The number of electrons should be odd iff the spin is odd.
            if nelec%2 != nspin%2:
                print(nprot, nelec, nspin)
                if self.printlvl >= 2: print("\x1b[91mInconsistent\x1b[0m charge/spin (charge %+.3f -> %+.3f, spin %+.3f -> %+.3f) ; not adding %s" % (curr_chg, new_chg, curr_spn, new_spn, formula))
            elif new_chg * chg < 0 and abs(new_chg) > tol:
                if self.printlvl >= 2: print("\x1b[91mOvershot\x1b[0m the reaction (charge %+.3f -> %+.3f) ; not adding molID %s" % (curr_chg, new_chg, formula))
            else:
                if self.printlvl >= 2: print("\x1b[91mReducing net charge of\x1b[0m reaction (charge %+.3f -> %+.3f) ; adding molID %s" % (curr_chg, new_chg, formula))
                curr_chg = new_chg
                curr_spn = new_spn
                valid = True
                keep_molID.append(molID)
            if valid and abs(new_chg) < tol:
                success = True
                break
            if len(keep_molID) == limit:
                if self.printlvl >= 2: print("\x1b[92mFailed\x1b[0m after adding %i molecules" % len(keep_molID))
                break
        if self.printlvl >= 1:
            print("Neutralization %s: charge \x1b[94m%+.3f\x1b[0m -> \x1b[92m%+.3f\x1b[0m, spin %+.3f -> %+.3f" % ('success' if success else 'failure', chg, curr_chg, spn, curr_spn), end=' ')
            if keep_molID: print("added %s (molIDs %s)" % (formulaSum([self.TimeSeries[m]['graph'].ef() for m in keep_molID]), ' '.join(keep_molID)))
            else: print()
        return keep_molID, success

    def writeColors(self):
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
        print(header, file=self.moviefile)
        for a in range(self.na):
            if a > 0:
                print("mol addrep 0", file=self.moviefile)
            print("mol modselect %i 0 index %i" % (a, a), file=self.moviefile)
            print("mol modstyle %i 0 VDW 0.50 27.0" % (a), file=self.moviefile)
            print("mol modmaterial %i 0 Transparent" % (a), file=self.moviefile)

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
        
        print(extra, file=self.moviefile)
        renderf = 0
        for f in range(0, self.ns):
            ColorByAtom = self.traj_color[f]
            # ColorByAtom = [self.ColorIdx[j] for j in self.IsoLabels[renderf]]
            print("animate goto %i" % f, file=self.moviefile)
            print("display resetview", file=self.moviefile)
            print("display height 4.0", file=self.moviefile)
            print("rotate y by %.1f" % ((0.1 * renderf) % 360), file=self.moviefile)
            for an, color in enumerate(ColorByAtom):
                if an + 1 != ColorByAtom.size:
                    print(color, end = " ", file=self.colortab)
                else:
                    print(color,end = "",  file=self.colortab)
                if ColorNow[an] != color:
                    ColorNow[an] = color
                    print("mol modcolor %i 0 ColorID %i" % (an, color), file=self.moviefile)
                    if color == 8:
                        print("mol modmaterial %i 0 Ghost" % an, file=self.moviefile)
                    else:
                        print("mol modmaterial %i 0 Transparent" % an, file=self.moviefile)
            print(file=self.colortab)
            if self.Render:
                print("render snapshot frame%04i.tga" % renderf, file=self.moviefile)
            renderf += 1
        
    def Output(self):
        # Print final data to file.
        self.moviefile = open('/dev/null','w')
        self.colortab = open('color.dat','w')
        self.writeColors()
        self.writeReactionEvents()
        # self.GetReactions()
        with open('bonds.dat','w') as bondtab: bondtab.write('\n'.join(self.BondLists)+'\n')
        self.WriteChargeSpinLabels()
        if hasattr(self, 'boxes'): self.write('whole.xyz')

