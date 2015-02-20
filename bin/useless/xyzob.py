#!/usr/bin/env python

import openbabel as ob
from xml.etree import ElementTree as ET
from nanoreactor import Molecule
from nanoreactor.chemistry import BondStrengthByLength
from nanoreactor.nifty import _exec
from nanoreactor.nifty import *
from collections import OrderedDict
from math import floor
import random
import numpy as np
import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', action="store_true", help='Overwrite existing pictures')
args, sys.argv= parser.parse_known_args(sys.argv)

# Exit if the reaction.png has already been written.
# Thus, to refresh pictures we must delete them first.
if os.path.exists('reaction.png') and not args.f:
    sys.exit()

rfn = sys.argv[1]
pfn = sys.argv[2]
chgfile = np.loadtxt(sys.argv[3])
spnfile = np.loadtxt(sys.argv[4])
delta = float(sys.argv[5])
barrier = float(sys.argv[6])
asterisk = 0
if len(sys.argv) > 7:
    asterisk = int(sys.argv[7])

rchg = chgfile[0]
rspn = spnfile[0]
pchg = chgfile[-1]
pspn = spnfile[-1]

# From OBabel documentation:
# This can be regarded in the present context as a measure of the hydrogen deficiency of an atom. Its value is:
#     0 for normal atoms,
#     2 for radical (missing one hydrogen) and
#     1 or 3 for carbenes and nitrenes (missing two hydrogens).

spinmap = {1:0, 2:2, 3:1}

def subscripts(string):
    unicode_integers = {'0': 8320, '1': 8321, '2': 8322,
                        '3': 8323, '4': 8324, '5': 8325,
                        '6': 8326, '7': 8327, '8': 8328,
                        '9': 8329}
    ustr = unicode(string)
    for i in unicode_integers:
        ustr = ustr.replace(i, unichr(unicode_integers[i]))
    return ustr

def round_array(arr):
    arr1 = arr.copy()
    # Obtain sorting by roundoff errors.
    rounded = [round(i) for i in arr1]
    errs = [round(i)-i for i in arr1]
    while abs(sum(rounded) - round(sum(arr))) > 0.01:
        if sum(rounded) > round(sum(arr)):
            arr1 -= 1e-4
        else:
            arr1 += 1e-4
        rounded = [round(i + 0.01*(random.random()*2-1)) for i in arr1]
    return rounded

def draw(fn, chg=None, spn=None):
    M = Molecule(fn)
    if chg == None: chg = np.zeros(M.na)
    if spn == None: spn = np.zeros(M.na)
    obm = ob.OBMol()
    obcsvg = ob.OBConversion()
    obcsvg.SetInAndOutFormats("xyz","svg")
    obcsvg.ReadFile(obm, fn)
    # obcpng = ob.OBConversion()
    # obcpng.SetInAndOutFormats("xyz","png")
    # obcpng.ReadFile(obm, fn)
    nh = np.zeros(M.na)
    for b in M.bonds:
        # First delete all OpenBabel bonds.
        obb = obm.GetBond(int(b[0]+1), int(b[1]+1))
        if obb != None:
            obm.DeleteBond(obb)
        # The determine bond order by hand and put it in.
        bo = BondStrengthByLength(M.elem[b[0]], M.elem[b[1]], np.linalg.norm(M.xyzs[0][b[0]] - M.xyzs[0][b[1]]), artol=0.33)[1]
        if bo == 1.5: bo = 5
        if sorted([M.elem[b[0]], M.elem[b[1]]]) == ['C', 'O'] and bo == 3: 
            if len(M.topology.neighbors(b[0])) > 1 or len(M.topology.neighbors(b[1])) > 1:
                bo = 2
        obm.AddBond(int(b[0]+1), int(b[1]+1), bo)
        # import IPython
        # IPython.embed()
        # obm.GetBond(int(b[0]+1), int(b[1]+1)).SetBondOrder(bo)
        if M.elem[b[0]] == 'H' and M.elem[b[1]] != 'H':
            chg[b[1]] += chg[b[0]]
            chg[b[0]] = 0
            spn[b[1]] += spn[b[0]]
            spn[b[0]] = 0
            nh[b[1]] += 1
        if M.elem[b[1]] == 'H' and M.elem[b[0]] != 'H':
            chg[b[0]] += chg[b[1]]
            chg[b[1]] = 0
            spn[b[0]] += spn[b[1]]
            spn[b[1]] = 0
            nh[b[0]] += 1
        if M.elem[b[1]] == 'H' and M.elem[b[0]] == 'H':
            nh[b[0]] += 1
            nh[b[1]] += 1

    round_chg = round_array(chg)
    round_spn = round_array(spn)
    # Formal charges ONLY when there's a net charge.
    if sum(np.array(round_chg)) == 0: 
        round_chg = np.zeros(len(round_chg))
    round_spn = np.zeros(len(round_spn))
    # printcool_dictionary(OrderedDict([("%i " % (i+1) + M.elem[i], "% .3f %i" % (chg[i], round_chg[i])) for i in range(M.na)]))
    # print len(chg), len(round_chg)
    # print sum(chg), sum(round_chg)
    # print [(M.elem[i], "% .3f %i" % (chg[i], round_chg[i])) for i in range(M.na)]
    # print M.na
    # print chg, round_chg
    # print spn, round_spn

    # Create a canonical SMILES molecule.
    obccan = ob.OBConversion()
    obccan.SetInAndOutFormats("xyz","can")
    obccan.AddOption("h",obccan.OUTOPTIONS) # Explicit hydrogens
    obccan.AddOption("n",obccan.OUTOPTIONS) # No molecule name in output.
    ocan = os.path.splitext(fn)[0]+".can"
    obccan.WriteFile(obm, ocan)
    can = open(ocan).readlines()[0].strip()
    
    for i in range(M.na):
        obm.GetAtom(i+1).SetFormalCharge(int(round_chg[i]))
        obm.GetAtom(i+1).SetSpinMultiplicity(spinmap[int(abs(round_spn[i]))+1])

    # Create a chemical markup molecule.
    obccml = ob.OBConversion()
    obccml.SetInAndOutFormats("xyz","cml")
    obccml.AddOption("p",obccml.OUTOPTIONS)
    ocml = os.path.splitext(fn)[0]+".cml"
    obccml.WriteFile(obm, ocml)
    tree = ET.parse(ocml)
    root = tree.getroot()
    root.set('id', can)
    for child in root:
        if child.tag == 'atomArray':
            for i, atom in enumerate(child):
                if M.elem[i] != 'H':# and 'spinMultiplicity' not in atom.attrib:
                    atom.set('hydrogenCount', "%i" % nh[i])
                    # atom.set('spinMultiplicity', "%i" % spinmap[int(abs(round_spn[i]))+1])
    with open(os.path.splitext(fn)[0]+".cml",'w') as f: tree.write(f)

    efs = ', '.join([m.ef() for m in M.molecules])
    efs = subscripts(efs)

    osvg = os.path.splitext(fn)[0]+".svg"
    obcsvg.AddOption("a",obcsvg.OUTOPTIONS)
    obcsvg.AddOption("d",obcsvg.OUTOPTIONS)
    obcsvg.AddOption("u",obcsvg.OUTOPTIONS)
    obcsvg.AddOption("i",obcsvg.OUTOPTIONS)
    obcsvg.AddOption("P",obcsvg.OUTOPTIONS,"600")
    obcsvg.WriteFile(obm, osvg)
    opng = os.path.splitext(fn)[0]+".png"
    os.system("convert %s %s" % (osvg, opng))

    # PNG writing does not work :(
    # opng = os.path.splitext(fn)[0]+".png"
    # obcpng.AddOption("a",obcpng.OUTOPTIONS)
    # obcpng.AddOption("w",obcpng.OUTOPTIONS,"1200")
    # obcpng.AddOption("h",obcpng.OUTOPTIONS,"600")
    # obcpng.WriteFile(obm, opng)

    return efs

efs0 = draw(rfn, rchg, rspn)
efs1 = draw(pfn, pchg, pspn)

rfbase = os.path.splitext(rfn)[0]
pfbase = os.path.splitext(pfn)[0]

dnm = os.path.split(rfn)[0]
if asterisk:
    strfnm = os.path.join(dnm, "string_ev.xyz")
else:
    strfnm = os.path.join(dnm, "string.irc_ev.xyz")

# print strfnm
# print strfnm
os.system("sh plot-rc.sh %s" % strfnm)

# This appears to use the OpenBabel executable to generate SVGs directly.
# os.system("obabel %s.cml %s.cml -h -O .tmp.svg -xd -xa --align" % (rfbase, pfbase))
# os.system("obabel %s.cml %s.cml -h -O .tmp.svg -xd -xa" % (rfbase, pfbase))
# os.system("inkscape -z -e .tmp.png -w 1200 .tmp.svg")

# This appears to convert the SVGs in draw() to PNG.
# os.system("convert +append %s %s .tmp.png" % (os.path.splitext(rfn)[0]+".png", os.path.splitext(pfn)[0]+".png"))

# Try to use OpenBabel's PNG plugin.
subprocess.call("obabel %s.cml %s.cml -O .tmp.png --gen2d -xa -xw 1200 -xh 600 &> /dev/null" % (rfbase, pfbase), shell=True)

# Draw the reaction arrowhead.
arrow_head="l -30,-10  +10,+10  -10,+10  +30,-10 z"
width, height = (int(i) for i in os.popen("identify .tmp.png  | awk '{print $3}' | awk -F 'x' '{print $1, $2}'").readlines()[0].split())
xcen, ycen = width/2, height/2
os.system("convert -roll +0-60 .tmp.png -draw \'line %i,%i %i,%i\' -gravity Center -draw \"stroke blue fill skyblue path \'M %i,%i  %s\' \" .tmp1.png" % (xcen-50, ycen-60, xcen+40, ycen-60, xcen+40, ycen-60, arrow_head))

cmd="composite rc.png .tmp1.png -gravity south .tmp11.png"
os.system(cmd)

cmd=u"convert .tmp11.png -gravity South -background Orange -font Adobe-Garamond-Pro-Regular -pointsize 30 -splice 0x36 -annotate +0+3 \'%s ---> %s\' .tmp2.png" % (efs0, efs1)

os.system(cmd.encode('utf-8'))

chg = int(round(sum(rchg)))
mult = int(round(np.abs(sum(rspn)))) + 1

os.system("convert .tmp2.png -gravity South -background YellowGreen -font Adobe-Garamond-Pro-Regular -pointsize 30 -splice 0x36 -annotate +0+3 \'%s\' .tmp3.png" % ("Charge = %i ; Multiplicity = %i" % (chg, mult)))
cmd=u"convert .tmp3.png -gravity South -background DodgerBlue -font Adobe-Garamond-Pro-Regular -pointsize 30 -splice 0x36 -annotate +0+3 \'%sE = %.2f kcal ; Barrier = %.2f kcal%s\' reaction.png" % (unichr(916), delta, barrier, "*" if asterisk else "")
os.system(cmd.encode('utf-8'))

# for b in R.bonds:
#     if obmR.GetBond(int(b[0]+1), int(b[1]+1)) == None:
#         obmR.AddBond(int(b[0]+1), int(b[1]+1), 1)

# for b in P.bonds:
#     if obmP.GetBond(int(b[0]+1), int(b[1]+1)) == None:
#         obmP.AddBond(int(b[0]+1), int(b[1]+1), 1)

# obcsmi = ob.OBConversion()
# obcsmi.SetInAndOutFormats("xyz","smi")

# osmi = os.path.splitext(sys.argv[1])[0]+".smi"
# obcsmi.AddOption("h",obcsmi.OUTOPTIONS) # Explicit hydrogens
# obcsmi.AddOption("n",obcsmi.OUTOPTIONS) # No molecule name in output.
# obcsmi.WriteFile(obm, osmi)

# smi = open(osmi).readlines()[0].strip()

# obm.SetTitle("%s Chg=%i Mult=%i" % (smi, chg, mult))
# osvg = os.path.splitext(sys.argv[1])[0]+".svg"
# obcsvg.AddOption("a",obcsvg.OUTOPTIONS)
# obcsvg.AddOption("c",obcsvg.OUTOPTIONS,"2")
# obcsvg.WriteFile(obm, "reaction.svg")

