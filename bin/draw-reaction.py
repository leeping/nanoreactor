#!/usr/bin/env python

"""
draw-reaction.py

Draw a summary of an individual reaction pathway.  Intended to be run
after a completed freezing string or transition state calculation in
the same folder as the calculation results.

This script requires a number of files to exist on disk:
1) Intrinsic reaction coordinate file (irc.xyz)
2) Coordinate file containing Mulliken charges and 
spins on the first and second columns (irc.pop)
3) Two-column file containing arc length vs. energy (irc.nrg)

Additionally, a Mayer bond order matrix stored as plaintext in
'irc_reactant.bnd' and 'irc_product.bnd' or Q-Chem output files
in 'irc_reactant.out' and 'irc_product.out' are helpful.  It will
use the matrix to determine the bond order instead of just using 
the distance.

The output is a SVG and PDF file - 'reaction.svg' and 'reaction.pdf'.

Dependencies (there are many):

- Gnuplot for drawing the energy diagram.

- lxml (Python interface to libxml2) for parsing XML files. Comes with
  Anaconda and Enthought Canopy. If installing from scratch, you need to
  install libxml2 and libxslt first.

- Openbabel (version from GitHub newer than September 26, 2014).  
  Used for generating SVG images of molecules.  As of September 26, 2014, 
  the latest release contained bugs so please check out the code from GitHub 
  (https://github.com/openbabel/openbabel) and build from source.  Make sure
  to build with Python bindings enabled.  Building Openbabel requires Eigen
  which is a pain, but it comes as a Ubuntu package.

- rsvg-convert for converting SVG to PDF.  SVG is a nice file format
  for editing vector graphics, but it may look different when viewed
  on different machines.
"""

from collections import Counter, defaultdict
import pybel
import numpy as np
import os, sys, shutil
import argparse
from lxml import etree as ET
from nanoreactor.chemistry import BondStrengthByLength
from nanoreactor.molecule import Molecule, MolEqual
from nanoreactor.nifty import extract_tar, _exec
from nanoreactor.output import logger
parser = ET.XMLParser(remove_blank_text=True)

def parse_user_input():
    # Parse user input - run at the beginning.
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store_true", help='Overwrite existing pictures')
    parser.add_argument('--xyz', type=str, default='irc.xyz', help='Reaction pathway coordinate file')
    parser.add_argument('--pop', type=str, default='irc.pop', help='Reaction pathway coordinate file')
    parser.add_argument('--energy', type=str, default='irc.nrg', help='Reaction pathway energy file')
    args, sys.argv= parser.parse_known_args(sys.argv[1:])
    return args

def load_bondorder(R, P):
    """
    Extracts QM bond order information from a calculation.  This
    actually reads the Mayer bond order matrix from a Q-Chem output
    file.  This function looks funny because I implemented it after
    some calculations already started running. :P Therefore it checks
    to see if the ".bnd" file exists, and if not, it creates one.

    Intended to be run in a directory with transition-state.tar.bz2
    obviously.
    
    Parameters
    ----------
    R, P: Molecule
        Molecule objects for the reactant and product.
    """
    for bz2 in ['transition-state.tar.bz2', 'freezing-string.tar.bz2']:
        extract_tar(bz2, ['irc_reactant.out', 'irc_product.out', 'irc_reactant.bnd', 'irc_product.bnd'])
    if not os.path.exists('irc_reactant.bnd'):
        if os.path.exists('irc_reactant.out'):
            M = Molecule('irc_reactant.out')
            if hasattr(M, 'qm_bondorder'):
                bo_reactant = Molecule('irc_reactant.out').qm_bondorder
                np.savetxt('irc_reactant.bnd', bo_reactant, fmt="%8.3f")
            else:
                logger.info("Warning: No QM bond order for reactant")
                bo_reactant = np.zeros((R.na, R.na))
        else:
            logger.info("Warning: No QM bond order for reactant")
            bo_reactant = np.zeros((R.na, R.na))
    else:
        bo_reactant = np.loadtxt('irc_reactant.bnd')
    if not os.path.exists('irc_product.bnd'):
        if os.path.exists('irc_product.out'):
            M = Molecule('irc_product.out')
            if hasattr(M, 'qm_bondorder'):
                bo_product = Molecule('irc_product.out').qm_bondorder
                np.savetxt('irc_product.bnd', bo_product, fmt="%8.3f")
            else:
                logger.info("Warning: No QM bond order for product")
                bo_product = np.zeros((R.na, R.na))
        else:
            logger.info("Warning: No QM bond order for product")
            bo_product = np.zeros((P.na, P.na))
    else:
        bo_product = np.loadtxt('irc_product.bnd')
    R.qm_bondorder = bo_reactant
    P.qm_bondorder = bo_product

unicode_subscripts = {'0': 8320, '1': 8321, '2': 8322,
                      '3': 8323, '4': 8324, '5': 8325,
                      '6': 8326, '7': 8327, '8': 8328,
                      '9': 8329}

unicode_superscripts = {'0': 8304, '1': 185,  '2': 178,
                      '3': 179,  '4': 8308, '5': 8309,
                      '6': 8310, '7': 8311, '8': 8312,
                      '9': 8313, '+': 8314 , '-': 8315}

subscript_entities = dict([(i, "&#%i;" % j) for i, j in unicode_subscripts.items()])
superscript_entities = dict([(i, "&#%i;" % j) for i, j in unicode_superscripts.items()])

def svg_subs(string):
    """ Given a molecular formula like C2H5, replace the numbers with the subscript entities. """
    strout = ''
    for i in string:
        strout += subscript_entities.get(i, i)
    return strout

def svg_sups(string):
    """ Given something like 2-, replace with superscript entities. """
    strout = ''
    for i in string:
        strout += superscript_entities.get(i, i)
    return strout

def subscripts(string):
    """
    Returns some unicode integers for writing subscripts.  I was
    using this before when I used ImageMagick to create the images, no
    longer used.
    """
    ustr = unicode(string)
    for i in unicode_subscripts:
        ustr = ustr.replace(i, unichr(unicode_subscripts[i]))
    return ustr

def superscripts(string):
    """ 
    Returns some unicode integers for writing superscripts.  I was
    using this before when I used ImageMagick to create the images, no
    longer used.
    """
    ustr = unicode(string)
    for i in unicode_superscripts:
        ustr = ustr.replace(i, unichr(unicode_superscripts[i]))
    return ustr

# Print a narrow image strip with fixed-width text.
svgtext = """<svg width="850" height="50" viewBox="0 0 2000 100"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="50" y="75" font-family="Consolas" font-size="36" fill="black" >
    {text}
  </text>
</svg>
"""

# Print a narrow image strip with text and a colored box.
svgrect = """<svg width="850" height="50" viewBox="0 0 2000 100"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <rect width="2000" height="100" style="fill:rgb({frgb});stroke-width:10;stroke:rgb({srgb})" />
  <text text-anchor="middle" x="1000" y="75" font-family="Adobe Garamond Pro" font-size="60" fill="black" >
    {text}
  </text>
</svg>
"""

# SVG arrow stolen from Wikipedia.
svgarrow="""<svg
   xmlns="http://www.w3.org/2000/svg"
   width="175"
   height="50"
   version="1.0"
   >
  <defs
     id="defs1977" />
  <path
     id="path15"
     d="M 20,25 L 156,25"
     style="fill:none;fill-rule:nonzero;stroke:black;stroke-width:2px;stroke-linecap:round;stroke-linejoin:round" />
  <path
     id="path17"
     d="M 125,17 L 156,25 L 125,33 L 130,25 L 125,17 z "
     style="fill:black;fill-rule:nonzero;stroke:none;stroke-linecap:round;stroke-linejoin:round" />
  <path
     id="path19"
     d="M 125,17 L 156,25 L 125,33 L 130,25 L 125,17"
     style="fill:none;fill-rule:nonzero;stroke:black;stroke-width:2px;stroke-linecap:round;stroke-linejoin:round" />
</svg>"""

# Empty SVG file used for composing the image.
# Note that the dimensions are specified here.
svgbase="""<svg
   xmlns="http://www.w3.org/2000/svg"
   width="850"
   height="600"
   version="1.0"
   >
</svg>"""

def round_array(arr):
    """ 
    Round off values in an array, ensuring 
    that the sum of values is preserved. 
    """
    # Add a small random noise to the working array.  This avoids the
    # issue of never converging because we have two values that are
    # nearly exactly the same, and raising / lowering the working
    # array causes them to BOTH be rounded.
    arr1 = arr.copy()
    while np.min(np.abs(arr1[:, np.newaxis] - arr1 + np.eye(len(arr1)))) < 2e-4:
        arr1 += np.random.random(len(arr1))*1e-3
    # Initial rounded values
    rounded = np.array([round(i) for i in arr1])
    # Iterate over this loop until sum of rounded values 
    # equals sum of the input array.
    while abs(sum(rounded) - round(sum(arr))) > 0.01:
        # If the sum of rounded values is too large, then lower the
        # working array so the future rounded values are smaller.
        if sum(rounded) > round(sum(arr)):
            arr1 -= 1e-4
        else:
            arr1 += 1e-4
        rounded = np.round(arr1)
    return rounded.astype(int)

def fix_svg(fsvg):
    """ 
    Apply simple in-place fixes to a svg file.  Hydrogens and carbons
    are darkened and we use Arial font.  Also ensure that the SMILES
    title fits in the box.
    """
    svg = ET.parse(fsvg, parser)
    for elem in svg.findall('.//{http://www.w3.org/2000/svg}text'):
        if elem.text == 'H':
            elem.attrib['fill'] = "rgb(128, 128, 128)"
            elem.attrib['stroke'] = "rgb(128, 128, 128)"
        elif elem.text == 'C':
            elem.attrib['fill'] = "rgb(0, 0, 0)"
            elem.attrib['stroke'] = "rgb(0, 0, 0)"
        elem.attrib['font-family'] = "Arial"
        elem.attrib['stroke-width'] = "0.0"
        elem.attrib['y'] = '%.6f' % (float(elem.attrib['y'])-3)
        if 'text-anchor' in elem.attrib:
            elem.attrib['font-size'] = '%.1f' % min(6.0, float(6.0 * 25 / len(elem.text)))
    svg.write(fsvg, pretty_print=True)

def make_obmol(M, prefix):
    """ 
    Create an OpenBabel molecule object from our molecule object and
    create an SVG image with the molecules and their SMILES string.

    Parameters
    ----------
    M : Molecule
        Molecule object containing a structure and QM Mulliken charges / spins
    prefix : str
        Base name of created images
    """
    if len(M) != 1:
        logger.error("I can only deal with length-1 Molecule objects")
        raise RuntimeError
    M.write('.coords.xyz')
    # Collapse hydrogen charges down into heavy atoms.
    nh = defaultdict(int)
    for i in range(M.na):
        if M.elem[i] != 'H': continue
        for bi, bj in M.bonds:
            if i == bi:
                M.qm_mulliken_charges[0][bj] += M.qm_mulliken_charges[0][bi]
                M.qm_mulliken_charges[0][bi] = 0.0
                nh[bj] += 1
            elif i == bj:
                M.qm_mulliken_charges[0][bi] += M.qm_mulliken_charges[0][bj]
                M.qm_mulliken_charges[0][bj] = 0.0
                nh[bi] += 1
    # Net charge and spin multiplicity
    net_charge = int(round(sum(M.qm_mulliken_charges[0])))
    net_mult = int(abs(round(sum(M.qm_mulliken_spins[0])))+1)
    # The rounded total charge on each molecule.
    round_mol_chg = [int(round(sum(M.qm_mulliken_charges[0][m.L()]))) for m in M.molecules]
    # The rounded spin multiplicity on each molecule (making sure the sum is preserved.)
    fchg = round_array(M.qm_mulliken_charges[0])
    spn = round_array(M.qm_mulliken_spins[0])
    sgn = lambda x: 1  if x>=0 else -1
    # Create the Pybel Molecule object (which contains an OBMol object)
    pbm = pybel.readfile("xyz", ".coords.xyz").next()
    # Clear all coordinates so we'll force OpenBabel to redraw them
    for i, a in enumerate(pbm.atoms):
        a.OBAtom.ClearCoordPtr()
        a.OBAtom.SetVector(0, 0, 0)
    obm = pbm.OBMol
    obm.SetAutomaticFormalCharge(False)
    obm.SetAutomaticPartialCharge(False)
    obm.SetTotalCharge(net_charge)
    obm.SetTotalSpinMultiplicity(net_mult)
    # Set formal charge on atoms.
    # Bad-Ass Molecular Formulas. :D
    mofos = []
    for im, m in enumerate(M.molecules):
        if round_mol_chg[im] != 0:
            # Within a molecule, this is the atom that has the largest
            # partial charge with the same sign as the molecular charge.
            # It will be the recipient of the formal charge.
            # LPW: This is not an ideal solution because it does not allow 
            # us to put formal charges onto zwitterions.  But we would need
            # a good definition for what is zwitterionic - for example a plain
            # C=O bond is not a zwitterion but simply using the Mulliken populations
            # would classify it as such.
            fatom = m.L()[np.argmax([M.qm_mulliken_charges[0][i] * sgn(round_mol_chg[im]) for i in m.L()])]
            pbm.atoms[fatom].OBAtom.SetFormalCharge(round_mol_chg[im])
        # Build a string for the net charge on the molecule
        q = int(round_mol_chg[im])
        if q > 0: sup = "+"
        elif q < 0: sup = "-"
        else: sup = ""
        if abs(q) > 1: sup = "%i%s" % (abs(q), sup)
        # Create a molecular formula string like C_2H_5^2+ with 
        # correct superscripts and subscripts when SVG is rendered 
        mofos.append(svg_subs(m.ef())+svg_sups(sup))
    # Set spin multiplicity on atoms.
    for i in range(M.na):
        # Determining formal charges simply from Mulliken charge doesn't work so well.
        # Mainly, it will turn things like CO bonds into zwitterions. :P
        # pbm.atoms[i].OBAtom.SetFormalCharge(fchg[i])
        mult = abs(spn[i])+1
        pbm.atoms[i].OBAtom.SetSpinMultiplicity({1:0, 2:2, 3:1}[mult])
    # Set bonds.
    qmbo = M.qm_bondorder
    for b in M.bonds:
        # The determine bond order by length and put it in.
        bol = BondStrengthByLength(M.elem[b[0]], M.elem[b[1]], np.linalg.norm(M.xyzs[0][b[0]] - M.xyzs[0][b[1]]), artol=0.33)[1]
        boq = max(1, (qmbo[b[0],b[1]] if qmbo[b[0], b[1]] != 0.0 else bol))
        # "5" stands for an aromatic bond.
        if bol == 1.5 and round(boq*2)/2 == 1.5: 
            bo = 5
        else:
            bo = int(round(boq))
        obm.AddBond(int(b[0]+1), int(b[1]+1), bo)
    # Removes any bonds generated by openbabel that are not in our molecule.
    for a in range(M.na):
        for b in range(a+1,M.na):
            if (a,b) not in M.bonds:
                obb = obm.GetBond(a+1,b+1)
                if obb != None:
                    obm.DeleteBond(obb)

    # Set title to canonical SMILES.
    pbm.title = ''
    pbm.title = pbm.write("can", opt={"h":True}).strip()
    # Write images.
    svgout = "%s.svg" % prefix
    pbm.write("svg", svgout, opt={"P":300, "a":True}, overwrite=True)
    fix_svg(svgout)
    # Return a string containing this side of the chemical equation.
    return pbm.title, ' + '.join(['%s%s' % (str(j) if j>1 else '', i) for i, j in Counter(mofos).items()])

def compose(fwd=False):
    """ Compose the final image by putting the individual SVG images together. """
    # First read the base SVG file.
    svgrxn = ET.parse("base.svg", parser)
    svgroot = svgrxn.getroot()
    # Pretty simple, append the folder text, it's in the top left.
    svgtitle = ET.parse("folder_text.svg", parser).getroot()
    svgroot.append(svgtitle)
    # Append the reactant image and move it downward by 50.
    svgreact = ET.parse("reactant.svg" if fwd else "product.svg", parser).getroot()
    svgreact.attrib["y"] = "50"
    # Remove the white colored background.
    for elem in svgreact:
        if 'rect' in elem.tag:
            svgreact.remove(elem)
    # Do the same for the product image.  
    # It's positioned 300 from the right edge and 50 down.
    svgprod = ET.parse("product.svg" if fwd else "reactant.svg", parser).getroot()
    svgprod.attrib["x"] = "550"
    svgprod.attrib["y"] = "50"
    for elem in svgprod:
        if 'rect' in elem.tag:
            svgprod.remove(elem)
    # Append reactant and product images.  Now we're in business!
    svgroot.append(svgreact)
    svgroot.append(svgprod)
    # The energy diagram is between the reactant and product images
    # and shifted slightly below.
    svggraph = ET.parse("irc_energy.svg", parser).getroot()
    svggraph.attrib["x"] = "300"
    svggraph.attrib["y"] = "225"
    svgroot.append(svggraph)
    # The reaction arrow is placed so it looks to be in the middle.
    svgarr = ET.parse("arrow.svg", parser).getroot()
    svgarr.attrib["x"] = "350"
    svgarr.attrib["y"] = "150"
    svgroot.append(svgarr)
    # The colored strips are placed at the bottom.
    svgstrip1 = ET.parse("reaction_text.svg", parser).getroot()
    svgstrip1.attrib["y"] = "450"
    svgroot.append(svgstrip1)
    svgstrip2 = ET.parse("chargemult_text.svg", parser).getroot()
    svgstrip2.attrib["y"] = "500"
    svgroot.append(svgstrip2)
    svgstrip3 = ET.parse("energy_text.svg", parser).getroot()
    svgstrip3.attrib["y"] = "550"
    svgroot.append(svgstrip3)
    # Make sure the elements all contain newline characters.
    for element in svgroot.iter():
        element.tail = '\n'
    # Write SVG image to disk and convert to PDF.
    # rsvg is part of GNOME and it has many dependencies, but it's easily installed on Ubuntu.
    # One drawback is that certain fonts get their subscripts all messed up (notably Arial).
    # This may be a problem with the Arial font rather than the rsvg package...
    svgrxn.write("reaction.svg", pretty_print=True)
    os.system("rsvg-convert -f pdf -o reaction.pdf reaction.svg")
    # Delete temporary files.
    for i in ["base", "folder_text", "reactant", "product", "irc_energy", "arrow", "reaction_text", "chargemult_text", "energy_text"]:
        os.remove("%s.svg" % i)

def main():
    # Parse user input.
    args = parse_user_input()
    # Load IRC coordinates.
    M = Molecule(args.xyz)
    M.load_popxyz(args.pop)
    # Rebuild bonds for the reactant (0th frame).
    R = M[0]
    R.build_topology()
    # Rebuild bonds for the product (last frame).
    P = M[-1]
    P.build_topology()
    # Don't draw reaction if the reactant and product are the same.
    if MolEqual(R, P):
        logger.info("Exiting because reactant and product molecules are the same")
        sys.exit()
    # Load Mayer bond order matrices.
    load_bondorder(R, P)
    # Create SVG drawings of the reactant and product.
    canr, strr = make_obmol(R, "reactant")
    canp, strp = make_obmol(P, "product")
    # IRC energy.
    ArcE = np.loadtxt(args.energy)
    E = ArcE[:, 1]
    # "Canonicalize" the reaction direction.
    fwd = True
    if max([len(m.L()) for m in R.molecules]) > max([len(m.L()) for m in P.molecules]):
        # Reactions should go in the direction of increasing molecule size.
        fwd = False
    elif (max([len(m.L()) for m in R.molecules]) == max([len(m.L()) for m in P.molecules])) and (E[0] < E[-1]):
        # If molecules on both sides are the same size, go in the exothermic direction.
        fwd = False
    if fwd: 
        shutil.copy2("irc.nrg", "plot.nrg")
        with open("reaction.can", "w") as f: print >> f, "%s>>%s" % (canr, canp)
    else:
        ArcE = ArcE[::-1]
        ArcE[:, 0] *= -1
        ArcE[:, 0] -= ArcE[:, 0][0]
        ArcE[:, 1] -= ArcE[:, 1][0]
        np.savetxt("plot.nrg", ArcE, fmt="% 14.6f", header="Arclength(Ang) Energy(kcal/mol)")
        strr, strp = strp, strr
        E = E[::-1]
        with open("reaction.can", "w") as f: print >> f, "%s>>%s" % (canp, canr)
    # This string looks something like C2H2 + H2O -> C2H4O.  
    # The funny character is a Unicode entity for the "right arrow"
    strrxn = strr + ' &#10230; ' + strp
    # Create the components of the final image.
    # First write a text box with the chemical equation.
    with open("reaction_text.svg", "w") as f: print >> f, svgrect.format(text=strrxn, frgb="255,210,80", srgb="255,165,128")
    # Next write a text box with the charge and multiplicity.
    net_charge = int(round(sum(M.qm_mulliken_charges[0])))
    net_mult = int(abs(round(sum(M.qm_mulliken_spins[0])))+1)
    with open("chargemult_text.svg", "w") as f: print >> f, svgrect.format(text="Charge = %i ; Multiplicity = %i" % (net_charge, net_mult), frgb="194,225,132", srgb="154,205,50")
    # Write a text box with the reaction energy and barrier height.
    with open("energy_text.svg", "w") as f: print >> f, svgrect.format(text="&#916;E = %.2f kcal ; E&#8336; = %.2f kcal" % (E[-1]-E[0], np.max(E)), frgb="142,200,255", srgb="30,144,255")
    # Run script to generate energy diagram plot (uses Gnuplot).
    _exec("plot-rc.sh", print_command=False)
    # Write a text heading with the location of the calculation on disk.
    with open("folder_text.svg", "w") as f: print >> f, svgtext.format(text="Path: %s" % os.getcwd().split(os.environ['HOME'])[-1].split("Refinement")[-1].strip('/'))
    # Print some skeleton SVG files (actually part of this script).
    with open("arrow.svg", "w") as f: print >> f, svgarrow
    with open("base.svg", "w") as f: print >> f, svgbase
    # Finally, compose the image.
    compose(fwd)

if __name__ == "__main__":
    main()

sys.exit()
