# Nanoreactor package
#### Authors: Lee-Ping Wang, Heejune Park, Robert McGibbon, Leah Isseroff Bendavid, Alexey Titov, Todd J. Martinez

This is a script and library to analyze reactive MD (nanoreactor)
simulations.  It provides two main capabilities:

### 1) Reaction extraction and visualization.  

To analyze a simulation trajectory, run "LearnReactions.py traj.xyz".
Use the "-h" argument to get help.  It will generate reaction_123.xyz
files that contain your chemical reactions, as well as bonds.dat,
color.dat, charge.dat and spin.dat used to highlight your reactive MD
trajectory.

To view the highlighted trajectory, you need to install VMD and
preferably be using a 3D-accelerated machine.  Make sure reactions.vmd
is in the same folder and run using: 'vmd -e reactions.vmd -args
coors.xyz'

### 2) Energy refinement of extracted reactions.

To convert a nanoreactor "reaction event" into a minimum energy path,
run "Refine.py reaction.xyz".  It will start a workflow which (1)
optimizes the geometries of subsampled frames, (2) find frame pairs
that contain chemically distinct species and construct a pathway
connecting the energy basins, (3) smooth the pathway in internal
coordinates, and (4) perform string method + transition state + 
intrinsic reaction coordinate calculations to locate the minimum
energy path. Finally it will summarize the reaction as a PDF at
the end.

#### Installation:

To install the package, run "python setup.py install".  Make sure all
dependencies are installed (below) - it's more challenging than
installing the package itself. :)

An example for LearnReactions.py is included.  Examples for Refine.py
are forthcoming.

#### Dependencies (there are many, though I tried to keep the # down): 

- Python 2.7, 3.5 or above, and the Python packages numpy, scipy and networkx.  The
sklearn package is recommended but not required.  These dependencies
are satisfied if you get the Python distribution from Enthought Canopy
or (I think) Anaconda.  Note that future versions of sklearn (0.17) 
plan to deprecate the Hidden Markov Model so this dependency may need
to be updated in the future.

- The Cooperative Computing Tools (http://ccl.cse.nd.edu/software/) is
recommended for rolling out a highly parallel energy refinement
calculation of many pathways.  Using the Work Queue library, hundreds
to thousands of Q-Chem calculations may be run in parallel across any
combination of available computing resources.

- Q-Chem 4.2 is required for running the energy refinement calculations.
MPI and OpenMP parallelism are both required, because the stability
analysis only works with MPI and the other components work with OpenMP.
IMPORTANT NOTE: The commercial version of Q-Chem is somewhat problematic
for the intrinsic reaction coordinate part of the calculation, because
the user has no way to set the IRC initial direction.  I have modified
the source code to make sure the rpath_direction rem variable works as
intended; the Q-Chem binary on Fire has this change, and I submitted
the updated source code to the Q-Chem repository.

The following is required for drawing the summary PDF at the end of 
energy refinement:

- Gnuplot for drawing the energy diagram.  Make sure the SVG terminal is supported.

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
on different machines.  This is part of the GNOME project and comes
as a Ubuntu package; building from souce might be painful.

Enjoy!

- Lee-Ping

License: This software uses the BSD license (except for nebterpolator
which uses GNU GPL 3.0).  See LICENSE and nebterpolator/LICENSE.
