#!/usr/bin/env python
"""
PyPackmol is a Python wrapper for packmol program that provide simple packing of molecules. 
Note that packmol needs to be installed separately.  
Molecules can be given in terms of input files (xyz, etc), or SMILES strings.  
Open Babel is required to do the conversion if the input format is not xyz.
"""

__title__     = "pypackmol"
__author__    = "Xiaolei Zhu"
__copyright__ = "Martinez Group, Stanford University, CA, USA, Planet Earth"
__maintainer__= "Xiaolei Zhu"
__email__     = "virtualzx@gmail.com"
__version__   = "0.3.0"
__status__    = "Prototype"

import tempfile
import os
import subprocess
import numpy as np
import time

# If openbabel is imported, it is possible to convert different file types,
# construct molecules from SMILES and/or perform force field optimizations.
# Note that the default is to use force field optimization for SMILES but 
# keep the geometries for file input.
try :
    import pybel as pb
    _has_openbabel=True
except :
    _has_openbabel=False

    
    
#Exception class for packing failure
class FailToPack(Exception):
    pass



def convert_to_xyz(input_data,input_format="auto",force_field="uff") :
    """Convert input format unrecognized by packmol into xyz files.
    
    If the input molecular structure data is in a xyz file, the function 
    simply transfer the data to a temp file. Otherwise, the function use 
    OpenBabel to convert into xyz format, then save this to a temp file.
    
    Args:
        input_data: Filename or SMILES string to be converted
        input_format: Format of the input.  When the value is "auto" which is 
            the default, the function will try to determine the type of input 
            file. In such event, if the input string does not correspond to an 
            existing file that can be opened, the string is instead taken to 
            be a SMILES string. 
        force_field: The force-field used by OpenBabel during constructions
            of 3D structure if only 2D structures are available (i.e., SMILES)
    
    Returns:
        A named temporary file containing the molecular structure in xyz format.  
        Note that since this is a temp file, it will automatically be deleted 
        when the file object is unreferenced or destroyed.
    """
    input_format=input_format.lower()
    if input_format == "auto" :
        try :
            with open(input_data) as input_file :
                input_format = input_data.split(".")[-1]
        except :
            input_format ="smi"
            
    # Create an xyz temp file and perform conversion
    output_file = tempfile.NamedTemporaryFile(mode="w+",suffix="xyz")            
    # If the file is xyz, there is no need to convert
    if input_format == "xyz" :
        with open(input_data) as in_file :
            output_file.write(in_file.read())
    else :
        # Conversion to xyz is needed. 
        # If do not have openbabel, then we are pretty screwed here.
        if not _has_openbabel :
            raise ImportWarning["Cannot import OpenBabel for format conversion. " +
                       "Please use xyz format instead."]
        # Construct a openbabel `pybel.Molecule` object then export to xyz
        if input_format=="smi" or imput_format=="smiles" :
            mol=pb.readstring(input_format,input_data)
            mol.make3D(forcefield=force_field,steps=200)
        else :
            mol=pb.readfile(input_format,input_data).next()
        output_file.write(mol.write("xyz"))
    
    output_file.flush()
    output_file.seek(0)
    return output_file



class Packmol (object):
    """Wrapper class for the packmol molecule packer program
    """
    
    # this is a list of options used by Packmol class but not the packmol program
    NONPACKMOL_OPTIONS=["command_line", "region_type","dimension","opt_force_field","seed"]
    
    
    
    def __init__(self, **kwargs ):
        """Construct packmol run by providing simulation parameters.
        
        There is no arguments, but this constructor accepts a keyword list which contains
        packmol related options, including the following:
            command_line: Packmol command line.  Default is ``packmol``
            dimension: Radius of the sphere.  Default is 10
            tolerance: Minimum distance between atoms in different molecules.  Default is 3
            output: Filename where the packed geometries are saved. Default is "packed.xyz"
            seed: Seed for the random number generator.  When set to None, which is the 
                default, current system timer will be used.
            nloop: Maximum number of packmol optimization steps.  Default is 100.
            opt_force_field: Force field used to construct 3D structure from SMILES strings.
            
        Any keyword not recognized by the module will still be dumped into packmol input file.
        """
        self._structure_list = []
        # default options
        self._options = {"command_line":"packmol", "region_type":"sphere","seed":None,
                                "dimension":10, "tolerance":3.0, "output":"packed.xyz",
                                "nloop":100, "opt_force_field":"uff"}       
        self.set_options(**kwargs)
        
        
        
        
    def set_options(self, **kwargs ):
        """Update packmol option values.  
        
        Use keyword list to change the value of packmol options."""
        for key,value in kwargs.items():
            self._options[key] = value

            
            

    def _prepare_input_file (self) :
        """Create temp file to be used as packmol input file
        
        This method is for internal use only.  Content of the file can be accessed from 
        instance variable ``_input_stash`` after the method.
        
        Returns:
            A named temporary file that contains options as well as structure definitions 
            formated as packmol input."""
        inp_file = tempfile.NamedTemporaryFile(mode="w+",suffix="inp")
        inp_file.write("filetype xyz\n")
        
        ranseed=self._options["seed"]
        if not ranseed : ranseed=time.clock()
        inp_file.write("seed {}\n".format(hash(ranseed)%10**9))
        
        # Write down other arguments 
        for key,value in self._options.items():
            if key not in Packmol.NONPACKMOL_OPTIONS:
                inp_file.write("{0} {1}\n".format(key,value))
        # 
        for mol_type in self._structure_list :
            inp_file.write("structure {}\n".format(mol_type["structure"].name))
            inp_file.write("  number {}\n".format(mol_type["number"]))
            if self._options["region_type"] == "sphere" :
                inp_file.write(
                    "  inside sphere 0.0 0.0 0.0 {}\n".format(self._options["dimension"]))
            elif self._options["region_type"] :
                raise NotImplementedError("Currently only sphere regions are supported.")
            for key,value in mol_type["options"].items():
                inp_file.write("  {0} {1}\n".format(key,value))
            inp_file.write("end structure\n")
        inp_file.flush()
        inp_file.seek(0)
        self._input_stash=inp_file.read()
        inp_file.seek(0)
        return inp_file
    
    
    def clear(self):
        """Clean the molecule table."""
        self._structure_list=[]
        
    
    def add_structure(self, input_data, count=1, input_format="auto", **kwargs) :
        """Add one or more structure group to be packed by packmol.
        
        Args:
            input_data: molecular structure of the new group(s).  A 
                string that contains filename that contains this 
                structure, or a SMILES string.  
            input_format: Format of `input_data`.  Can be any OpenBabel recognized file
                format or `smi` for a SMILES.  When set to `auto`, which is the default,
                the format will be determined from filename, or set to SMILES if it does
                not correspond to any existing file.
            count: Number of this structure to be packed.  Default is 1.
        Any other unrecognized keyword input will be dumped into the structure block of 
        packmol input file.
        """
        self._structure_list.append({ 
                "structure": convert_to_xyz(input_data,input_format,
                                            self._options["opt_force_field"]) ,
                "number"   : count ,
                "options"  : kwargs
            })

        
        
    def pack(self, output=None) :
        """Perform packmol runs.  
        
        Construct input file then creates a subprocess to run packmol.  
        
        Args:
            output: Specify the filename of the packed molecule geometry file.  
            
        Return is a dict that contains the following fields:
            "filename" :  Filename of the packed geoemtry file.
            "packmol output" : Print out from packmol run.
        """
        if output :
            self._options["output"] = output
        outfile = tempfile.NamedTemporaryFile(mode="w+")
        code = subprocess.call(self._options["command_line"], stdin=self._prepare_input_file(),
                        stdout=outfile )
        outfile.seek(0)
        self._output_stash = outfile.read()
        if code :
            raise Exception("Packmol error termination. exit code={}".format(code))
        if "ERROR" in self._output_stash :
            raise Exception("Individual molecules cannot be staged in packing region. " + 
                            "Increase tolerance value.")
        if "STOP" in self._output_stash :
            raise FailToPack(("Packing failed.  Cannot fulfill tolerance constraints. " +
                           "Best result found in [{}]").format(self._options["output"]))
        self._last_result={"filename":self._options["output"],"packmol output":self._output_stash}
        if _has_openbabel :
            self._packed = pb.readfile("xyz",self._options["output"]).next()
            ff=pb._forcefields[self._options['opt_force_field']]
            ff.Setup(self._packed.OBMol)
            self._last_result["energy"]=ff.Energy()
        return self._last_result

    
    def last_result(self):
        """Get the result from the last packing run"""
        return self._last_result
    
    
    def autosize(self, min_radius,max_radius=20,increment=0.2,report=True) :
        """Automatically determine the dimension of the sphere.  
        
        The method repeatedly runs packmol and increase the radius slightly every time until 
        the system can be successful packed.
        
        Args:
            min_radius:  Initial radius to start the test from.
            increment: The amount of radius to be increased every time packmol fails.
            
        Returns:
            The minimum radius that the system can be packed."""
        for r in np.arange(min_radius,max_radius,increment):
            self.set_options(dimension=r)
            if report : print "\rTesting packing with radius {}".format(r),
            try :
                result=self.pack()
                #packing successful!
                if report : print "\rPacking successful with radius {}".format(r)
                return r
            except FailToPack:
                pass

        
# When called as standalone, perform a doctest run
if __name__ == "__main__" :
    import doctest
    doctest.testmod()
