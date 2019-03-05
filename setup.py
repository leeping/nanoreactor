"""
setup.py: Install nanoreactor learning script.  
"""
VERSION=4.2
__author__ = "Lee-Ping Wang, Alexey Titov, Robert McGibbon"
__version__ = "%.1f"%VERSION

import os, sys
from distutils.core import setup,Extension
import numpy
import glob

requirements = ['numpy', 'networkx']

# Declare the C extension modules
CONTACT = Extension('nanoreactor/_contact_wrap',
                    sources = ["src/contact/contact.c",
                               "src/contact/contact_wrap.c"],
                    extra_compile_args=["-std=c99","-O3","-shared",
                                        "-fopenmp", "-Wall"],
                    extra_link_args=['-lgomp'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

DIHEDRAL = Extension('nanoreactor/_dihedral_wrap',
                    sources = ["src/dihedral/dihedral.c",
                               "src/dihedral/dihedral_wrap.c"],
                    extra_compile_args=["-std=c99","-O3","-shared",
                                        "-fopenmp", "-Wall"],
                    extra_link_args=['-lgomp'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "nanoreactor"
    setupKeywords["version"]           = "%.1f" %VERSION
    setupKeywords["author"]            = __author__
    setupKeywords["author_email"]      = "leeping@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["packages"]          = ["nanoreactor", "nebterpolator", "nebterpolator.io", "nebterpolator.core"]
    setupKeywords["package_dir"]       = {"nanoreactor": "src"}
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh") + glob.glob("bin/*.exe")
    setupKeywords["ext_modules"]       = [CONTACT, DIHEDRAL]
    setupKeywords["py_modules"]       = ["pypackmol"]
    setupKeywords["platforms"]         = ["Linux", "Mac OS X", "Windows"]
    setupKeywords["description"]       = "Machine learning for reactive MD."
    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    print "%s" % outputString
    return setupKeywords
    

def main():
    setup_keywords = buildKeywordDictionary()
    setup(**setup_keywords)
    for requirement in requirements:
      try:
          exec('import %s' % requirement)
      except ImportError as e:
          print >> sys.stderr, '\nWarning: Could not import %s' % e
          print >> sys.stderr, 'Warning: Some package functionality may not work'

if __name__ == '__main__':
    main()

