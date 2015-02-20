#!/bin/bash

# This code creates and uses /tmp/leeping/local/bin/python to install
# the nanoreactor package, because it's faster this way on the compute
# nodes.

# Create a temporary folder on the head node
mkdir -p /tmp/leeping
if [ -e /tmp/leeping ] ; then
    rsync -a --delete /u/sciteam/leeping/local /tmp/leeping
    rm -rf build
    /tmp/leeping/local/bin/python setup.py install
    rsync -a --delete /tmp/leeping/local/ /u/sciteam/leeping/local 
fi
