# input parameters generation module for nanoreactor
# Author Laszlo R Seress - seress@stanford.edu

# Version History: 

# 1.0 - 12/14/2015
# Initial creation
# 2.0 - 1/5/2016
# r2, k2, min_inner_radius, inner_radius, and outer_radius

from __future__ import print_function
import csv
import math


def mdk2listgenerator(mdk2_count_input):
	mdk2feed = [3.0, 2.5, 3.5, 2.0, 4.0, 1.5, 4.5, 1.0, 5.0, 0.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
	mdk2list = mdk2feed[:mdk2_count_input]
	mdk2list = sorted(mdk2list)
   	return mdk2list

def mdr2listgenerator(mdr2_count_input,inner_radius,min_inner_radius):
    mdr2list = [inner_radius]
    if mdr2_count_input > 1:
    	for index in range(2,mdr2_count_input+1):
    		mdr2list.append(inner_radius + inner_radius*((-1)**index) * mdr2_factor * math.floor(index/2.) )
    		if (inner_radius + inner_radius*((-1)**index) * mdr2_factor * math.floor(index/2.)) < min_inner_radius:
    			print("Error: a generated r2 corresponds to a sphere with a smaller volume than the sum of the volumes of the elements.")
    			sys.exit(50)
    mdr2list = sorted(mdr2list)
    return mdr2list

def findminradius(coordinate_filename):
    # read in atomic radii (ar_csv = atomic radii, comma separated values)

    #initialize list
    atomic_radii = list()
    #read in values from file, add to array
    with open('atomicradii.csv','rb') as ar_csv:
        ar = csv.DictReader(ar_csv)
        index = 0
        for row in ar:
            atomic_radii.append(list([row['element'],int(row['atomicnumber']),int(row['radius_pm'])]))
            index +=1
    ar_csv.close()

    # variable to keep sum of atomic volumes
    volume_sum_pm = 0  # in pm^3
    coors = open(coordinate_filename,'r')
    coors_lines = coors.readlines()
    # get number of elements from first line of .xyz file
    num_elements = int(coors_lines[0])
    # loop through file to get elements, check if one or two letter element name
    for index in range(2,2+num_elements):
    	coors_lines[index] = coors_lines[index].lstrip()
    	element = coors_lines[index][0]
    	if coors_lines[index][1] != ' ':
    		element += coors_lines[index][1] 

    	# search through list of elements for a match in atomic radii, add to sum if match
    	for index in range(0,85):
    		if atomic_radii[index][0] == element:
    			volume_sum_pm += (atomic_radii[index][2])**3
    coors.close()

    # convert sum to angstroms 
    volume_sum_a = float(volume_sum_pm)/(10**6)
    min_inner_radius = float((volume_sum_a) ** (1./3))
    # note that the 4/3 pi factor was omitted at both finding the radius of the atoms and then of the total sphere, so it cancels
    return min_inner_radius

def findinnerradius(min_radius, inner_volume_ratio):
    inner_radius = float(min_radius* (inner_volume_ratio ** (1./3)))
    return inner_radius

def findouterradius(inner_radius, outer_volume_ratio):
    outer_radius = float(inner_radius * (outer_volume_ratio ** (1./3)) )
    return outer_radius
