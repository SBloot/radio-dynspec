import os
import astropy.io.fits as pf
import sys
import astropy.wcs as wcs
import numpy as np
import matplotlib.pyplot as plt

'''
This code will take a calibrated measurement set, subtract all sources that are not your target, and then phaseshift the data to the exact location of the source.
The file is set up to run on Stokes I, which only works if there is a Stokes I detection. If you want to do Stokes V instead, skip step 2.
If your target is not detected in images, skip ahead to step 3 and enter the coordinates manually.
'''


field='name_of_target_field'
ms='/path/to/calibrated/data.ms'
output_dir='/path/to/output/directory/'
ms_datacolumn='corrected' # If your calibrated data is in the corrected data column
#ms_datacolumn='data' # If your calibrated data is in the data column

'''
Step 1: find the exact location of the source. 
The location is found by making a high-resolution image with very small pixels around the coordinates you specify, 
to find the coordinates of the center of the source.
'''

#Approximate location in degrees.
ra_guess=12.098
dec_guess=-47.7634

#Write the file that casa will execute to split the data and make the image.
#This will make a Stokes I image, but can be changed to Stokes V by changing stokes='I' to stokes='V'.
new_file="output_dir='"+output_dir+"'"+os.linesep+"ms='"+ms+"'"+os.linesep+ "tclean(vis=ms, imagename=output_dir+'I_center', imsize=1024, cell='0.05arcsec',interactive=False, stokes='I', niter=0, field='"+field+"', phasecenter='J2000 "+str(ra_guess)+"deg "+str(dec_guess)+"deg')"+os.linesep+"exportfits(output_dir+'I_center.image', output_dir+'I_center.fits')"+os.linesep+"split(ms, output_dir+'split.ms', datacolumn='corrected', field='"+field+"')"


fout = open('casa_part_1.py', "w")
fout.write(new_file)
fout.close()

execfile_string="execfile('casa_part_1.py')"

#Run that file in casa
os.system("casa -c \""+execfile_string+"\"")


# Step 2: Subtract all sources in the field except the target.

# Make a large image of the entire field. The center is masked out, so your target should not be cleaned if it is near the phase center. If it is not, you will need to create a new mask that only excludes your source.
# This command also writes a model of the cleaned sources to the model column.
# The cell size and image size can be changed to work well with the data. In that case, a new mask is required.
os.system("wsclean -size 9600 9600 -weight briggs 0.0 -weighting-rank-filter 3 -mgain 0.8 -fit-spectral-pol 3 -fits-mask mask.fits -kernel-size 7 -data-column DATA -pol i -name "+output_dir+"/I_large -deconvolution-channels 12 -scale 1.5arcsec -niter 10000 -join-channels -threshold 0.0001 -channels-out 12 "+output_dir+"split.ms")


#Write the file casa will execute to subtract the model column (where the sources are saved) from the datacolumn.
new_file="uvsub('"+output_dir+"'+'split.ms')"+os.linesep
fout = open('casa_part_2.py', "w")
fout.write(new_file)
fout.close()
execfile_string="execfile('casa_part_2.py')"

os.system("casa -c \""+execfile_string+"\"")


#Step 3: Phase-shift to the location of the target.

#Get the coordinates of the target from the image you made in step 1.
h=pf.getheader(output_dir+'I_center.fits')
data=pf.getdata(output_dir+'I_center.fits')
w = wcs.WCS(h)
pixel=np.where(np.abs(data)==np.max(np.abs(data)))
ra=w.pixel_to_world(pixel[3],pixel[2],pixel[1],pixel[0])[0].ra.deg
print(ra)
dec=w.pixel_to_world(pixel[3],pixel[2],pixel[1],pixel[0])[0].dec.deg
print(dec)


#If you want to enter the coordinates manually, use the lines below. The unit should be degrees.
#ra=[(12.0]
#dec=[(-45.9)]

#Write the file that will execute the phase-shift step
file = open('phaseshift.parset', "r")
new_file=''
for line in file:
	if line.startswith('phaseshift.phasecenter'):
		new_line='phaseshift.phasecenter=['+str(ra[0])+'deg, '+str(dec[0])+'deg]'+os.linesep
	elif line.startswith('msin='):
		#new_line='msin='+output_dir+'split.ms'+os.linesep
		new_line='msin='+output_dir+'split.ms'+os.linesep
	elif line.startswith('msin.data'):
		new_line='msin.datacolumn=CORRECTED_DATA'+os.linesep # Change to DATA if you skipped the subtraction step.
	elif line.startswith('msout='):
		new_line='msout='+output_dir+'split_out.ms'+os.linesep
	else:
		new_line=line
	new_file+=new_line
file.close()
# opening the file in write mode
fout = open('phaseshift.parset', "w")
fout.write(new_file)
fout.close()


#Execute the phase-shift step
os.system("DP3 phaseshift.parset")

