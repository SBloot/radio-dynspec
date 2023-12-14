import os
import astropy.io.fits as pf
import sys
import astropy.wcs as wcs
import numpy as np
import matplotlib.pyplot as plt
import casacore.tables as pt

'''
This code takes the split and subtracted measurement set from dynspec.py and makes the dynamic spectra.
'''

output_dir='/path/to/split/ms/directory/'
plot_dir='/path/to/plot/location'

freq_low=1.0 #Lower end of the frequency band in GHz
freq_high=3.0 #Higher end of the frequency band in GHz


msname=output_dir+'split_out.ms'
t = pt.table(msname,readonly=True)



# Read in the antenna column (describes the antenna positions)
a21 = t.getcol("ANTENNA2") - t.getcol("ANTENNA1")
amin,amax = np.amin(a21),np.amax(a21)


nant = np.amax(a21)+1
if amin==0:             # Check if the dataset includes autocorrelations
	nbase = int(nant*(nant-1)/2+nant)
else:                   # or not
	nbase = int(nant*(nant-1)/2)



# Read in the data, the flags (whether a given data point is flagged or not), 
# the baseline lengths in 3 dimensions, the weights (how reliable a datapoint it),
# and the time column (the timestamp for each datapoint)
# Note that the number of time integrations here counts every baseline as a separate time integration as well, 
# so 1 timestep has (number of baselines) data points.
# This code assumes you have only 1 spectral window in the ms. If you have more, it will put each spectral window in a separate row.
# For each time integration, you would have the datapoints per baseline for each spectral window.

d = t.getcol("DATA")                            # Data. The shape is (number of time integrations * number of channels * number of polarizations)
f = t.getcol("FLAG")                            # Flags. The shape is (number of time integrations * number of channels * number of polarizations)
weights=t.getcol("WEIGHT_SPECTRUM")				# Weights. The shape is (number of time integrations * number of channels * number of polarizations)
#If the WEIGHT_SPECTRUM column does not exist, there is usually a WEIGHT column that is the same thing, except the shape is (number of time integrations * number of polarizations)
#It assumes the same weight for each channel

time=t.getcol("TIME")							# Timestamps of the datapoints. The shape is (number of time integrations). The unit is seconds of Julian date
timestep=time[1]-time[0]
uvw = t.getcol("UVW")
blen = (np.sum(uvw**2,axis=1))**0.5             # Baseline lengths in meters

d[f]=np.nan 									# Applying the flags to the data

indices=range(nbase)						    # If you want to ignore a certain baseline, this is where you can do that, by taking it out of the indices.

max_time=d.shape[0]/nbase 						# The number of time integrations in the dataset, assuming there is only 1 spectral window.
nchan=d.shape[1]								# The number of channels in the dataset.
chan_width=(freq_high-freq_low)/nchan

d_bl_av=np.zeros([int(max_time),nchan,4], dtype=complex) # Setting up the array to store the data averaged over baselines.

for i in range(int(max_time)):
	d_part=d[i*nbase:(i+1)*nbase][indices]
	weights_part=weights[i*nbase:(i+1)*nbase][indices]
	d_bl_av[i,:,:]=np.nanmean(d[i*nbase:(i+1)*nbase][indices]*weights_part/np.mean(weights_part[np.isnan(d_part)==False]), axis=0)



# Calculate (in order) Stokes V flux density, imaginary component of Stokes V, Stokes I, and the imaginary component of Stokes I, in units of mJy.

vraw_bl_av = 1e3*np.real(-1j*d_bl_av[:,:,1]+1j*d_bl_av[:,:,2])/2
imag_v = 1e3*np.imag(-1j*d_bl_av[:,:,1]+1j*d_bl_av[:,:,2])/2
iraw_bl_av= 1e3*np.real(d_bl_av[:,:,0]+d_bl_av[:,:,3])/2
imag_i=1e3*np.imag(d_bl_av[:,:,0]+d_bl_av[:,:,3])/2


#First, create a dynamic spectrum at low resolution to determine the limits on the color bars for all plots.

bin_size=(15*60/(timestep)) # The number of time integrations in 1 bin, currently set to 15 minutes
nbins=int(max_time/bin_size) # The number of bins in time
nfreqbins=42 # The number of frequency bins
freqbin=nchan/nfreqbins #The size of the frequency bins

#Define the arrays for the averaged data
v_binned=np.zeros([nbins,nfreqbins])
imag_binned=np.zeros([nbins,nfreqbins])
i_binned=np.zeros([nbins,nfreqbins])
i_imag_binned=np.zeros([nbins,nfreqbins])

#For each bin, calculate the average flux in Stokes I and Stokes V, and the standard deviation for the imaginary components.
for i in range(nbins):
	for k in range(nfreqbins): 
		v_binned[i,k]=np.nanmean(vraw_bl_av[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])
		imag_binned[i,k]=np.nanstd(imag_v[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum(np.where(np.isnan(imag_v[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])==False)))		
		i_binned[i,k]=np.nanmean(iraw_bl_av[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])
		i_imag_binned[i,k]=np.nanstd(imag_i[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum(np.where(np.isnan(imag_i[int(i*bin_size):int((i+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])==False)))


#Find the maximum signal to noise in Stokes V
sigma=np.abs(v_binned/imag_binned)
sigma_real=sigma[np.isfinite(sigma)]
max_sig=np.max(sigma_real)

#Find the flux density in Stokes V that goes with the maximum signal-to-noise point
mask=np.where(sigma==max_sig)
flux_max=np.abs(v_binned[mask])

#Define the colorbar limit
lim=np.ceil(flux_max)


# Same process for Stokes I
sigma=(i_binned/i_imag_binned)
sigma_real=sigma[np.isfinite(sigma)]
max_sig=np.max(sigma_real)

mask=np.where(sigma==max_sig)
flux_max=i_binned[mask]
lim_i=np.ceil(flux_max)


#Define the different resolutions for which you want to create the plots. 
bin_size_list=np.array([30*60/timestep,15*60/timestep, 3*60/timestep,1*60/timestep,40/timestep,20/timestep ])
nbins_list=np.array(max_time/bin_size_list, dtype=int)
nfreqbins_list=np.array([21,42,128,256,512,1024])

# Sometimes, the SNR becomes unreasonably high due to one bin being heavily contaminated with RFI.
# To limit the effect of that, we define an upper limit on the SNR colorbar, defined separately for the different bins.
lim_sigma_mins=[50,40,30,20,10,5]

# We flag bins where the noise is too high. This list determines when we flag a bin.
factor=[1.0, 1.0, 0.5, 0.5, 0.5, 0.5]

for i in range(len(nbins_list)): # This loops over the bin sizes.
	print(i)

	nbins=nbins_list[i]
	bin_size=bin_size_list[i]
	nfreqbins=nfreqbins_list[i]

	#Define the arrays to store the averages and standard deviations.
	v_binned=np.zeros([nbins,nfreqbins])
	imag_binned=np.zeros([nbins,nfreqbins])
	i_binned=np.zeros([nbins,nfreqbins])
	i_imag_binned=np.zeros([nbins,nfreqbins])
	freqbin=nchan/nfreqbins

	#Calculate the spectra and the uncertainties.
	v_spec=np.array([np.nanmean(vraw_bl_av[:, int(k*freqbin):int((k+1)*freqbin)]) for k in range(nfreqbins)])
	v_imag_spec=np.array([np.nanstd(imag_v[:, int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum((np.isnan(imag_v[:, int(k*freqbin):int((k+1)*freqbin)])==False))) for k in range(nfreqbins)])
	i_spec=np.array([np.nanmean(iraw_bl_av[:, int(k*freqbin):int((k+1)*freqbin)]) for k in range(nfreqbins)])
	i_imag_spec=np.array([np.nanstd(imag_i[:, int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum((np.isnan(imag_i[:, int(k*freqbin):int((k+1)*freqbin)])==False))) for k in range(nfreqbins)])

	#Calculate the light curves and uncertainties.
	v_lightcurve=np.array([np.nanmean(vraw_bl_av[int(k*bin_size):int((k+1)*bin_size), :]) for k in range(nbins)])
	v_imag_lightcurve=np.array([np.nanstd(imag_v[int(k*bin_size):int((k+1)*bin_size), :])/np.sqrt(np.sum((np.isnan(imag_v[int(k*bin_size):int((k+1)*bin_size),:])==False))) for k in range(nbins)])
	i_lightcurve=np.array([np.nanmean(iraw_bl_av[int(k*bin_size):int((k+1)*bin_size), :]) for k in range(nbins)])
	i_imag_lightcurve=np.array([np.nanstd(imag_i[int(k*bin_size):int((k+1)*bin_size), :])/np.sqrt(np.sum((np.isnan(imag_i[int(k*bin_size):int((k+1)*bin_size),:])==False))) for k in range(nbins)])


	# For each bin, find the averages and standard deviations
	for j in range(nbins):
		for k in range(nfreqbins): 
			v_binned[j,k]=np.nanmean(vraw_bl_av[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])
			imag_binned[j,k]=np.nanstd(imag_v[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum((np.isnan(imag_v[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])==False)))		
			i_binned[j,k]=np.nanmean(iraw_bl_av[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])
			i_imag_binned[j,k]=np.nanstd(imag_i[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])/np.sqrt(np.sum((np.isnan(imag_i[int(j*bin_size):int((j+1)*bin_size), int(k*freqbin):int((k+1)*freqbin)])==False)))


	# Flag the bins where the uncertainty is too high
	v_binned[imag_binned>(np.nanmedian(imag_binned)+factor[i]*np.nanstd(imag_binned))]=np.nan
	i_binned[i_imag_binned>(np.nanmedian(i_imag_binned)+factor[i]*np.nanstd(i_imag_binned))]=np.nan

	# If 65% of the row/column is flagged, flag the entire row/column.
	for l in range(len(v_binned)):
		if np.sum(np.isnan(v_binned[l]))/len(v_binned)>0.65:
			v_binned[l]=np.nan
			v_lightcurve[l]=np.nan
	for l in range(len(v_binned[0])):
		if np.sum(np.isnan(v_binned[:,l]))/len(v_binned[0])>0.65:
			v_binned[:,l]=np.nan
			v_spec[l]=np.nan
	for l in range(len(i_binned)):
		if np.sum(np.isnan(i_binned[l]))/len(i_binned)>0.65:
			i_binned[l]=np.nan
			i_lightcurve[l]=np.nan
	for l in range(len(i_binned[0])):
		if np.sum(np.isnan(i_binned[:,l]))/len(i_binned[0])>0.65:
			i_binned[:,l]=np.nan
			i_spec[l]=np.nan
	

	# Plot 1: Stokes V flux density dynamic spectrum.
	fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
	img=axes[1,0].imshow(v_binned.T, aspect='auto',interpolation='none', cmap='RdBu_r', vmin=-1*lim, vmax=lim,extent=[0,max_time*timestep/60,freq_low,freq_high], origin='upper')
	axes[1,0].set_xlabel('Time (min)')
	axes[1,0].set_ylabel('Frequency (GHz)')
	axes[0,0].plot(np.linspace(0+(bin_size/2)*timestep/60,nbins*bin_size*timestep/60-(bin_size/2)*timestep/60, nbins) ,v_lightcurve)
	axes[1,1].plot(np.flip(v_spec),np.linspace(freq_low+freqbin*chan_width/2,freq_high-freqbin*chan_width/2, nfreqbins))
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.colorbar(img, label='Flux density (mJy)',ax=axes.ravel().tolist())
	fig.delaxes(axes[0,1])
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_V.pdf')
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_V.png', dpi=1000)
	plt.close()

	# Plot 2: Stokes V signal-to-noise dynamic spectrum.

	# Define the limit for the SNR plot. This can be strongly biased by 1 point, which is why there is an upper limit to how high the limit can be.
	lim_sigma = np.min([np.nanmax(np.abs(v_binned/imag_binned)[np.isfinite(np.abs(v_binned/imag_binned))]),lim_sigma_mins[i]])
	fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
	img=axes[1,0].imshow(v_binned.T/imag_binned.T, aspect='auto',interpolation='none', cmap='RdBu_r', vmin=-lim_sigma, vmax=lim_sigma,extent=[0,max_time*timestep/60,freq_low,freq_high])

	axes[1,0].set_xlabel('Time (min)')
	axes[1,0].set_ylabel('Frequency (GHz)')
	axes[0,0].plot(np.linspace(0+(bin_size/2)*timestep/60,nbins*bin_size*timestep/60-(bin_size/2)*timestep/60, nbins) ,v_lightcurve/v_imag_lightcurve)
	axes[1,1].plot(np.flip(v_spec/v_imag_spec),np.linspace(freq_low+freqbin*chan_width/2,freq_high-freqbin*chan_width/2, nfreqbins))

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.colorbar(img, label='Flux density ($\sigma$)',ax=axes.ravel().tolist())
	fig.delaxes(axes[0,1])
	#plt.tight_layout()

	plt.savefig(plot_dir+str(nfreqbins)+'_bins_V_snr.pdf')
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_V_snr.png')
	plt.close()
	

	#Plot 3: Stokes I flux density dynamic spectrum
	fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
	img=axes[1,0].imshow(i_binned.T, aspect='auto',interpolation='none', cmap='RdBu_r', vmin=-1*lim_i, vmax=lim_i,extent=[0,max_time*timestep/60,freq_low,freq_high])

	axes[1,0].set_xlabel('Time (min)')
	axes[1,0].set_ylabel('Frequency (GHz)')
	axes[0,0].plot(np.linspace(0+(bin_size/2)*timestep/60,nbins*bin_size*timestep/60-(bin_size/2)*timestep/60, nbins) ,i_lightcurve)
	axes[1,1].plot(np.flip(i_spec),np.linspace(freq_low+freqbin*chan_width/2,freq_high-freqbin*chan_width/2, nfreqbins))

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.colorbar(img, label='Flux density (mJy)',ax=axes.ravel().tolist())
	fig.delaxes(axes[0,1])
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_I.png')
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_I.pdf')
	plt.close()

	#Plot 4: Stokes I signal-to-noise dynamic spectrum
	lim_sigma = np.min([np.nanmax(np.abs(i_binned/i_imag_binned)[np.isfinite(np.abs(i_binned/i_imag_binned))]),lim_sigma_mins[i]])

	fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
	img=axes[1,0].imshow(i_binned.T/i_imag_binned.T, aspect='auto',interpolation='none', cmap='RdBu_r', vmin=-lim_sigma, vmax=lim_sigma,extent=[0,max_time*timestep/60,freq_low,freq_high])

	axes[1,0].set_xlabel('Time (min)')
	axes[1,0].set_ylabel('Frequency (GHz)')
	axes[0,0].plot(np.linspace(0+(bin_size/2)*timestep/60,nbins*bin_size*timestep/60-(bin_size/2)*timestep/60, nbins) ,i_lightcurve/i_imag_lightcurve)
	axes[1,1].plot(np.flip(i_spec/i_imag_spec),np.linspace(freq_low+freqbin*chan_width/2,freq_high-freqbin*chan_width/2, nfreqbins))

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.colorbar(img, label='Flux density ($\sigma$)',ax=axes.ravel().tolist())
	fig.delaxes(axes[0,1])

	plt.savefig(plot_dir+str(nfreqbins)+'_bins_I_snr.pdf')
	plt.savefig(plot_dir+str(nfreqbins)+'_bins_I_snr.png')
	plt.close()




	