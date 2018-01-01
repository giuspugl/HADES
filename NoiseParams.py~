import numpy as np
from hades.params import BICEP
a=BICEP()

def create_params(map_size=a.map_size,sep=a.sep):
	""" Create parameter space + map_id grid for batch processing."""
	# Load in good map IDs;
	goodMaps=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	
	# Create meshed list of all parameters
	ID,NP,FW=np.meshgrid(goodMaps,a.noi_par_NoisePower,a.noi_par_FWHM)
	
	# Save output
	np.savez(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(map_size,sep),map_id=ID.ravel(),noise_power=NP.ravel(),FWHM=FW.ravel())
	
	# returns length of parameter string
	return len(ID.ravel()) 

if __name__=='__main__':
	""" Code to create data for some noise parameters defined in param file to plot output epsilon against """
	import time
	start_time=time.time()
	import sys
	import os
	index=int(sys.argv[1])
	
	# First load in parameters
	from hades.NoiseParams import create_params
	LEN=create_params()
	
	if index<LEN:
		delensing_fraction=0.1
	elif (index>=LEN) and (index<2*LEN):
		 delensing_fraction=1.
		 index-=LEN
	else:
		print 'at end of data'
		sys.exit()
		 
	paramFile=np.load(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(a.map_size,a.sep))
	
	if index==LEN-1:
		from hades.NoiseParams import sendMail
		sendMail('Noise Parameter - Delensing %s' %delensing_fraction)
	if index==2*LEN-1:
		from hades.NoiseParams import sendMail
		sendMail('Noise Parameter - Delensing %s' %delensing_fraction)
		
	map_id=paramFile['map_id'][index]
	noise_power=paramFile['noise_power'][index]
	FWHM=paramFile['FWHM'][index]
	length=len(paramFile['FWHM']) # total number of processes
	paramFile.close()
	
	
	tries=0 # retry if error
	
	while tries<2:
		
		# Now compute the estimators:
		from hades.wrapper import best_estimates
		output=best_estimates(map_id,FWHM=FWHM,noise_power=noise_power,delensing_fraction=delensing_fraction)
	
		eps_est=output[5][0] # just estimate
		eps_MC=output[7][5] # all MC values
		ang_est=output[6][0] # angle
		A_est=output[0][0] # monopole amplitude
		A_MC=output[7][0] # MC amplitude
		if len(eps_MC)==a.N_sims:
			break # successful run
		else:
			tries+=1
	if tries==3:
		print 'poor data at index %s' %index
				
	# Save output to file:
	outDir=a.root_dir+'NoiseParamsBatch_d%s/' %delensing_fraction
	if not os.path.exists(outDir):
		os.makedirs(outDir)
		
	# Just save the epsilon estimate and MC values to reduce file space
	np.savez(outDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power),\
	eps=eps_est,eps_MC=eps_MC,ang=ang_est,A=A_est,A_MC=A_MC)
	
	print 'Job %s of %s complete in %s seconds' %(index+1,length,time.time()-start_time)
	
	
def reconstruct_epsilon(map_size=a.map_size,sep=a.sep,freq=a.freq,delensing_fraction=a.delensing_fraction):
	import os
	""" Create a 2D array of epsilon for noise parameter data."""
	# First load in all iterated parameters
	paramFile=np.load(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(map_size,sep))
	
	# Initialise output array
	eps_num=np.zeros([len(a.noi_par_NoisePower),len(a.noi_par_FWHM)]) # numerator
	norm=np.zeros_like(eps_num) # normalisation
	fwhm_arr=np.zeros_like(eps_num) # array of FWHM for plotting
	power_arr=np.zeros_like(eps_num) # noise power values
	
	# For MC values:
	eps_MC_num=np.zeros([len(a.noi_par_NoisePower),len(a.noi_par_FWHM),a.N_sims])
	norm_MC=np.zeros_like(eps_MC_num)
	
	inDir=a.root_dir+'NoiseParamsBatch_d%s/' %delensing_fraction
	
	# Iterate over parameter number:
	for index in range(len(paramFile['map_id'])):
		if index%50==0:
			print 'Reconstructing job %s of %s' %(index+1,len(paramFile['map_id']))
		map_id=paramFile['map_id'][index]
		noise_power=paramFile['noise_power'][index]
		FWHM=paramFile['FWHM'][index]
	
		# Load in data file
		path=inDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power)
		if not os.path.exists(path): # if file not found
			continue
		dat=np.load(path)
		eps=dat['eps']
		eps_err=np.std(dat['eps_MC'])
		eps_MC=dat['eps_MC']
		if len(np.array(eps_MC))!=a.N_sims:
			print 'dodgy dat at index %s' %index
			errTag=True # to catch errors from dodgy data
			eps_MC=np.zeros(a.N_sims)
		else:
			errTag=False
		dat.close()
		
		# Find relevant position in array
		noi_pow_index=np.where(a.noi_par_NoisePower==noise_power)[0][0]
		fwhm_index=np.where(a.noi_par_FWHM==FWHM)[0][0]
		
		power_arr[noi_pow_index,fwhm_index]=noise_power
		fwhm_arr[noi_pow_index,fwhm_index]=FWHM
		
		# Construct mean epsilon
		SNR=1./eps_err*2.
		eps_num[noi_pow_index,fwhm_index]+=SNR*eps
		if not errTag:
			for j in range(len(eps_MC)):
				eps_MC_num[noi_pow_index][fwhm_index][j]+=SNR*eps_MC[j]
				norm_MC[noi_pow_index][fwhm_index][j]+=SNR
		norm[noi_pow_index,fwhm_index]+=SNR
	
	paramFile.close()
	
	# Now compute normalised mean epsilon
	patch_eps=eps_num/norm
	patch_eps_MC=eps_MC_num/norm_MC
	
	# Compute number of significance of detection
	sig=(patch_eps-np.mean(patch_eps_MC,axis=2))/np.std(patch_eps_MC,axis=2)
	
	# Save output
	np.savez(a.root_dir+'PatchEpsilonNoiseParams_f%s_d%s.npz' %(freq,delensing_fraction),\
	eps=patch_eps,eps_MC=patch_eps_MC,sig=sig,FWHM=fwhm_arr,noise_power=power_arr)
	
def noise_params_plot(map_size=a.map_size,sep=a.sep,\
	delensing_fraction=a.delensing_fraction,freq=a.freq,createData=True):
	""" Create a 2D plot of mean epsilon for patch against FWHM and noise-power noise parameters.
	Input: createData- > whether to reconstruct from batch output files or just read from file
	Plot is saved in NoiseParamsPlot/ directory"""
	import matplotlib.pyplot as plt
	import os
	
	# First load in mean epsilon + significance:
	from .NoiseParams import reconstruct_epsilon
	
	if createData: # create epsilon data from batch data
		reconstruct_epsilon(map_size,sep,delensing_fraction=delensing_fraction)
	patch_dat=np.load(a.root_dir+'PatchEpsilonNoiseParams_f%s_d%s.npz' %(freq,delensing_fraction))
	eps_arr=patch_dat['eps']
	eps_MC=patch_dat['eps_MC']
	X=len(eps_MC)
	Y=len(eps_MC[0])
	eps_iso_mean=np.zeros([X,Y])
	eps_iso_std=np.zeros([X,Y])
	for x in range(X):
		for y in range(Y):
			id=np.where(eps_MC[x][y]!=0.) # avoid spurious data
			eps_iso_mean[x,y]=np.mean(eps_MC[x][y][id])
			eps_iso_std[x,y]=np.std(eps_MC[x][y][id])
	#eps_iso_mean=np.mean(eps_MC,axis=2)
	eps_unbiased=np.array(eps_arr)-np.array(eps_iso_mean)
	#eps_iso_std=np.std(eps_MC,axis=2)
	sig_arr=patch_dat['sig']
	FWHM=patch_dat['FWHM']
	noise_power=patch_dat['noise_power']
	patch_dat.close()
	
	XLIM=[-0.1,5.95]
	YLIM=[-0.5,35]
	S=700
	
	# Create plot
	plt.figure()
	plt.scatter(noise_power,FWHM,c=eps_arr,s=S,marker='s',edgecolors='face',alpha=0.8)
	cbar=plt.colorbar()
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Mean Patch Epsilon')
	plt.xlim(XLIM)
	plt.ylim(YLIM)
	# Save plot
	outDir=a.root_dir+'NoiseParamsPlot_d%s_f%s/' %(delensing_fraction,freq)
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	plt.savefig(outDir+'MeanPatchEpsilon-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
	plt.clf()
	plt.close()
	
	# Create plot 2
	plt.figure()
	plt.scatter(noise_power,FWHM,c=sig_arr,s=S,marker='s',edgecolors='face',alpha=0.8)
	cbar=plt.colorbar()
	CS=plt.contour(noise_power,FWHM,sig_arr,levels=[5,8],colors='k',linestyles='--',alpha=0.8)
	plt.clabel(CS,colors='k',fontsize=14,fmt='%d')
	plt.plot(1,1.5,c='k',marker='*',ms=20)
	plt.plot(3,15,c='k',marker='*',ms=20)
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Epsilon Significance')
	plt.xlim(XLIM)
	plt.ylim(YLIM)
	# Save plot
	plt.savefig(outDir+'PatchEpsilonSignificance-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
	plt.clf()
	plt.close()
	
	# Create plot 3
	plt.figure()
	plt.scatter(noise_power,FWHM,c=eps_iso_mean,s=S,marker='s',edgecolors='face',alpha=0.8)
	cbar=plt.colorbar()
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Mean Isotropic MC Epsilon')
	plt.xlim(XLIM)
	plt.ylim(YLIM)
	# Save plot
	plt.savefig(outDir+'MeanMCIsoEpsilon-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
	plt.clf()
	plt.close()
	
	# Create plot 4
	plt.figure()
	plt.scatter(noise_power,FWHM,c=eps_iso_std,s=S,marker='s',edgecolors='face',alpha=0.8)
	cbar=plt.colorbar()
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Stdeviation Isotropic MC Epsilon')
	plt.xlim(XLIM)
	plt.ylim(YLIM)
	# Save plot
	plt.savefig(outDir+'StdevMCIsoEpsilon-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
	plt.clf()
	plt.close()
	
	# Create plot 5
	plt.figure()
	plt.scatter(noise_power,FWHM,c=eps_unbiased,s=S,marker='s',edgecolors='face',alpha=0.8)
	cbar=plt.colorbar()
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Unbiased Epsilon')
	plt.xlim(XLIM)
	plt.ylim(YLIM)
	# Save plot
	plt.savefig(outDir+'UnbiasedEpsilon-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
	plt.clf()
	plt.close()
	
def sendMail(typ):
	# Import smtplib for the actual sending function
	import smtplib

	# Import the email modules we'll need
	from email.mime.text import MIMEText

	# Open a plain text file for reading.  For this example, assume that
	# the text file contains only ASCII characters.
	text='Wowzer that was quick! Your HTCondor %s job is now complete!! \\ Parameters: map_size = %s, sep = %s, freq = %s, root_directory = %s ' %(typ,a.map_size,a.sep,a.freq,a.root_dir)
	 
	# Create a text/plain message
	msg = MIMEText(text)
	
	
	me = 'ohep2@cam.ac.uk'
	you = 'ohep2@cam.ac.uk'
	msg['Subject'] = 'HTCondor Job Complete'
	msg['From'] = me
	msg['To'] = you

	# Send the message via our own SMTP server, but don't include the
	# envelope header.
	s = smtplib.SMTP('localhost')
	s.sendmail(me, [you], msg.as_string())
	s.quit()
