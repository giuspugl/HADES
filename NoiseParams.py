import numpy as np
from hades.params import BICEP
a=BICEP()

def create_params(map_size=a.map_size,sep=a.sep):
	""" Create parameter space + map_id grid for batch processing."""
	# Load in good map IDs;
	goodMaps=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	# Create meshed list of all parameters
	NP,FW,DF=np.meshgrid(a.noi_par_NoisePower,a.noi_par_FWHM,a.noi_par_delensing)
	
	# Save output
	np.savez(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(map_size,sep),map_id=goodMaps,noise_power=NP.ravel(),FWHM=FW.ravel(),delensing_fraction=DF.ravel())
	
	# returns length of parameter string
	return len(NP.ravel()) 

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
	
	if index>LEN-1: # crash if at end of data
		print 'at end of data'
		sys.exit()
		
	paramFile=np.load(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(a.map_size,a.sep))
	all_map_id=paramFile['map_id']
	noise_power=paramFile['noise_power'][index]
	FWHM=paramFile['FWHM'][index]
	delensing_fraction=paramFile['delensing_fraction'][index]
	length=len(paramFile['FWHM']) # total number of processes
	paramFile.close()
	outDir=a.root_dir+'DebiasedNoiseParamsBatch_d%s/' %delensing_fraction
			
	
	# Compute all maps with these parameters
	for mi,map_id in enumerate(all_map_id):
		if os.path.exists(outDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power)):
			#os.remove(outDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power))
			print 'already exists; continuing'
		else:
			#pass
		#if True:
			# Now compute the estimators:
			from hades.fast_wrapper import padded_wrap
			output=padded_wrap(map_id,FWHM=FWHM,noise_power=noise_power,delensing_fraction=delensing_fraction)
			H2_est=output[9][0] # just estimate
			H2_MC=output[7][7] # all MC values
			ang_est=output[6][0] # angle
			A_est=output[0][0] # monopole amplitude
			A_MC=output[7][0] # MC amplitude
			
			# Save output to file:
			if not os.path.exists(outDir):
				os.makedirs(outDir)
				
			# Just save the H2 estimate and MC values to reduce file space
			np.savez(outDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power),\
				H2=H2_est,H2_MC=H2_MC,ang=ang_est,A=A_est)
			print 'Map %s of %s complete in %s seconds' %(mi+1,len(all_map_id),time.time()-start_time)
			
	print 'Job %s of %s complete in %s seconds' %(index+1,length,time.time()-start_time)
	
	# send email notification
	if index==LEN-1:
		from hades.NoiseParams import sendMail
		sendMail('Noise Parameter Space')
	
	
def reconstruct_hexadecapole(map_size=a.map_size,sep=a.sep,freq=a.freq):#,delensing_fraction=a.delensing_fraction):
	import os
	""" Create a 2D array of hexadecapole for noise parameter data."""
	# First load in all iterated parameters
	paramFile=np.load(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(map_size,sep))
	
	# Initialise output array
	H2_num=np.zeros([len(a.noi_par_NoisePower),len(a.noi_par_FWHM),len(a.noi_par_delensing)]) # numerator
	norm=np.zeros_like(H2_num) # normalisation
	fwhm_arr=np.zeros_like(H2_num) # array of FWHM for plotting
	power_arr=np.zeros_like(H2_num) # noise power values
	delensing_arr=np.zeros_like(H2_num)
	
	# For MC values:
	H2_MC_num=np.zeros([len(a.noi_par_NoisePower),len(a.noi_par_FWHM),len(a.noi_par_delensing),a.N_sims])
	norm_MC=np.zeros_like(H2_MC_num)
	
	
	errFiles=[] # runs with errors
	
	# Iterate over parameter number:
	for mi,map_id in enumerate(paramFile['map_id']):
		print 'Reconstructing tile %s of %s' %(mi+1,len(paramFile['map_id']))
			
		for index in range(len(paramFile['FWHM'])):
			delensing_fraction=paramFile['delensing_fraction'][index]
			noise_power=paramFile['noise_power'][index]
			FWHM=paramFile['FWHM'][index]
			inDir=a.root_dir+'DebiasedNoiseParamsBatch_d%s/' %delensing_fraction
			
			# Load in data file
			path=inDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power)
			if not os.path.exists(path): # if file not found
				continue
			#print map_id,FWHM,noise_power
			dat=np.load(path)
			H2=dat['H2']
			H2_MC=dat['H2_MC']
			H2_err=np.std(H2)
		
			if len(np.array(H2_MC))!=a.N_sims:
				print 'dodgy dat at index %s' %index
				errTag=True # to catch errors from dodgy data
				H2_MC=np.zeros(a.N_sims)
				errFiles.append(index)
			else:
				errTag=False
			dat.close()
		
			# Find relevant position in array
			noi_pow_index=np.where(a.noi_par_NoisePower==noise_power)[0][0]
			fwhm_index=np.where(a.noi_par_FWHM==FWHM)[0][0]
			delens_index=np.where(a.noi_par_delensing==delensing_fraction)[0][0]
		
			power_arr[noi_pow_index,fwhm_index,delens_index]=noise_power
			fwhm_arr[noi_pow_index,fwhm_index,delens_index]=FWHM
			delensing_arr[noi_pow_index,fwhm_index,delens_index]=delensing_fraction
		
			# Construct mean epsilon
			SNR=1.#/eps_err*2.
			H2_num[noi_pow_index,fwhm_index]+=SNR*H2
			if not errTag:
				for j in range(len(H2_MC)):
					H2_MC_num[noi_pow_index][fwhm_index][delens_index][j]+=SNR*H2_MC[j]
					norm_MC[noi_pow_index][fwhm_index][delens_index][j]+=SNR
			norm[noi_pow_index,fwhm_index,delens_index]+=SNR
			del noi_pow_index,fwhm_index,delens_index,SNR,H2,H2_MC,H2_err
	paramFile.close()
	import pickle
	pickle.dump([H2_MC_num,norm_MC,norm,H2_num,fwhm_arr,power_arr,delensing_arr],open('pcklall.pkl','w'))
	
	# Now compute normalised mean epsilon
	patch_H2=H2_num/norm
	patch_H2_MC=H2_MC_num/norm_MC
	
	# Compute number of significance of detection
	sig=(patch_H2-np.mean(patch_H2_MC,axis=3))/np.std(patch_H2_MC,axis=3)
	
	# Save output
	np.save(a.root_dir+'ErrorFiles.npz',errFiles)
	
	np.savez(a.root_dir+'PatchHex2NoiseParams.npz',H2=patch_H2,H2_MC=patch_H2_MC,sig=sig,FWHM=fwhm_arr,noise_power=power_arr,delensing_fraction=delensing_arr)
	
def noise_params_plot(map_size=a.map_size,sep=a.sep,\
	freq=a.freq,createData=False,S=175):
	""" Create a 2D plot of mean epsilon for patch against FWHM and noise-power noise parameters.
	Input: createData- > whether to reconstruct from batch output files or just read from file
	S -> pixel size in plot
	Plot is saved in NoiseParamsPlot/ directory"""
	import matplotlib.pyplot as plt
	import os
	
	# First load in mean epsilon + significance:
	from .NoiseParams import reconstruct_hexadecapole
	
	if createData: # create epsilon data from batch data
		reconstruct_hexadecapole(map_size,sep)
	patch_dat=np.load(a.root_dir+'PatchHex2NoiseParams.npz')
	for DL,delensing_fraction in enumerate(a.noi_par_delensing):
		H2_arr=patch_dat['H2']
		H2_arr=H2_arr[:,:,DL]
		H2_MC=patch_dat['H2_MC']
		H2_MC=H2_MC[:,:,DL]
		X=len(H2_MC)
		Y=len(H2_MC[0])
		H2_iso_mean=np.zeros([X,Y])
		H2_iso_std=np.zeros([X,Y])
		for x in range(X):
			for y in range(Y):
				id=np.where(H2_MC[x][y]!=0.) # avoid spurious data
				H2_iso_mean[x,y]=np.mean(H2_MC[x][y][id])
				H2_iso_std[x,y]=np.std(H2_MC[x][y][id])
		#eps_iso_mean=np.mean(eps_MC,axis=2)
		H2_unbiased=np.array(H2_arr)#-np.array(H2_iso_mean)
		#eps_iso_std=np.std(eps_MC,axis=2)
		sig_arr=patch_dat['sig']
		sig_arr=sig_arr[:,:,DL]
		FWHM=patch_dat['FWHM']
		FWHM=FWHM[:,:,DL]
		noise_power=patch_dat['noise_power']
		noise_power=noise_power[:,:,DL]
		#patch_dat.close()
	
		XLIM=[min(noise_power.ravel())-0.2,max(noise_power.ravel())+0.2]#[-0.1,5.95]
		YLIM=[min(FWHM.ravel())-0.3,max(FWHM.ravel())+0.3]#[-0.5,35]
		
		# Create plot
		plt.figure()
		plt.scatter(noise_power,FWHM,c=H2_arr,s=S,marker='s',edgecolors='face',alpha=0.8)
		cbar=plt.colorbar()
		plt.ylabel('FWHM / arcmin')
		plt.xlabel('Noise-Power / uK-arcmin')
		plt.title('Mean Debiased Patch H2')
		plt.xlim(XLIM)
		plt.ylim(YLIM)
		# Save plot
		outDir=a.root_dir+'NoiseParamsPlot_d%s_f%s/' %(delensing_fraction,freq)
		if not os.path.exists(outDir):
			os.makedirs(outDir)
		plt.savefig(outDir+'MeanPatchH2-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		# Create plot 2
		plt.figure()
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.scatter(noise_power,FWHM,c=sig_arr,s=S,marker='s',edgecolors='face',alpha=0.8)#,vmax=300)
		cbar=plt.colorbar()
		CS=plt.contour(noise_power,FWHM,sig_arr,levels=[5,10,20],colors='k',linestyles='--',alpha=0.8)
		plt.clabel(CS,colors='k',fontsize=14,fmt='%d')
		plt.plot(1,1.5,c='k',marker='*',ms=20)
		plt.plot(5,30,c='k',marker='*',ms=20)
		plt.ylabel('FWHM / arcmin')
		plt.xlabel(r'Noise-Power / $\mu$K-arcmin')
		plt.title('Significance of Patch $\mathcal{H}^2$ Detection')
		plt.xlim(XLIM)
		plt.ylim(YLIM)
		# Save plot
		plt.savefig(outDir+'PatchH2Significance-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		# Create plot 4
		plt.figure()
		plt.scatter(noise_power,FWHM,c=H2_iso_std,s=S,marker='s',edgecolors='face',alpha=0.8)
		cbar=plt.colorbar()
		plt.ylabel('FWHM / arcmin')
		plt.xlabel('Noise-Power / uK-arcmin')
		plt.title('Stdeviation Isotropic MC H2')
		plt.xlim(XLIM)
		plt.ylim(YLIM)
		# Save plot
		plt.savefig(outDir+'StdevMCIsoH2-%sdeg%s.png' %(map_size,sep),bbox_inches='tight')
		plt.clf()
		plt.close()
		
	print 'plotting complete'
		
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
