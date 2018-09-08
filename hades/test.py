def reconstruct_epsilon(map_size=a.map_size,sep=a.sep):
	import numpy as np
	""" Create a 2D array of epsilon for noise parameter data."""
	# First load in all iterated parameters
	paramFile=np.load(a.root_dir+'%sdeg%sBatchNoiseParams.npz' %(map_size,sep))
	
	# Initialise output array
	eps_num=np.zeros(len(a.noi_par_NoisePower),len(a.noi_par_FWHM)) # numerator
	norm=np.zeros_like(eps_num) # normalisation
	
	# For MC values:
	eps_MC_num=np.zeros(len(a.noi_par_NoisePower,len(a.noi_par_FWHM),a.N_sims)
	norm_MC=np.zeros_like(eps_MC_num)
	
	inDir=a.root_dir+'NoiseParamsBatch/'
	
	# Iterate over parameter number:
	for index in range(len(paramFile['map_id'])):
		if index%10==0:
			print 'Reconstructing job %s of %s' %(index+1,len(paramFile['map_id']))
		map_id=paramFile['map_id'][index]
		noise_power=paramFile['noise_power'][index]
		FWHM=paramFile['FWHM'][index]
	
		# Load in data file
		dat=np.load(outDir+'id%s_fwhm%s_power%s.npz' %(map_id,FWHM,noise_power))
		eps=dat['eps']
		eps_err=np.std(dat['eps_MC'])
		eps_MC=dat['eps_MC']
		dat.close()
		
		# Find relevant position in array
		noi_pow_index=np.where(a.noi_par_NoisePower==noise_power)
		fwhm_index=np.where(a.noi_par_FWHM=FWHM)
		
		SNR=1./eps_err*2.
		eps_num[noi_pow_index,fwhm_index]+=SNR*eps
		for j in range(len(eps_MC)):
			eps_MC_num[j]+=SNR*eps_MC[j]
			norm_MC[j]+=SNR
		norm[noi_pow_index,fwhm_index]+=SNR
	
	paramFile.close()
	
	# Now compute normalised mean epsilon
	patch_eps=eps_num/norm
	patch_eps_MC=eps_MC_num/norm_MC
	
	# Compute number of significance of detection
	sig=(patch_eps-np.mean(patch_eps_MC,axis=2)/np.std(patch_eps_MC,axis=2)
	return patch_eps,patch_eps_MC,sig
	
def noise_params_plot(a.map_size,a.sep):
	""" Create a 2D plot of mean epsilon for patch against FWHM and noise-power noise parameters.
	Plot is saved in NoiseParamsPlot/ directory"""
	import matplotlib.pyplot as plt
	import os
	
	# First load in mean epsilon + significance:
	from .NoiseParams import reconstruct_epsilon
	eps_arr,_,sig_arr=reconstruct_epsilon(map_size,sep)
	
	# Construct X,Y axes:
	NP,FW=np.meshgrid(a.noi_par_NoisePower,a.noi_par_FWHM)
	
	# Create plot
	plt.figure()
	plt.scatter(NP,FW,c=eps_arr,s=30,marker='o')
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Noise Parameter Space for mean Patch Epsilon')
	plt.colorbar()
	
	# Save plot
	outDir=a.root_dir+'NoiseParamsPlot/'
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	plt.savefig(outDir+'MeanPatchEpsilon-%sdeg%s.png' %(map_size,sep))
	plt.clf()
	plt.close()
	
	# Create plot 2
	plt.figure()
	plt.scatter(NP,FW,c=sig_arr,s=30,marker='o')
	plt.ylabel('FWHM / arcmin')
	plt.xlabel('Noise-Power / uK-arcmin')
	plt.title('Noise Parameter Space for Epsilon Significance')
	plt.colorbar()
	
	# Save plot
	plt.savefig(outDir+'PatchEpsilonSignificance-%sdeg%s.png' %(map_size,sep))
	plt.clf()
	plt.close()
