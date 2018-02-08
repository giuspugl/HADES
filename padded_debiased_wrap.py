import numpy as np
from flipper import *
from hades.params import BICEP
a=BICEP()



def tile_wrap(map_id,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,N_bias=a.N_bias,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,l_step=a.l_step,lMin=a.lMin,lMax=a.lMax,rot=a.rot,freq=a.freq,\
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors,f_dust=a.f_dust,\
	rot_average=a.rot_average,useBias=a.useBias):
	""" Compute the estimated angle, amplitude and polarisation fraction with noise, correcting for bias.
	Noise model is from Hu & Okamoto 2002 and errors are estimated using MC simulations, which are all saved.
	
	Input: map_id (tile number)
	map_size (tile width in degrees)
	sep (separation of tile centres in degrees)
	N_sims (number of MC simulations)
	N_bias (no. sims used for bias computation)
	noise_power (noise power in microK-arcmin)
	FWHM (noise FWHM in arcmin)
	slope (fiducial slope of C_l isotropic dust dependance)
	lMin / lMax (range of ell values to apply the estimators over)
	l_step (step size for binning of 2D spectra)
	rot (angle to rotate by before applying estimators)
	freq (desired map frequency; 150 GHz for BICEP, 353 GHz for Vansyngel)
	delensing_fraction (efficiency of delensing; i.e. 0.1=90% removed)
	useTensors (Boolean, whether to include tensor noise from IGWs with r = 0.1)
	f_dust (factor to reduce mean dust amplitude by - for null testing - default = 1 - no reduction)
	rot_average (Boolean, whether to correct for pixellation error by performing (corrected) map rotations)
	useBias (Boolean, whether to use bias)
	
	Output: First 6 values: [estimate,isotropic mean, isotropic stdev] for {A,Afs,Afc,fs,fc,str,ang}
	7th: full data for N_sims as a sequence of 7 lists for each estimate (each of length N_sims)
	8th: full data for N_sims for the monopole bias term
	9th: [estimate,isotropic mean, isotropic stdev] for Hexadecapole power
	10: true monopole (from noiseless simulation) - for testing
	11th: bias (isotropic estimate of <H^2>)
	"""
	
	# First compute B-mode map from padded-real space map with desired padding ratio. Also compute the padded window function for later use
	from .PaddedPower import MakePowerAndFourierMaps
	fBdust,padded_window=MakePowerAndFourierMaps(map_id,padding_ratio=a.padding_ratio,map_size=map_size,sep=sep,freq=freq,fourier=True,power=False,returnMask=True)
	
	if a.hexTest:
		# TESTING - replace fourier B-mode from dust with random isotropic realisation of self
		powDust=fftTools.powerFromFFT(fBdust) # compute power
		from .PowerMap import oneD_binning
		ll,pp=oneD_binning(powDust.copy(),lMin,lMax,l_step,binErr=False,exactCen=False) # compute one-D binned spectrum
		from .RandomField import fill_from_Cell
		fBdust.kMap=fill_from_Cell(fBdust.copy(),ll,pp,fourier=True,power=False) # generate Gaussian realisation
	
	# Reduce dust amplitude by 'dedusting fraction'
	fBdust.kMap*=f_dust
	
	# Input directory:
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
	
	# First compute the total noise (instrument+lensing+tensors)
	from .NoisePower import noise_model,lensed_Cl,r_Cl
	Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction) # function for lensed Cl
	
	if useTensors: # include r = 0.1 estimate
		Cl_r_func=r_Cl()
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)+Cl_r_func(l)
	else:
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)
		
	from .RandomField import fill_from_model,fill_from_Cell
	fourierNoise=fBdust.copy() # template
	fourierNoise.kMap=fill_from_model(fourierNoise.copy(),total_Cl_noise,fourier=True,power=False)
		
	# Compute total map
	totFmap=fBdust.copy()
	totFmap.kMap+=fourierNoise.kMap# for B modes
	
	# Now convert to power-space
	totPow=fftTools.powerFromFFT(totFmap) # total power map
	Bpow=fftTools.powerFromFFT(fBdust) # dust only map
				
	# Compute true amplitude using ONLY dust map
	from .KKdebiased import derotated_estimator
	p=derotated_estimator(Bpow.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,factor=None,FWHM=0.,\
			noise_power=1.e-400,rot=rot,delensing_fraction=0.,useTensors=False,debiasAmplitude=False,rot_average=rot_average)
	trueA=p[0]
			
	
	# Compute rough semi-analytic C_ell spectrum
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)
	
	## Testing
	if False:#a.hexTest:
		#print 'testing'
		# test using a known power spectrum which replaces the data
		testmap=totPow.copy()
		testF=totFmap.copy()
		from .RandomField import fill_from_model
		def trial_mod(ell):
			return analytic_model(ell,1.0e-13,2.42)
		testmap.powerMap,testF.kMap=fill_from_model(Bpow.copy(),trial_mod,fourier=True,power=True)
		totPow=testmap # replace data by MC map for testing
		totFmap=testF

	# TESTING with an isotropic sim map
	if False:
		from .PowerMap import oneD_binning
		l_cen,mean_pow = oneD_binning(totPow.copy(),lMin,lMax,l_step,binErr=False,exactCen=False)#a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 
		totPow.powerMap,totFmap.kMap=fill_from_Cell(totPow.copy(),l_cen,mean_pow,fourier=True,power=True)
		# Replace data with a single MC sim

	# Compute anisotropy parameters
	A_est,fs_est,fc_est,Afs_est,Afc_est,finalFactor=derotated_estimator(totPow.copy(),map_id,lMin=lMin,\
		lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,rot=rot,\
		delensing_fraction=delensing_fraction,useTensors=useTensors,debiasAmplitude=True,rot_average=rot_average)
	# (Factor is expected monopole amplitude (to speed convergence))
	
	## Run MC Simulations	
	
	# Compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totPow.copy(),lMin,lMax,l_step,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 
	# gives central binning l and mean power in annulus using window function corrections 
		
	# First compute the bias factor
	if useBias:
		print 'Computing bias'
		bias_data=np.zeros(N_bias)
		for n in range(N_bias):
			fBias=totFmap.copy()  # template
			fBias.kMap=fill_from_Cell(totPow.copy(),l_cen,mean_pow,fourier=True,power=False)
			bias_cross=fftTools.powerFromFFT(fBias.copy(),totFmap.copy()) # cross map
			bias_self=fftTools.powerFromFFT(fBias.copy()) # self map
			# First compute estimators on cross-spectrum
			cross_ests=derotated_estimator(bias_cross.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,\
							factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
							rot=rot,delensing_fraction=delensing_fraction,useTensors=useTensors,\
							debiasAmplitude=True,rot_average=rot_average)
			self_ests=derotated_estimator(bias_self.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,\
							factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
							rot=rot,delensing_fraction=delensing_fraction,useTensors=useTensors,\
							debiasAmplitude=True,rot_average=rot_average)
			bias_data[n]=-1.*(self_ests[3]**4.+self_ests[4]**2.)+4.*(cross_ests[3]**2.+cross_ests[4]**2.)
		# Now compute the mean bias - this debiases the DATA only
		bias=np.mean(bias_data)
	else:
		print 'No bias subtraction'
		bias=0.
			
	## Now run the MC sims proper:
	# Initialise arrays
	A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow2_MC=[np.zeros(N_sims) for _ in range(8)]
	MC_map=Bpow.copy() # template for SIM-SIM data
	
	for n in range(N_sims): # for each MC map
		if n%50==0:
			print('MapID %s: Starting simulation %s of %s' %(map_id,n+1,N_sims))
		# Create the map with a random implementation of Cell, returning both power map and Fourier space map
		MC_map.powerMap=fill_from_Cell(totPow.copy(),l_cen,mean_pow,fourier=False,power=True)
		
		# Now use the estimators on the MC sims
		output=derotated_estimator(MC_map.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
			rot=rot, delensing_fraction=delensing_fraction,useTensors=a.useTensors,\
			debiasAmplitude=True,rot_average=rot_average) 
		# compute MC anisotropy parameters  
		A_MC[n]=output[0]
		fs_MC[n]=output[3]/output[0]
		fc_MC[n]=output[4]/output[0]
		Afs_MC[n]=output[3] # these are fundamental quantities here
		Afc_MC[n]=output[4]
		epsilon_MC[n]=np.sqrt((output[3]**2.+output[4]**2.))/output[0] # NOT corrected for bias in <H^2>
		ang_MC[n]=0.25*180./np.pi*np.arctan(output[3]/output[4]) # NB: this is not corrected for bias
		HexPow2_MC[n]=output[3]**2.+output[4]**2. 
		
	HexPow2_MC-=np.mean(HexPow2_MC)*np.ones_like(HexPow2_MC) # remove the bias
	
	print 'MC sims complete'	
	# Regroup data
	allMC=[A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow2_MC]
	
	# Compute anisotropy fraction and angle from data
	ang_est=0.25*180./np.pi*(np.arctan(Afs_est/Afc_est)) # in degrees (already corrected for rotation) - NB: not debiased
	frac_est=np.sqrt((Afs_est**2.+Afc_est**2.))/A_est # BIASED sqrt(<H^2>)/A
	HexPow2_est=Afs_est**2.+Afc_est**2.-bias # hexadecapolar power - debiased
	
	# Compute means and standard deviations
	A_mean=np.mean(A_MC)
	A_std=np.std(A_MC)
	fc_mean=np.mean(fc_MC)
	fs_mean=np.mean(fs_MC)
	fc_std=np.std(fc_MC)
	fs_std=np.std(fs_MC)
	frac_mean=np.mean(epsilon_MC)
	frac_std=np.std(epsilon_MC)
	ang_mean=np.mean(ang_MC)
	ang_std=np.std(ang_MC)
	HexPow2_mean=np.mean(HexPow2_MC)
	HexPow2_std=np.std(HexPow2_MC)
	Afs_mean=np.mean(Afs_MC)
	Afc_mean=np.mean(Afc_MC)
	Afs_std=np.std(Afs_MC)
	Afc_std=np.std(Afc_MC)
	
	# Regroup data
	Adat=[A_est,A_mean,A_std]
	fsdat=[fs_est,fs_mean,fs_std]
	fcdat=[fc_est,fc_mean,fc_std]
	Afsdat=[Afs_est,Afs_mean,Afs_std]
	Afcdat=[Afc_est,Afc_mean,Afc_std]
	fracdat=[frac_est,frac_mean,frac_std] # hexadecapolar anisotropy fraction (epsilon)
	angdat=[ang_est,ang_mean,ang_std] # anisotropy angle
	HexPow2dat=[HexPow2_est,HexPow2_mean,HexPow2_std] # hexadecapole amplitude
	
	# Return all output
	return Adat,fsdat,fcdat,Afsdat,Afcdat,fracdat,angdat,allMC,[],HexPow2dat,trueA,bias # (empty set to avoid reordering later code)


