from flipper import *
import numpy as np
from hades.params import BICEP
a=BICEP()

if __name__=='__main__':
	""" This is the iterator for batch processing the map creation through HTCondor. Each map is done separately, and argument is map_id."""
	import time
	start_time=time.time()
	import sys
	import pickle
	sys.path.append('/data/ohep2/')
	sys.path.append('/home/ohep2/Masters/')
	import os
	
	batch_id=int(sys.argv[1]) # batch_id number
	
	# First load good IDs:
	goodFile=a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep)
	
	outDir=a.root_dir+'HexBatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
	
	if a.remakeErrors:
		if os.path.exists(outDir+'%s.npy' %batch_id):
			print 'output exists; exiting'
			sys.exit()
	
	if batch_id<110: # create first time
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids()
		print 'creating good IDs'
		
	goodIDs=np.load(goodFile)
	
	
	if batch_id>len(goodIDs)-1:
		print 'Process %s terminating' %batch_id
		sys.exit() # stop here
	
	map_id=goodIDs[batch_id] # this defines the tile used here
	
	print '%s starting for map_id %s' %(batch_id,map_id)

		
	# Now run the estimation
	from hades.hex_wrap import debiased_wrap#debiased_wrap # estimator_wrap
	output=debiased_wrap(map_id) # debiased_wrap(map_id) estimator_wrap(map_id)
	
	# Save output to file
	if not os.path.exists(outDir): # make directory
		os.makedirs(outDir)
		
	np.save(outDir+'%s.npy' %batch_id, output) # save output
	
	print "Job %s complete in %s seconds" %(batch_id,time.time()-start_time)
	
	if batch_id==len(goodIDs)-2:
		if a.send_email:
			from hades.NoiseParams import sendMail
			sendMail('Single Map')

def low_dust_estimator(map,map_id,lMin=a.lMin,lMax=a.lMax,FWHM=a.FWHM,noise_power=a.noise_power,\
    slope=a.slope,factor=None,rot=0.,delensing_fraction=a.delensing_fraction,useTensors=a.useTensors,debiasAmplitude=True):
    """Use modified KK14 estimators to compute polarisation hexadecapole amplitude and angle via Afs,Afc parameters.
    This uses the noise model in hades.NoisePower.noise_model.
    A is computed recursively, since the S/N ratio depends on it (weakly)
    
    Inputs: map (in power-space)
    map_size = width of map in degrees
    lMin,lMax= fitting range of l
    slope -> fiducial C_l map slope
    rot -> optional angle for pre-rotation of power-space map in degrees.
    delensing_fraction -> efficiency of delensing (0.1-> 90% removed)
    factor -> expected amplitude factor (to speed convergence)
    useTensors -> Boolean whether to include r = 0.1 tensor modes from IGWs
    debiasAmplitude -> Boolean whether to subtract noise+lensing C_l for estimation of A
    
    Outputs:
    A,fs,fc, Afs, Afc from estimators. NB these are corrected for any map pre-rotation.
    """
    from hades.NoisePower import lensed_Cl,r_Cl,noise_model
    
    def Cl_total_noise_func(l):
        """This is total C_l noise from lensing, B-modes and experimental noise"""
        Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction)
        if useTensors:
            Cl_r_func=r_Cl()
            return Cl_lens_func(l)+Cl_r_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)
        else:
            return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)
    
    if factor==None:
        # If no estimate for monopole amplitude, compute this recursively (needed in SNR)
        # A only weakly depends on the SNR so convergence is usually quick
        N=0
        if a.f_dust>0.:
        	Afactor=1e-12*a.f_dust**2. # initial estimate
        else:
        	Afactor=1e-16 # some arbitrary small value

        while N<20: # max no. iterations
            goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax)) # correct pixels
            lMap=map.modLMap.ravel()[goodPix] # mod(l)
            PowerMap=map.powerMap.ravel()[goodPix] # B-mode (biased) power
            OtherClMap=Cl_total_noise_func(lMap) # extra C_l power
            if debiasAmplitude:
            	debiasedPowerMap=PowerMap-OtherClMap # power estimate only due to dust
            else:
            	debiasedPowerMap=PowerMap
            fiducialClMap=lMap**(-slope) # fiducial Cl

            SNMap = (Afactor*fiducialClMap)/(Afactor*fiducialClMap+OtherClMap) # SN ratio

            # Compute estimate for A
            Anum=np.sum(debiasedPowerMap*(SNMap**2.)/fiducialClMap)
            Aden=np.sum(SNMap**2.)

            # Now record output
            lastFactor=Afactor
            Afactor=Anum/Aden
            
            # Stop if approximate convergence reached
            if np.abs((Afactor-lastFactor)/Afactor)<0.01:
                break
            N+=1

        if N==20:
            print 'Map %s did not converge with slope: %.3f, Afactor %.3e, last factor: %.3e' %(map_id,slope,Afactor,lastFactor)
        finalFactor=Afactor
        
    else:
        finalFactor=factor # just use the A estimate from input      

        # Now compute A,Afs,Afc (recompute A s.t. all best estimators use same SNR)
    
    goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax)) # pixels in correct range
    angMap=(map.thetaMap.ravel()[goodPix]+rot*np.ones_like(map.thetaMap.ravel()[goodPix]))*np.pi/180. # angle in radians
    cosMap=np.cos(4.*angMap) # map of cos(4theta)
    sinMap=np.sin(4.*angMap)
    lMap=map.modLMap.ravel()[goodPix] # |ell|
    PowerMap=map.powerMap.ravel()[goodPix] # B-mode biased power
    OtherClMap=Cl_total_noise_func(lMap) # extra C_l power from lensing + noise (+ tensor modes)
    if debiasAmplitude:
    	debiasedPowerMap=PowerMap-OtherClMap # power estimate only due to dust
    else:
    	debiasedPowerMap=PowerMap
    fiducialClMap=lMap**(-slope) # fiducial Cl
    
    SNmap=(finalFactor*fiducialClMap)/(finalFactor*fiducialClMap+OtherClMap) # signal-to-noise ratio
    
    # Now compute estimates for A, Afs, Afc
    Anum=np.sum(debiasedPowerMap*(SNmap**2.)/fiducialClMap) # noise-debiased
    Aden=np.sum(SNmap**2.)
    Afcnum=np.sum(debiasedPowerMap*cosMap*(SNmap**2.)/fiducialClMap) # cosine coeff
    Afcden=np.sum((SNmap*cosMap)**2.)
    Afsnum=np.sum(debiasedPowerMap*sinMap*(SNmap**2.)/fiducialClMap) # sine coeff 
    Afsden=np.sum((SNmap*sinMap)**2.)
    A=Anum/Aden
    Afc=Afcnum/Afcden
    Afs=Afsnum/Afsden
    fs=Afs/A # fs,fc are less reliable since have error from both A and Afs
    fc=Afc/A

    # Now correct for the map rotation
    rot_rad=rot*np.pi/180. # angle of rotation in radians
    fs_corr=fs*np.cos(rot_rad*4.)-fc*np.sin(rot_rad*4.)
    fc_corr=fs*np.sin(rot_rad*4.)+fc*np.cos(rot_rad*4.)
    Afs_corr=Afs*np.cos(rot_rad*4.)-Afc*np.sin(rot_rad*4.)
    Afc_corr=Afs*np.sin(rot_rad*4.)+Afc*np.cos(rot_rad*4.)
    
    if factor==None:
    	return A,fs_corr,fc_corr,Afs_corr,Afc_corr,finalFactor # to save finalFactor if different to A
    else:
        return A,fs_corr,fc_corr,Afs_corr,Afc_corr


def estimator_wrap(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,l_step=a.l_step,lMin=a.lMin,lMax=a.lMax,rot=a.rot,freq=a.freq,\
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors,f_dust=a.f_dust):
	""" Compute the estimated angle, amplitude and polarisation fraction with noise, using zero-padding.
	Noise model is from Hu & Okamoto 2002 and errors are estimated using MC simulations, which are all saved.
	
	Input: map_id (tile number)
	padding_ratio (ratio of real-space padded tile width to initial tile width)
	map_size (tile width in degrees)
	sep (separation of tile centres in degrees)
	N_sims (number of MC simulations)
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
	
	Output: First 6 values: [estimate,isotropic mean, isotropic stdev] for {A,Afs,Afc,fs,fc,str,ang}
	7th: full data for N_sims as a sequence of 7 lists for each estimate (each of length N_sims)
	8th: full data for N_sims for the monopole bias term
	9th: [estimate,isotropic mean, isotropic stdev] for Hexadecapole power
	10: true monopole (from noiseless simulation) - for testing
	"""
	
	# First compute high-resolution B-mode map from padded-real space map with desired padding ratio
	from .PaddedPower import MakePowerAndFourierMaps
	Bpow,fourierB=MakePowerAndFourierMaps(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq)
	
	# Reduce dust amplitude by 'dedusting fraction'
	print f_dust
	Bpow.powerMap*=f_dust**2. # squared since in power space
	fourierB.kMap*=f_dust
	
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
		
	from .RandomField import fill_from_model
	noiselensedMap=Bpow.copy() # template
	fourierNoise=fourierB.copy()

	noiselensedMap.powerMap,fourierNoise.kMap=fill_from_model(Bpow,total_Cl_noise,fourier=True)
		
	# Compute total map
	totFourier=fourierNoise.copy()
	totFourier.kMap+=fourierB.kMap
	totMap=fftTools.powerFromFFT(totFourier)
	#totMap=Bpow.copy()
	#totMap.powerMap=Bpow.powerMap+noiselensedMap.powerMap
			
	# Compute true amplitude
	from .hex_wrap import low_dust_estimator
	p=low_dust_estimator(Bpow.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,factor=None,FWHM=0.,\
			noise_power=1.e-300,rot=rot,delensing_fraction=0.,useTensors=False)
	trueA=p[0]
			
	
	# Compute rough semi-analytic C_ell spectrum
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)
	
	H_all=[]
	for _ in range(a.repeat):
		noiselensedMap.powerMap,fourierNoise.kMap=fill_from_model(Bpow,total_Cl_noise,fourier=True)
		
		# Compute total map
		totFourier=fourierNoise.copy()
		totFourier.kMap+=fourierB.kMap
		totMap=fftTools.powerFromFFT(totFourier)
	
		## Testing
		if a.hexTest:
			if False: # way 1 of testing
				# Compute 1D power spectrum by binning in annuli
				from .PowerMap import oneD_binning
				l_cen,mean_pow = oneD_binning(totMap.copy(),lMin,lMax,l_step,binErr=False,exactCen=False)
				testmap=totMap.copy()
				from .RandomField import fill_from_Cell
				testmap.powerMap=fill_from_Cell(Bpow.copy(),l_cen,mean_pow)
				totMap=testmap		
				
			if True: # way 2 of testing
				testmap=totMap.copy()
				from .RandomField import fill_from_model
				def trial_mod(ell):
					return analytic_model(ell,1.0e-13,2.42)
				testmap.powerMap=fill_from_model(Bpow.copy(),trial_mod)
				totMap=testmap # replace data by MC map for testing
					
		
		from .hex_wrap import low_dust_estimator
		A_est,fs_est,fc_est,Afs_est,Afc_est,finalFactor=low_dust_estimator(totMap.copy(),map_id,lMin=lMin,\
			lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,rot=rot,\
			delensing_fraction=delensing_fraction,useTensors=a.useTensors)
		# (Factor is expected monopole amplitude (to speed convergence))
		# NB: Afs, Afc are the fundamental quantities here
	
		H_all.append(np.sqrt(Afs_est**2.+Afc_est**2.))
	
	## Run MC Simulations	
	
	# Compute spline fit to all model data
	#goodPix=np.where((totMap.modLMap<lMax)&(totMap.modLMap>lMin))
	#ells=totMap.modLMap[goodPix].ravel()
	#pows=totMap.powerMap[goodPix].ravel()
	#from scipy.interpolate import UnivariateSpline
	
	#spl=UnivariateSpline(ells,np.log10(pows),k=3)
		
	#def spl_fit(l):
	#	return 10.**spl(l)
	
	# Compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totMap.copy(),lMin,lMax,l_step,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 
	# gives central binning l and mean power in annulus using window function corrections 
	
		
	# Initialise arrays
	A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow_MC=[np.zeros(N_sims) for _ in range(8)]
	
	#from hades.NoisePower import single_MC
	from .RandomField import fill_from_Cell
	MC_map=Bpow.copy() # template
	
	for n in range(N_sims): # for each MC map
		if n%100==0:
			print('MapID %s: Starting simulation %s of %s' %(map_id,n+1,N_sims))
		# Create the map with a random implementation of Cell
		MC_map.powerMap=fill_from_Cell(totMap,l_cen,mean_pow)
		#def model(ell):
		#	"""Model for MC power spectrum"""
		#	return analytic_model(ell,A_est=0.,slope=slope)#A_est,slope=slope)
		#print A_est
		#MC_map.powerMap=fill_from_model(totMap,model)
		#MC_map.powerMap=fill_from_model(MC_map.copy(),spl_fit)
		# Now use the estimators
		output=low_dust_estimator(MC_map.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
			rot=rot, delensing_fraction=delensing_fraction,useTensors=a.useTensors) 
		# compute MC anisotropy parameters  
		A_MC[n]=output[0]
		fs_MC[n]=output[3]/output[0]
		fc_MC[n]=output[4]/output[0]
		Afs_MC[n]=output[3]
		Afc_MC[n]=output[4]
		epsilon_MC[n]=np.sqrt((output[3]**2.+output[4]**2.)/(output[0]**2.))#fc_MC[-1]**2.+fs_MC[-1]**2.))
		ang_MC[n]=0.25*180./np.pi*np.arctan(output[3]/output[4])
		HexPow_MC[n]=np.sqrt(output[3]**2.+output[4]**2.)
		
	allMC=[A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow_MC]
	
	# Compute anisotropy fraction and angle from data
	ang_est=0.25*180./np.pi*(np.arctan(Afs_est/Afc_est)) # in degrees (already corrected for rotation)
	frac_est=np.sqrt(fs_est**2.+fc_est**2.)
	HexPow_est=np.mean(H_all)#np.sqrt(Afs_est**2.+Afc_est**2.) # biased hexadecapolar power
	
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
	HexPow_mean=np.mean(HexPow_MC)
	HexPow_std=np.std(HexPow_MC)
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
	HexPowdat=[HexPow_est,HexPow_mean,HexPow_std] # hexadecapole amplitude
	
	# Return all output
	return Adat,fsdat,fcdat,Afsdat,Afcdat,fracdat,angdat,allMC,[],HexPowdat,trueA # (empty set to avoid reordering later code)

def debiased_wrap(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,l_step=a.l_step,lMin=a.lMin,lMax=a.lMax,rot=a.rot,freq=a.freq,\
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors,f_dust=a.f_dust):
	""" Compute the estimated angle, amplitude and polarisation fraction with noise, using zero-padding.
	Noise model is from Hu & Okamoto 2002 and errors are estimated using MC simulations, which are all saved.
	
	Input: map_id (tile number)
	padding_ratio (ratio of real-space padded tile width to initial tile width)
	map_size (tile width in degrees)
	sep (separation of tile centres in degrees)
	N_sims (number of MC simulations)
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
	
	Output: First 6 values: [estimate,isotropic mean, isotropic stdev] for {A,Afs,Afc,fs,fc,str,ang}
	7th: full data for N_sims as a sequence of 7 lists for each estimate (each of length N_sims)
	8th: full data for N_sims for the monopole bias term
	9th: [estimate,isotropic mean, isotropic stdev] for Hexadecapole power
	10: true monopole (from noiseless simulation) - for testing
	"""
	
	# First compute B-mode map from padded-real space map with desired padding ratio
	from .PaddedPower import MakePowerAndFourierMaps
	Bpow,fBdust=MakePowerAndFourierMaps(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq)
	
	# Reduce dust amplitude by 'dedusting fraction'
	print f_dust
	Bpow.powerMap*=f_dust**2. # squared since in power space - this is power due to ONLY dust
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
		
	from .RandomField import fill_from_model
	noiselensedMap=Bpow.copy() # template
	fourierNoise=fBdust.copy() # template
	
	noiselensedMap.powerMap,fourierNoise.kMap=fill_from_model(Bpow,total_Cl_noise,fourier=True)
		
	# Compute total map
	totFmap=fBdust.copy()
	totFmap.kMap+=fourierNoise.kMap# for B modes
	
	# Now convert to power-space
	totPow=fftTools.powerFromFFT(totFmap) # total power map
	
	#totMap2=Bpow.copy() # old depracated method
	#totMap2.powerMap=Bpow.powerMap+noiselensedMap.powerMap
	
			
	# Compute true amplitude using ONLY dust map
	from .hex_wrap import low_dust_estimator
	p=low_dust_estimator(Bpow.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,factor=None,FWHM=0.,\
			noise_power=1.e-400,rot=rot,delensing_fraction=0.,useTensors=False,debiasAmplitude=False)
	trueA=p[0]
			
	
	# Compute rough semi-analytic C_ell spectrum
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)
	
	## Testing
	if a.hexTest:
		print 'testing'
		# test using a known power spectrum which replaces the data
		testmap=totPow.copy()
		from .RandomField import fill_from_model
		def trial_mod(ell):
			return analytic_model(ell,1.0e-13,2.42)
		testmap.powerMap=fill_from_model(Bpow.copy(),trial_mod,fourier=False)
		totPow=testmap # replace data by MC map for testing
					
		
	from .hex_wrap import low_dust_estimator
	A_est,fs_est,fc_est,Afs_est,Afc_est,finalFactor=low_dust_estimator(totPow.copy(),map_id,lMin=lMin,\
		lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,rot=rot,\
		delensing_fraction=delensing_fraction,useTensors=a.useTensors,debiasAmplitude=True)
	# (Factor is expected monopole amplitude (to speed convergence))
	# NB: Afs, Afc are the fundamental quantities here
	
	#H_est=np.sqrt(Afs_est**2.+Afc_est**2.) # this is the hexadecapole estimated from the anisotropic DATA
	
	## Run MC Simulations	
	
	# Compute spline fit to all model data
	#if True:
	#	goodPix=np.where((totMap.modLMap<lMax)&(totMap.modLMap>lMin))
	#	ells=totMap.modLMap[goodPix].ravel()
	#	pows=totMap.powerMap[goodPix].ravel()
	#	from scipy.interpolate import UnivariateSpline
	#
	#	spl=UnivariateSpline(ells,np.log10(pows),k=3)
	#	
	#	def spl_fit(l):
	#		return 10.**spl(l)
	
	# Compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totPow.copy(),lMin,lMax,l_step,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 
	# gives central binning l and mean power in annulus using window function corrections 
	
		
	# Initialise arrays
	A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow2_MC=[np.zeros(N_sims) for _ in range(8)]
	HexPow2_old=np.zeros(N_sims)
	
	#from hades.NoisePower import single_MC
	from .RandomField import fill_from_Cell
	MC_map=Bpow.copy() # template for SIM-SIM data
	MC_fourier=totFmap.copy() # template for Fourier-space data
	
	for n in range(N_sims): # for each MC map
		if n%50==0:
			print('MapID %s: Starting simulation %s of %s' %(map_id,n+1,N_sims))
		# Create the map with a random implementation of Cell, returning both power map and Fourier space map
		_,MC_fourier.kMap=fill_from_Cell(totPow.copy(),l_cen,mean_pow,fourier=True)
		
		## NEW TESTING
		#_,MC_fourier.kMap=fill_from_model(totPow.copy(),total_Cl_noise,fourier=True)
		
		# Now compute the 2D cross-power spectrum of data and MC map
		MC_cross=fftTools.powerFromFFT(MC_fourier.copy(),totFmap.copy())
		MC_map=fftTools.powerFromFFT(MC_fourier.copy())
		#return MC_map,MC_cross
		# Now use the estimators on the SIM-SIM data
		output=low_dust_estimator(MC_map.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
			rot=rot, delensing_fraction=delensing_fraction,useTensors=a.useTensors,debiasAmplitude=True) 
		# compute MC anisotropy parameters  
		A_MC[n]=output[0]
		fs_MC[n]=output[3]/output[0]
		fc_MC[n]=output[4]/output[0]
		Afs_MC[n]=output[3]
		Afc_MC[n]=output[4]
		epsilon_MC[n]=np.sqrt((output[3]**2.+output[4]**2.)/(output[0]**2.))#fc_MC[-1]**2.+fs_MC[-1]**2.))
		ang_MC[n]=0.25*180./np.pi*np.arctan(output[3]/output[4])
		HexPow2_SimSim=output[3]**2.+output[4]**2.
		
		# Now use the estimators on the DAT-SIM data
		output2=low_dust_estimator(MC_cross.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
			rot=rot, delensing_fraction=delensing_fraction,useTensors=a.useTensors,debiasAmplitude=False) 
		# compute MC anisotropy parameters  
		HexPow2_Cross=output2[3]**2.+output2[4]**2.
		
		
		# Now compute corrected squared hexadecapole power
		HexPow2_MC[n]=HexPow2_Cross*4.-HexPow2_SimSim
		HexPow2_old[n]=HexPow2_SimSim
		
		# TESTING
		#A2_Cross=output2[0]**2.
		#A_MC[n]=A2_Cross*4.-A_MC[n]**2.
		#A_est=A_est**2.
		
	#return HexPow2_MC,HexPow2_old,Afs_est**2.+Afc_est**2.
	# NB: we use SQUARED stats for HexPow
		
	allMC=[A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow2_MC]
	
	# Compute anisotropy fraction and angle from data
	ang_est=0.25*180./np.pi*(np.arctan(Afs_est/Afc_est)) # in degrees (already corrected for rotation)
	frac_est=np.sqrt(fs_est**2.+fc_est**2.)
	HexPow2_est=Afs_est**2.+Afc_est**2. # biased hexadecapolar power
	
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
	return Adat,fsdat,fcdat,Afsdat,Afcdat,fracdat,angdat,allMC,[],HexPow2dat,trueA # (empty set to avoid reordering later code)



def hex_fullsky_stats(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,\
	freq=a.freq,delensing_fraction=a.delensing_fraction,makePlots=False):
	""" Function to create plots for each tile. These use hexadecapolar power mostly
	Plots are saved in the CorrectedMaps/ directory. This is designed to read masked full sky data such as from cutoutAndSimulate.py """
	
	raise Exception('Update to include H^2 stats')
	import matplotlib.pyplot as plt
	from scipy.stats import percentileofscore
	import os
	
	# Import good map IDs
	goodDir=a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep)
	if not os.path.exists(goodDir):
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids()
		print 'creating good IDs'
	goodMaps=np.load(goodDir)
	
	# Define arrays
	A,Afs,Afc,fs,fc,ang,frac,probA,probP,logA,epsDeb,HPow,HPowDeb,HPowSig,HprobA,HprobP,logHPowMean=[np.zeros(len(goodMaps)) for _ in range(17)]
	logHexPow,logHexPowDeb,A_err,Af_err,f_err,ang_err,frac_err,frac_mean=[np.zeros(len(goodMaps)) for _ in range(8)]
	
	# Define output directories:
	outDir=a.root_dir+'CorrectedMaps/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
	
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	
	# Iterate over maps:
	for i in range(len(goodMaps)):
		map_id=goodMaps[i] # map id number
		if i%100==0:
			print 'loading %s of %s' %(i+1,len(goodMaps))
		# Load in data from tile
		data=np.load(a.root_dir+'PaddedBatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,map_id))
		
		# Load in data
		A[i],fs[i],fc[i],Afs[i],Afc[i],frac[i],ang[i]=[d[0] for d in data[:7]] # NB: only A, Afs, Afc, ang are useful
		# NB: A is already debiased
		if A[i]>0:
			logA[i]=np.log10(A[i])
		else:
			logA[i]=1000. # to avoid errors - treat these separately
			
		A_err[i],fs_err,fc_err,Afs_err,Afc_err,frac_err[i]=[d[2] for d in data[:6]]
		frac_mean[i]=data[5][1]
		
		# Compute debiased H
		HPow[i]=data[9][0]
		HPow_MC=data[7][7]
		logHPowMean[i]=np.log10(np.mean(HPow_MC))
		HPowDeb[i]=data[9][0]-data[9][1]
		HPowSig[i]=(data[9][0]-data[9][1])/data[9][2] # not very useful
		logHexPow[i]=np.log10(HPow[i])
		if HPowDeb[i]>0.:
			logHexPowDeb[i]=np.log10(HPowDeb[i])
		else:
			logHexPowDeb[i]=1000.
		
		# Compute other errors
		f_err[i]=np.mean([fs_err,fc_err]) # not useful
		Af_err[i]=np.mean([Afs_err,Afc_err])
		ang_err[i]=Af_err[i]/(4*HPow[i])*180./np.pi # error in angle
		
		# Compute statistics for Hexadecapolar power
		HprobP[i]=percentileofscore(HPow_MC,HPow[i],kind='mean') # statistical percentile of estimated data
		def H_CDF(H,sigma_Af):
			"""CDF of HexPow modified chi2 distribution"""
			return 1 - np.exp(-H**2./(2.*sigma_Af**2.))
		#def H_PDF(H,sigma_Af):
		#	"""PDF of HexPow modified Chi2 distribution"""
		#	return H/(sigma_Af**2.)*np.exp(-H**2./(2.*sigma_Af**2.))
		
		# Compute analytic CDF percentile:
		HprobA[i]=100.*H_CDF(HPow[i],Af_err[i])
		
		
		# Now repeat for epsilon - NB: this uses the debiased monopole amplitudes implicitly
		eps=data[7][5] # all epsilon data
		eps_est=data[5][0]
		epsDeb[i]=eps_est-np.mean(eps)
						
		percentile=percentileofscore(eps,eps_est,kind='mean') # compute percentile of estimated data
		probP[i]=percentile
		sigma_f=np.mean([data[1][2],data[2][2]]) # mean of fs,fc errors
		def eps_CDF(eps):	
			""" CDF of epsilon modified chi-squared distribution)"""
			return 1-np.exp(-eps**2./(2.*sigma_f**2.))
		def eps_PDF(eps):
			""" PDF of epsilon modified chi-squared distribution"""
			return eps/(sigma_f**2.)*np.exp(-eps**2./(2.*sigma_f**2.)) 
		# Compute analytic CDF percentile:
		probA[i]=100.*eps_CDF(eps_est)
		
		
	## Now compute the whole patch maps
	# First define dataset:
	dat_set = [HPow,HPowDeb,HPowSig,logHPowMean,logHexPow,logHexPowDeb,A,frac,epsDeb,ang,ang_err,HprobA,HprobP,probA,probP,logA]
	names = [r'Hexadecapole Amplitude',r'Hexadecapole Debiased Amplitude',r'Hexadecapole "Significance"',r'log(Hexadecapole MC Isotropic Mean)',\
		r'log(Hexadecapole Amplitude)',r'Log(Hexadecapole Debiased Amplitude)',r'Debiased Monopole Amplitude',r'Anisotropy Fraction',r'Debiased Epsilon',\
		r'Anisotropy Angle',r'Anisotropy Angle Error',r'Hexadecapole Percentile (Analytic)',r'Hexadecapole Percentile (Statistical)',\
		r'Epsilon Percentile (Analytic)',r'Epsilon Percentile (Statistical)','log(Monopole Amplitude)']
	file_str =['hexPow','hexPowDeb','hexPowSig','HexPowMeanMC','logHexPow','logHexPowDeb','Adebiased','eps','epsDeb','ang','angErr','hexProbAna','hexProbStat',\
		'epsProbAna','epsProbStat','logA']
	
	if len(names)!=len(file_str) or len(names)!=len(dat_set):
		raise Exception('Wrong length of files') # for testing
		
	# Load coordinates of map centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,len(goodMaps))
	# Load in border of BICEP region if necessary:
	border=False
	if a.root_dir=='/data/ohep2/WidePatch/' or a.root_dir=='/data/ohep2/CleanWidePatch/' or a.root_dir=='/data/ohep2/FFP8/':
		border=True # for convenience
		from hades.plotTools import BICEP_border
		temp=BICEP_border(map_size,sep)
		if temp!=None:
			edge_ra,edge_dec=temp
			# to only use cases where border is available
		else:
			border=False
	if border!=False:
		border_coords=[edge_ra,edge_dec]
	else:
		border_coords=None # for compatibility
	
	# Now plot on grid:
	from hades.plotTools import mollweide_map
	import cmocean # for angle colorbar
	for j in range(len(names)):
		print 'Generating patch map %s of %s' %(j+1,len(names))
		cmap='jet'
		minMax=None
		if file_str[j]=='ang':
			cmap=cmocean.cm.phase
		if file_str[j]=='angErr':
			minMax=[0.,45.]
		if file_str[j]=='epsDeb':
			minMax=[-5,5]
		if file_str[j]=='eps':
			vmin=min(dat_set[j])
			vmax=min([1.,np.percentile(dat_set[j],95)])
			minMax=[vmin,vmax]
		decA,raA=dec.copy(),ra.copy()
		if file_str[j]=='logA':
			ids=np.where(dat_set[j]!=1000.)
			dat_set[j]=dat_set[j][ids]
			raA=raA[ids]
			decA=decA[ids]
		if file_str[j]=='logHexPowDeb':
			ids=np.where(dat_set[j]!=1000.)
			dat_set[j]=dat_set[j][ids]
			raA=raA[ids]
			decA=decA[ids]
		# Create plot
		mollweide_map(dat_set[j],raA,decA,cbar_label=names[j],cmap=cmap,minMax=minMax,\
			border=border_coords,outFile=outDir+file_str[j]+'.png',decLims=[-90,90,10],raLims=[-180,180,10])
	print 'Plotting complete'


def patch_hexadecapole(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,\
			freq=a.freq,delensing_fraction=a.delensing_fraction,N_sims=a.N_sims,suffix='',folder=None,\
			root_dir=a.root_dir,I2SNR=a.I2SNR,returnAll=False,plot=True,monopole=False):
	"""Compute the global hexadecapole anisotropy over the patch, summing the epsilon values weighted by the S/N.
	The estimate is assumed Gaussian by Central Limit Theorem.
	Errors are obtained by computing estimate for many MC sims
	
	returnAll parameter returns mean,std,estimate H^2 values
	plot parameter controls whether to create H^2 histogram
	"""
	# Load array of map ids
	import os
	
	goodDir=root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep)
	if not os.path.exists(goodDir):
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids(map_size=map_size,sep=sep,root_dir=root_dir)
		print 'creating good IDs'
	goodMaps=np.load(root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	
	if I2SNR:
		I2dir = root_dir+'%sdeg%s/meanI2.npy' %(map_size,sep)
		QUdir = root_dir+'%sdeg%s/meanQU.npy' %(map_size,sep)
		import os
		if not os.path.exists(I2dir):
			raise Exception('Must compute <I^2> data by running batchI2.py')
		I2data=np.load(I2dir)
		QUdata=np.load(QUdir)
	
	# Initialize variables
	Asum,trueAsum=0.,0.
	count=0
	hex2_patch_num=0.
	A_patch_num=0. # for mean amplitude shift
	norm=0. # normalisation
	hex2_patch_MC_num=np.zeros(N_sims)
	A_patch_MC_num=np.zeros(N_sims)
	h2_est,h2_mean,h2_eps,bias_data=[],[],[],[]
	for i in range(len(goodMaps)):
		# Load dataset
		if root_dir=='/data/ohep2/liteBIRD/':
			ij=goodMaps[i]
		else:
			ij=i # for compatibility
		if folder==None:
			if a.true_lensing:
				folder='LensedPaddedBatchData'
			else:
				folder='PaddedBatchData'
		datPath=root_dir+folder+'/f%s_ms%s_s%s_fw%s_np%s_d%s/%s%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,ij,suffix)
		if root_dir=='/data/ohep2/liteBIRD/':
			if not os.path.exists(datPath):
				continue
		data=np.load(datPath)
		A=data[0][1]	
		A_est=data[0][0]
		A_MC=data[7][0]
		A_eps=data[0][2]	
		hex2_est=data[9][0]
		hex2_MC=data[7][7]
		hex2_mean=data[9][1]
		hex2_eps=data[9][2]
		trueA=data[10]
		Asum+=A_est
		trueAsum+=trueA
		count+=1
		if returnAll:
			h2_est.append(hex2_est)
			h2_mean.append(hex2_mean)
			h2_eps.append(hex2_eps)
			bias_data.append(data[11])
		
		# Compute contribution to mean epsilon
		if I2SNR:
			SNR=1.#(trueA**2./hex2_eps)**2.#/A_est#(hex2_mean/hex2_eps)**2.#(data[0][1]/data[0][2])**2.#/hex_eps**2.#I2data[i]**2./hex_eps**2.
		else:
			SNR=1.#(trueA/hex_eps)**2.
		hex2_patch_num+=SNR*(hex2_est)#-hex2_mean)
		A_patch_num+=SNR*A_est
		norm+=SNR
		for j in range(N_sims):
			hex2_patch_MC_num[j]+=SNR*(hex2_MC[j])#-hex2_mean)
			A_patch_MC_num[j]+=SNR*A_MC[j]
			
	# Compute mean epsilon + MC values
	hex2_patch=hex2_patch_num/norm
	hex2_patch_MC=hex2_patch_MC_num/norm
	A_patch=A_patch_num/norm
	A_patch_MC=A_patch_MC_num/norm
	
	# Compute monopole mean
	trueA=trueAsum/count
	A=Asum/count
	# Compute mean and standard deviation
	MC_mean=np.mean(hex2_patch_MC)
	MC_std=np.std(hex2_patch_MC)
	A_MC_mean=np.mean(A_patch_MC)
	A_MC_std=np.std(A_patch_MC)
	
	# Compute significance of detection
	if a.useBias:
		sigmas=(hex2_patch)/MC_std#-MC_mean)/MC_std
	else:
		sigmas=(hex2_patch-MC_mean)/MC_std # remove bias here
	sigmasA=(A_patch-A_MC_mean)/A_MC_std
	
	if plot:
		# Now plot
		import matplotlib.pyplot as plt
		y,x,_=plt.hist(hex2_patch_MC,bins=30,normed=True)
		plt.ylabel('PDF')
		plt.xlabel('Patch Averaged Bias-corrected H^2')
		plt.title('%.2f Sigma // Patch Averaged Corrected H^2 // %s patches & %s sims' %(sigmas,len(goodMaps),N_sims))
		xpl=np.ones(100)*hex2_patch
		ypl=np.linspace(0,max(y),100)
		plt.plot(xpl,ypl,ls='--',c='r')
		plt.ylim(0,max(y))
		outDir=root_dir+'PatchHexCorrectedPow/'
		import os
		if not os.path.exists(outDir):
			os.makedirs(outDir)
		plt.savefig(outDir+'hist_f%s_ms%s_s%s_fw%s_np%s_d%s.png' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction),bbox_inches='tight')
		plt.clf()
		plt.close()
		
	if returnAll:
		return np.mean(h2_mean),np.mean(h2_eps),np.mean(h2_est),sigmas,sigmasA,np.mean(bias_data)
	else:
		if monopole:
			#print sigmas,sigmasA,A,trueA
			return sigmas,sigmasA,A,trueA
		else:
			print sigmas,sigmasA
			return sigmas,sigmasA
