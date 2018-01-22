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
	from hades.hex_wrap import estimator_wrap
	output=estimator_wrap(map_id)
	
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
    slope=a.slope,factor=None,rot=0.,delensing_fraction=a.delensing_fraction,useTensors=a.useTensors):
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
        Afactor=1e-13 # initial estimate

        while N<10: # max no. iterations
            goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax)) # correct pixels
            lMap=map.modLMap.ravel()[goodPix] # mod(l)
            PowerMap=map.powerMap.ravel()[goodPix] # B-mode (biased) power
            OtherClMap=Cl_total_noise_func(lMap) # extra C_l power
            debiasedPowerMap=PowerMap-OtherClMap # power estimate only due to dust
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

        if N==10:
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
    debiasedPowerMap=PowerMap-OtherClMap # power estimate only due to dust
    fiducialClMap=lMap**(-slope) # fiducial Cl
    
    SNmap=(finalFactor*fiducialClMap)/(finalFactor*fiducialClMap+OtherClMap) # signal-to-noise ratio
    
    # Now compute estimates for A, Afs, Afc
    Anum=np.sum(debiasedPowerMap*(SNmap**2.)/fiducialClMap)
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
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors):
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
	
	Output: First 6 values: [estimate,isotropic mean, isotropic stdev] for {A,Afs,Afc,fs,fc,str,ang}
	7th: full data for N_sims as a sequence of 7 lists for each estimate (each of length N_sims)
	8th: full data for N_sims for the monopole bias term
	9th: [estimate,isotropic mean, isotropic stdev] for Hexadecapole power
	10: true monopole (from noiseless simulation) - for testing
	"""
	
	# First compute high-resolution B-mode map from padded-real space map with desired padding ratio
	from .PaddedPower import MakePaddedPower
	Bpow=MakePaddedPower(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq)
	
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

	noiselensedMap.powerMap=fill_from_model(Bpow,total_Cl_noise)
		
	# Compute total map
	totMap=Bpow.copy()
	totMap.powerMap=Bpow.powerMap+noiselensedMap.powerMap
			
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
		## Testing
		if a.hexTest:
			testmap=totMap.copy()
			from .RandomField import fill_from_model
			def trial_mod(ell):
				return analytic_model(ell,3.0e-13,2.42)
			testmap.powerMap=fill_from_model(Bpow.copy(),trial_mod)
			totMap=testmap # replace data by MC map for testing
				
		# Apply the KK estimators 
		from .hex_wrap import low_dust_estimator
		A_est,fs_est,fc_est,Afs_est,Afc_est,finalFactor=low_dust_estimator(totMap.copy(),map_id,lMin=lMin,\
			lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,rot=rot,\
			delensing_fraction=delensing_fraction,useTensors=a.useTensors)
		# (Factor is expected monopole amplitude (to speed convergence))
		# NB: Afs, Afc are the fundamental quantities here
	
		H_all.append(np.sqrt(Afs_est**2.+Afc_est**2.))
	
	## Run MC Simulations	
	
	# Compute spline fit to all model data
	goodPix=np.where((totMap.modLMap<lMax)&(totMap.modLMap>lMin))
	ells=totMap.modLMap[goodPix].ravel()
	pows=totMap.powerMap[goodPix].ravel()
	from scipy.interpolate import UnivariateSpline
	
	spl=UnivariateSpline(ells,np.log10(pows),k=3)
		
	def spl_fit(l):
		return 10.**spl(l)
	
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
		if n%50==0:
			print('MapID %s: Starting simulation %s of %s' %(map_id,n+1,N_sims))
		# Create the map with a random implementation of Cell
		MC_map.powerMap=fill_from_Cell(totMap,l_cen,mean_pow)
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

