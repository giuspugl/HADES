def rotation_est(map_id,map_size=a.map_size,sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,slope=a.slope):
	""" Compute the estimated angle, amplitude and polarisation strength in the presence of noise, following Hu & Okamoto 2002 noise model. Error is from MC simulations. This averages the estimated quantities over map rotations to avoid errors.
	Output: list of data for A, fs, fc (i.e. output[0]-> A etc.), with structure [map estimate, MC_standard_deviation, MC_mean]
	"""
	# First calculate the B-mode map (noiseless)
	from .PowerMap import MakePower
	Bpow=MakePower(map_id,map_size=map_size,map_type='B')
	
	# Load the relevant window function
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
	mask=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
	
	# Compute mean square window function
	windowFactor=np.mean(mask.data**2.)
	
	# Now compute the noise power-map
	from .NoisePower import noise_map
	noiseMap=noise_map(powMap=Bpow.copy(),noise_power=a.noise_power,FWHM=a.FWHM,windowFactor=windowFactor)
	
	# Compute total map
	totMap=Bpow.copy()
	totMap.powerMap=Bpow.powerMap+noiseMap.powerMap
	
	# Initially using NOISELESS estimators
	from .KKtest import rotation_estimator
	est_data=rotation_estimator(totMap.copy(),slope=slope) # compute anisotropy parameters
		
	## Run MC Simulations	
	# First compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totMap.copy(),0.8*a.lMin,1.2*a.lMax,a.l_step,binErr=False,windowFactor=windowFactor) # gives central binning l and mean power in annulus using window function corrections
	
	# Compute univariate spline model fit to 1D power spectrum
	from scipy.interpolate import UnivariateSpline
	spline_fun = UnivariateSpline(np.log10(l_cen),np.log10(mean_pow),k=4) # compute spline of log data
	
	def model_power(ell):
		return 10.**spline_fun(np.log10(ell)) # this estimates 1D spectrum for any ell
	
	# Now run MC simulation N_sims times
	ang_MC,frac_MC=[],[]
	
	for n in range(N_sims): # for each MC map
		MC_map=single_MC(totMap.copy(),model_power,windowFactor=windowFactor) # create random map from isotropic spectrum
		p=rotation_estimator(MC_map.copy(),slope=slope) # compute MC anisotropy parameters  
		frac_MC.append(p[1])
		ang_MC.append(p[2])
	
	# Compute mean and standard deviation of MC statistics
	frac_mean=np.mean(frac_MC)
	ang_mean=np.mean(ang_MC)
	sigma_frac=np.std(frac_MC)
	
	
	# Regroup output (as described above)
	output = [est_data[0],est_data[1],est_data[2],frac_mean,ang_mean,sigma_frac]		
	
	return output
