class BICEP:
	""" Class containing (mostly) essential model parameters for KK estimators using BICEP data"""
	
	root_dir = '/data/ohep2/BICEP2/' # '/data/ohep2/sims/'# root directory for simulations
	
	map_size = 3 # Width of each map
	N_sims = 100#100 # Number of MC sims
	l_step = 'a'#100#100 # width of binning in l-space for power spectra
	lMin = 100
	lMax = 2000 # ranges to fit spectrum over 
	
	
	# Noise model
	FWHM = 10. # full width half maximum of beam in arcmin
	noise_power = 5. # noise power of BICEP -> in microK-arcmin
