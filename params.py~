import numpy as np
import sys
sys.path.append('/data/ohep2/')

class BICEP:
	""" Class containing (mostly) essential model parameters for KK estimators using BICEP data"""
	
	# Root directory
	root_dir = '/data/ohep2/WidePatch/' # '/data/ohep2/{sims/,BICEP2/,WidePatch/}'# root directory for simulations
	
	# Tile parameters
	map_size =  3 # Width of each map
	sep = 3  # Separation of map centres
	padding_ratio=1. # padded map width / original map width - no padding here
	
	N_sims = 200 # Number of MC sims
	freq = 150 # Frequency of simulation in GHz (353 is Vansyngel, 150 is BICEP)
	
	# Estimator parameters
	l_step = 120.*3./map_size #120.# width of binning in l-space for power spectra
	lMin = 120.*3./map_size #120.
	lMax = 2000. # ranges to fit spectrum over 
	rot=11.25 # pre rotation before applying estimators
	
	# Planck XXII parameters for dust SED
	dust_temperature = 19.6
	dust_spectral_index = 1.59
	reference_frequency = 353 # in GHz for Vansyngel sims
	
	# Noise model
	## Defaults:
	# BICEP: (30,5)
	# S4: (1.5,1) or (30,1)
	# No noise: (0,1e-30) (for stability)
	FWHM = 1.5 # full width half maximum of beam in arcmin
	noise_power = 1. # noise power of BICEP -> in microK-arcmin
	
	# Lensing
	lensedDir='/data/ohep2/CAMB_lensing.npz'
	delensing_fraction = 1. # efficiency of delensing -> 1 = no delensing
	
	# Fiducial C_l 
	slope=2.42 # C_l^f power law slope (from Planck X.X.X. paper)
	
	
	## OTHER TESTING PARAMETERS
	exactCen = True#True # for computing centre of one-D bins
	useLensing = True # DEPRACATED: if False, just set delensing_fraction=0.
	KKmethod=False # if False, apply Sherwin SNR ratio not KK SNR
	
	# Rotation angles for KK map rotation (to avoid pixellation errors)
	rotation_angles=np.arange(0,22.5,0.9)
	
	# If run analyis for different noise powers - DEPRACATED
	NoiseAnalysis = False#True# 
	ComparisonSetting='noise_power'#'noise_power'# must be in ['FWHM','noise_power']
	NoisePowerLists = np.logspace(0,2,5)#np.logspace(-2,2,5) # 0.01 to 100 range
	FWHM_lists=[0, 10, 20, 30, 40, 50]
	
	# For parameter space analysis
	param_space_noise_powers=[1e-20,1,3,5,10,20]
	param_space_FWHMs=[0.,1,8,15,22,30]
	param_space_lMin=np.arange(50,500,50)
	
	# Hyperparameter study
	hyp_lMin=np.arange(120,481,120)
	hyp_slope=np.arange(2.2,3.0,0.2)
	hyp_map_size=np.array([2,3,5])
	
	# NoiseParamsSpace parameters
	noi_par_NoisePower=np.arange(1e-30,5.1,0.4)
	noi_par_FWHM=np.arange(0,31,2.5)
