import numpy as np
import sys
sys.path.append('/data/ohep2/')

class BICEP:
	""" Class containing (mostly) essential model parameters for KK estimators using BICEP data"""
	
	root_dir = '/data/ohep2/BICEP2/' # '/data/ohep2/sims/'# root directory for simulations
	
	map_size =  3 # Width of each map
	N_sims = 2000 # Number of MC sims
	l_step = 120.#100#100 # width of binning in l-space for power spectra
	lMin = 120.
	lMax = 2000. # ranges to fit spectrum over 
	
	sep = 3#'0.5 # separation of map centres
	
	
	# Noise model
	
	## Defaults:
	# BICEP: (30,5)
	# S4: (1.5,1) or (30,1)
	# No noise: (0,0)
	FWHM = 1.5#1e-10. #'10. # full width half maximum of beam in arcmin
	noise_power = 1.#.1e-5#100.000000#100.000000 # noise power of BICEP -> in microK-arcmin
	
	# Fiducial C_l 
	slope=2.42#3.0 # C_l^f power law slope (from Planck X.X.X. paper)
	
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
	
	# For zero padding
	padding_ratio=1. # padded map width / original map width (NB: padding_ratio=1. recovers unpadded map)
	
	# Hyperparameter study
	hyp_lMin=np.arange(120,481,120)
	hyp_slope=np.arange(2.2,3.0,0.2)
	hyp_map_size=np.array([2,3,5])
	
	# NoiseParamsSpace parameters
	noi_par_NoisePower=np.arange(1e-30,5,0.3)
	noi_par_FWHM=np.arange(0,30,2)
