import numpy as np
import sys
sys.path.append('/data/ohep2/')

class BICEP:
	""" Class containing (mostly) essential model parameters for KK estimators using BICEP data"""
	
	# Root directory
	root_dir = '/data/ohep2/FFP8/'#'/data/ohep2/{Simons/,sims/,BICEP2/,WidePatch/,FFP8/,FullSky/,liteBIRD/,CleanWidePatch/}'# root directory for simulations
	
	# Tile parameters
	map_size =  3 # Width of each map
	sep = 3# Separation of map centres
	
	N_sims = 500 # Number of MC sims
	N_bias = 500 # no. sims used for bias computation
	freq = 150 # Frequency of simulation in GHz (353 is Vansyngel, 150 is BICEP)
	padding_ratio=2. # padded map width / original map width 
	unPadded=False # do not apply any zero-padding if true
	
	useQU=False # use root(Q^2+U^2) maps for dedusting - else use I maps
	rTest = False # replace data with r= 0.1 spectrum
	
	rescale_freq = True # rescale to correct frequency - turn OFF for rTests etc.
	
	KKdebiasH2 = False # subtract expected noise spectrum for Afs, Afc in estimators
	
	log_noise = False # use log scaling for noise - only for large noise levels
	
	# Estimator parameters
	l_step=400./map_size*1./padding_ratio # pixel size is 360/map_size/padding_ratio)
	if padding_ratio==1:
		lMin=180.*3./map_size
	else:
		lMin=180.*3./map_size*1./padding_ratio #240.*3./map_size*1./padding_ratio
	lMax = 2000. # ranges to fit spectrum over 
	f_dust=1.

	# Planck dust SED (XXII/LIV papers)
	dust_temperature = 19.6
	dust_spectral_index = 1.53 # Planck Intermediate LIV
	reference_frequency = 353 # in GHz for input sims
	
	delensing_fraction = 0.1 # efficiency of delensing -> 1 = no delensing
	
	# Noise model
	## Defaults:
	# BICEP: (30,5)
	# S4: (1.5,1) or (30,1)
	# No noise: (0,1e-30) (for stability)
	# liteBIRD: (30,3)
	if root_dir=='/data/ohep2/liteBIRD/':
		FWHM=30.
		noise_power=3.
		lMax=1000
		lMin=90.
		l_step=60.
		padding_ratio=2.
		unPadded=False#True
		delensing_fraction=0.5 
		slope=2.42 # C_l^f power law slope (from Planck X.X.X. paper)
		flipU=True # if using COSMO polarisation convention, reverse sign of U for compatibility with flipper (else IAU convention)
	elif root_dir=='/data/ohep2/Simons/':
		FWHM=1.8
		noise_power=5.
		delensing_fraction=0.4
		slope=2.42
		flipU=True # if using COSMO polarisation convention, reverse sign of U for compatibility with flipper (else IAU convention)
	elif root_dir=='/data/ohep2/FFP8/':
		FWHM=1.5
		noise_power=1.
		delensing_fraction=0.1
		slope=2.42
		flipU=False # if using COSMO polarisation convention, reverse sign of U for compatibility with flipper (else IAU convention)
	else:
		FWHM = 1.5 # full width half maximum of beam in arcmin
		noise_power = 1.# noise power of S4 -> in microK-arcmin
		delensing_fraction=0.1
		slope=2.42
		flipU=True # if using COSMO polarisation convention, reverse sign of U for compatibility with flipper (else IAU convention)
	
	# Lensing
	lensedDir='/data/ohep2/CAMB_lensedCl.npz'
	CAMBrDir='/data/ohep2/CAMB_r.npz' # for r = 0.1
	rot_average = True # pre-rotate to correct for pixellations
	
	# NoiseParamsSpace parameters
	remakeErrors=True
	noi_par_NoisePower=np.linspace(0.1,5.1,20)#'0.2) #)np.arange(1e-30,2.1,0.16)#np.arange(1e-30,5.1,0.4)
	noi_par_FWHM=np.linspace(0,31.,20)#1.25)#np.arange(0,11,0.8)#np.arange(0,31,2.5)
	noi_par_delensing=[0.1,1.0]
	
	# Null testing parameters
	f_dust_all = list(np.arange(1.0,-0.05,-0.05))#np.arange(1,-0.2,-0.2)##[1e-6]+list(np.logspace(-3,0,30))#np.arange(1.,-0.05,-0.05) [1e-10,1.0]#[1e-10,1.0]#
	err_repeats = 10#0 # repeat for uncertainties
	
	## OTHER TESTING PARAMETERS
	hexTest = False#True # test methods using fake isotropic map
	rot=11.25 # pre rotation before applying estimators
	useBias=True # correct for SIM-SIM - DATA-SIM bias
	useTensors=False #True # include r = 0.1 tensor modes
	I2SNR = False # use <I^2> to estimate the SNR for patch anisotropy measurements
	debiasA = True # for debiasing of monopole amplitude using noise only sims
	exactCen = True#True # for computing centre of one-D bins
	useLensing = True # DEPRACATED: if False, just set delensing_fraction=0.
	KKmethod=False # if False, apply Sherwin SNR ratio not KK SNR
	send_email=False # send email on completion
	repeat=1#50#200 # repeat application of noise + estimation to see errors
	
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
	
	
