import numpy as np
import sys

class BICEP:
    """ Class containing (mostly) essential model parameters for KK estimators using BICEP data"""
    
    hades_dir = '/home/oliver/HADES/' # directory where HADES is installed
    root_dir = '/home/oliver/hades_testing/' # directory to house all simulation cut-outs + analysis products
    flipU = True # This converts between COSMO [in many input maps] and IAU [used by flipper] polarisation conventions. If true, this reverses the sign of the U-polarisation map which has non-trivial effects on the analysis.
    
    ## Tile parameters
    map_size = 3 # tile (cut-out section) width in degrees
    sep = 3 # separation of the center of each tile. This may be set as less than the map_size to allow higher resolution plots with overlapping tiles but destroys the statistical independence
    padding_ratio = 2. # amount of zero-padding. This is the ratio of padded to unpadded map width. 2 gives best performance.
    rot_average = True # pre-rotate tiles to reduce pixellation error (see paper appendix E).
    rotation_angles=np.arange(0,22.5,0.9) # Angles to pre-rotate tiles by
    
    ## MC parameters
    N_sims = 500 # No. of MC simulations used to create parameter distributions
    N_bias = 500 # No. simulations used for realisation-dependent debiasing

    ## Experimental parameters
    freq = 150 # Desired map frequency in GHz. Currently only 150 GHz (peak BICEP sensitivity) and 353 GHz (original simulations) are implemented - [these rely on non-linear color conversions]
    f_dust = 1. # Scalable dust fraction where f_dust = 1 has no cleaning and f_dust = 0 has 100% dust cleaning efficiency.
    delensing_fraction = 0.1 
    
    ## Noise model [see table 1 of Philcox+2018 for full descriptions]
    experiment_profiles= ['Zero', 'BICEP2', 'Simons', 'S4'] # Loaded experimental profiles
    experiment_profile_index = 0

    if experiment_profiles[experiment_profile_index]=='BICEP2':
        FWHM = 30. # Experimental noise FWHM [theta_FWHM] in arcmin
        noise_power = 5. # Experimental noise_power [delta_P] in uK-arcmin
        delensing_fraction = 1. # Delensing efficiency, where 0.1 implies C_l_lens is reduced by 90%.
    if experiment_profiles[experiment_profile_index]=='Zero': # no noise or lensing
        FWHM = 0.
        noise_power = 1.e-30 # for stability
        delensing_fraction = 0. 
    if experiment_profiles[experiment_profile_index]=='Simons':
        FWHM = 1.8
        noise_power = 5. 
        delensing_fraction = 0.4 
    if experiment_profiles[experiment_profile_index]=='S4':
        FWHM = 1.5
        noise_power = 1.
        delensing_fraction = 0.1 
    
    ## Estimator parameters
    l_step=400./map_size*1./padding_ratio # Bin width in ell-space. [Pixel size is 360/map_size/padding_ratio)]
    lMin=180.*3./map_size*1./padding_ratio # Minimum ell for analysis. 
    lMax = 2000. # Maximum ell for analysis
        
    ## Lensing directories
    lensedDir=hades_dir+'CAMB_Profiles/CAMB_lensedCl.npz' # .npz file containing a 1D CAMB lensing profile
    CAMBrDir = hades_dir+'CAMB_Profiles/CAMB_r.npz' #.npz file containing an r = 0.1 CAMB tensor profile. (Currently unused)
    #lensedDir='/data/ohep2/CAMB_lensedCl.npz' # .npz file containing
    #CAMBrDir='/data/ohep2/CAMB_r.npz' # for r = 0.1
    
    # Dust SED (Planck XXII/LIV papers). This uses experimental parameters from Planck Int. XXII/LIV papers.
    dust_temperature = 19.6 # T_dust in K
    dust_spectral_index = 1.53 # beta_dust
    reference_frequency = 353 # in GHz for input simulations
    slope = 2.42 # Fiducial C_l^{dust} power law slope (from Planck XXX)
    
    ## Null testing parameters [paper figure 2]
    f_dust_all = list(np.arange(1.0,-0.05,-0.05)) # List of values of dust fraction to be tested.
    err_repeats = 10 # Number of times to repeat each data-point (for error-bars)
    
    ## Noise parameter space studies [paper figure 4]
    noi_par_NoisePower=np.linspace(0.1,5.1,20) # noise_power ranges
    noi_par_FWHM=np.linspace(0,31.,20) # noise FWHM values
    noi_par_delensing=[0.1,1.0] # delensing values
    remakeErrors=True

    ########################################################################################
    ## OTHER TESTING PARAMETERS
    unPadded=False # do not apply any zero-padding if true
    useQU=False # use root(Q^2+U^2) maps for dedusting - else use I maps
    rTest = False # replace data with r= 0.1 spectrum
    rescale_freq = True # rescale to correct frequency - turn OFF for rTests etc.
    KKdebiasH2 = True # subtract expected noise spectrum for Afs, Afc in estimators
    log_noise = False # use log scaling for noise - only for large noise levels
    true_lensing = False # use FFP10 lensing spectrum rather than CAMB one
    camb_spectrum = False # rescale FFP10 to CAMB 1D power spectrum for compatibility
    ffp10_spectrum = True # use FFP10 spectrum rather than 1D CAMB one

    hexTest = False#True # test methods using fake isotropic map
    rot=11.25 # pre-rotation before applying estimators
    useBias=True # correct for SIM-SIM - DATA-SIM bias
    useTensors=False #True # include r = 0.1 tensor modes
    I2SNR = False # use <I^2> to estimate the SNR for patch anisotropy measurements
    debiasA = True # for debiasing of monopole amplitude using noise only sims
    exactCen = True#True # for computing centre of one-D bins
    useLensing = True # DEPRACATED: if False, just set delensing_fraction=0.
    KKmethod=False # if False, apply Sherwin SNR ratio not KK SNR
    send_email=False # send email on completion
    repeat=1 # repeat application of noise + estimation to see errors
    
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
    
    
