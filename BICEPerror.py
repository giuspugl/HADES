from .params import BICEP
a=BICEP()

from flipper import *
import numpy as np

def noisy_est_and_err(map_id,map_size=a.map_size,N_sims=a.N_sims,l_step=a.l_step):
	""" Compute the estimated angle, amplitude + polarisation strength of a BICEP map in the presence of noise, following Hu & Okamoto 2002 prescription. Error is from MC simulations."""
	
def noise_model(l,FWHM=a.FWHM,noise_power=a.noise_power):
	""" Noise model of Hu & Okamoto 2002. 
	Inputs: l, 
	Full-Width-Half-Maximum of experiment in arcminutes, 
	noise-power (Delta_P) in microK-arcmin.
	
	Unit conversions are done to put output in K^2 
	-> to match input maps in K_CMB. """
	# NB: Check dimensions
	
	#T_CMB = 2.728e6 # in microK
	FWHM_rad=FWHM/60.*np.pi/180. # in radians
	noise_power_rad_K=1.0e-6*noise_power/60.*np.pi/180. # in K-rad'
	
	exponent=l*(l+1)*FWHM_rad**2 / (8.*np.log(2))
	Cl_K = (noise_power_rad)**2*np.exp(exponent) # Cl is in K^2-rad
	return Cl_K

def est_and_err(map_id,map_size=a.map_size,N_sims=a.N_sims,l_step=a.l_step):
    """ Compute estimated angle, amplitude and polarisation of map compared to Gaussian error from MC simulations.
    Input - map_id, map_size
    N_sims = no. of MC sims
    l_step = bin width for computing slope of B-space map

    Out: best estimator and error as a list in order (strength,angle,amplitude).
    Yanked from MCerror module and modified for BICEP
    """
    from .KKtest import angle_estimator
    from .PowerMap import MakePower

    ## First calculate the binned power spectrum across all l for the map
    # Load in the map
    Bmap=MakePower(map_id,map_size=map_size,map_type='B')

    # Define arrays
    l_bin = []
    pow_mean = []
    pow_std = []
    l_min=a.lMin#l_step # don't include l=0
    l_max=a.lMax # max/min of map

    # Compute power
    #print('Computing Full Power Spectrum')
    
    if True: # new method -> treat every data point separately
    	lvals=[] # each mod(l) value
	powervals=[] # each required power value
	for i in range(len(Bmap.modLMap)): 
		# filter only values in correct l range (at start and end of row only)
    		N=len(Bmap.modLMap[i])
    		for j in range(N): 
        		if Bmap.modLMap[i,j]<a.lMax and Bmap.modLMap[i,j]>a.lMin:
            			lvals.append(Bmap.modLMap[i,j])
           			powervals.append(Bmap.powerMap[i,j])
       			else:
            			break
    		for j in range(N): # to get all required values
        		if Bmap.modLMap[i,N-j-1]<a.lMax and Bmap.modLMap[i,N-j-1]>a.lMin:
            			lvals.append(Bmap.modLMap[i,N-j-1])
            			powervals.append(Bmap.powerMap[i,N-j-1])
        		else:
            			break
        # optimization
    	from scipy.optimize import curve_fit
    	def fun(l,log10A,gamma):
    		return log10A-gamma*np.log10(l)
	
	# Compute A and gamma values
	p,_=curve_fit(fun,lvals,np.log10(powervals))
	A=10.**p[0]
	slope=p[1] # fed into MC maps as the fiducial slope
	
	#print A,slope
	
	l_bin=np.arange(a.lMin,a.lMax,5)
	pow_mean=10.**fun(l_bin,A,slope)
    
    if False: # old method binning in annuli
    	print 'old'
    	for l in np.arange(l_min,l_max,l_step):
        	mean,std,pix=Bmap.meanPowerInAnnulus(l,l+l_step)
        	pow_mean.append(mean)
        	pow_std.append(std)
        	l_bin.append(l+0.5*l_step)
    	# Compute the map slope for the utilised l range
    	from .PowerMap import MapSlope
    	slope,_,_=MapSlope(Bmap,l_min=l_min,l_max=l_max,l_step=l_step)
    	# This is then fed into the MC maps as the fiducial slope
    

    # Use KK estimators to find best estimate of angle and polarisation strength
    #print('Computing best KK estimators')
    p_str,p_ang,A,fs,fc,Afs,Afc=angle_estimator(map_id,map_size,map=Bmap,l_step=l_step,lMin=l_min,lMax=l_max,slope=slope)
    best_preds=[p_str,p_ang,A,fs,fc,Afs,Afc]
    #print l_bin,pow_mean
    
    # Compute MC statistics for errors
    #print('Computing MC Statistics')
    means,stds=error_wrapper(map_id,l_bin,np.array(pow_mean),map_size=map_size,N_sims=N_sims,l_step=l_step,slope=slope)
	# must multiply pow_mean by pi to get correct Gaussian maps

    best_estim=[]
    for i in range(len(best_preds)):
        best_estim.append([best_preds[i],stds[i],means[i]])

    return best_estim

def error_wrapper(map_id,l_bin,Cl,map_size=a.map_size,N_sims=a.N_sims,l_step=a.l_step,slope=None):
    """ Computes many random Monte Carlo simulations of Gaussian field realisation of Cl spectrum to test estimations. This calls error_estimator to perform the analysis.
    In: map_id==map number`
    map_size = size in degrees (3,5,10 only)
    N_sims = no. of estimates
    l_step = step-size for slope estimation
    l_bin,Cl are binned power spectrum from entire map (used for MC computation)
    slope = fiducial slope from the initial B-mode binned map
   
    Out: mean + std of pol. strength, pol. angle + isotropic amplitude
    """
   
    # Configure directories
    indir=a.root_dir+str(map_size)+'deg/'
    
    # Load in real space map (just used as a template)
    Tmap=liteMap.liteMapFromFits(indir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')

    # Load in window
    window=liteMap.liteMapFromFits(indir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')

    # Initialise arrays
    angs,strs,amps,fs,fc,Afs,Afc=[],[],[],[],[],[],[]
    
    # Compute N_sims realisations:
    for n in range(N_sims):
    	#print n
    	#if n % 10 ==0:
        #	print('%s out of %s completed' %(n,N_sims))
          
        # Compute estimators
        p_str,p_ang,A_est,fs_est,fc_est,Afs_est,Afc_est,l_bin,Cl=error_estimator(map_id,map_size,l_step,Tmap,window,l_bin,Cl,slope)
        strs.append(p_str) # polarisation strength
        amps.append(A_est) # isotropic amplitude
        angs.append(p_ang) # polarisation angles
        fs.append(fs_est) # f_s parameter
        fc.append(fc_est) # f_c parameter
        Afc.append(Afc_est) # these are probably the unbiased quantities
        Afs.append(Afs_est)

    # Compute mean and standard deviations
    means=[np.mean(strs),np.mean(angs),np.mean(amps),np.mean(fs),np.mean(fc),\
    np.mean(Afs),np.mean(Afc)]
    stdevs=[np.std(strs),np.std(angs),np.std(amps),np.std(fs),np.std(fc),\
    np.std(Afs),np.mean(Afc)]
    
    # Catch any errors:
    if np.isnan(stdevs).any(): 
    	print 'err'
    return means,stdevs 

def makeGaussian(MCmap,l_bin,Cl):
    	MCmap.fillWithGaussianRandomField(np.array(l_bin),np.array(Cl),bufferFactor=2)
    	i=0
    	while np.isnan(MCmap.data).any():
    		print('retry %d' %(i))
    		i+=1
    		l_bin=l_bin[1:-1]
    		Cl=Cl[1:-1]
    		MCmap.fillWithGaussianRandomField(np.array(l_bin),np.array(Cl),bufferFactor=2)
    	return l_bin,Cl

def error_estimator(map_id,map_size,l_step,Tmap,window,l_bin,Cl,slope):
    """ This function computes a random Gaussian field and applies statistical estimators to it to find error in predictions.
    Inputs:real space Bmap
    slope=best fit slope of power spectrum map
    A=amplitude of Bmap
    map_id, map_size,l_step (as before)
    l_bin,Cl are 1D binned spectra of entire map (for complete l range)
    slope is fiducial B-mode binned map slope


    Out: strength, angle and amplitude,f_s,f_c,Af_s,Af_c parameters from KK estimator
    """
    import numpy as np
    # Compute random map from the given power spectrum
    MCmap=Tmap.copy()
    		
    l_bin,Cl=makeGaussian(MCmap,l_bin,Cl)
    l_min=np.min(l_bin)-(-l_bin[0]+l_bin[1])/2.
    l_max=np.max(l_bin)+(l_bin[1]-l_bin[0])/2. # new max/min of array
    
    #print(l_min,l_max)
    #MCmap.fillWithGaussianRandomField(np.array(l_bin),np.array(Cl),bufferFactor=2)
    # make 2x size map + cut out to avoid periodic boundary condition erorrs

    # Create power space map
    MCpower=fftTools.powerFromLiteMap(MCmap)
    #MCpower=Power2DMap(MCmap,window)
    # Now apply statistical estimator - this RECOMPUTES slope for each
    from .KKtest import angle_estimator
    p_str,p_ang,A_est,fs_est,fc_est,Afs_est,Afc_est=angle_estimator(map_id,map_size=map_size,l_step=l_step,lMin=l_min,lMax=l_max,map=MCpower,slope=slope)

    return p_str,p_ang,A_est,fs_est,fc_est,Afs_est,Afc_est,l_bin,Cl

#def Power2DMap(Bmap,window):
#    """ Compute 2D power map of Bmode map from real space map. This is taken from fftPol code, just for only one map"""
    #Bmap.data*=window.data # multiply by window
#    fftB=fftTools.fftFromLiteMap(Bmap) # compute Fourier Transform

    # Now compute power map
#    BB_power = fftTools.powerFromLiteMap(Bmap)

#    return BB_power
    
def plot_est_and_err(map_size,flatSky=False):
    """ PPlot the KK estimated angle maps and errors using the Gaussian errors. Here we plot the significance of the detection at each position.
    Input: map_size = 3,5,10. 
    flatSky -> if False, plot on mollweide projection, else on flat plane
    Output - 6 maps saved in SkyPolMaps/ directory"""
    
    import numpy as np

    # Read in data
    data=np.load(a.root_dir+'MCestimates'+str(map_size)+'deg.npy')
    
    
    N = len(data)#  no. patches
    
    # Estimator values
    est_str=[d[0][0] for d in data] # polarisation strength
    est_ang=[d[1][0] for d in data] # polarisation angle
    est_A=[d[2][0] for d in data] # monopole amplitude
    est_fs=[d[3][0] for d in data] # f_s anisotropy
    est_fc=[d[4][0] for d in data] # f_c anisotropy
    est_Afs=[d[5][0] for d in data] # Af_s anisotropy
    est_Afc=[d[6][0] for d in data]
    
    
    # Gaussian errors
    err_str=[d[0][1] for d in data] # strength gaussian error
    err_ang=[d[1][1] for d in data] # angle gaussian error
    err_A=[d[2][1] for d in data] # Amplitude gaussian error
    err_fs=[d[3][1] for d in data] # fs gaussian error -> unbiased?
    err_fc=[d[4][1] for d in data] # fc gaussian error
    err_Afs=[d[5][1] for d in data] # Afs Gaussian error -> unbiased?
    err_Afc=[d[6][1] for d in data] # Afc Gaussian error 
    
    # Gaussian means
    mean_str=[d[0][2] for d in data]
    mean_ang=[d[1][2] for d in data]
    mean_A=[d[2][2] for d in data]
    mean_fs=[d[3][2] for d in data]
    mean_fc=[d[4][2] for d in data] 
    mean_Afs=[d[5][2] for d in data]
    mean_Afc=[d[6][2] for d in data]
    
    # log data
    est_log_A=[np.log10(A) for A in est_A]
    
    # Compute estimate of strength without zero error
    unbiased_str=[est_str[i]-mean_str[i] for i in range(len(est_str))]
    
    # Compute significances for Afs,Afc data
    sig_Afs=[est_Afs[i]/err_Afs[i] for i in range(N)]
    sig_Afc=[est_Afc[i]/err_Afc[i] for i in range(N)]
    
    # Compute unbiased errors in fs,fc
    err_fs_from_Afs=[np.sqrt((est_fs[i]*err_Afs[i]/est_Afs[i])**2+(est_fs[i]*err_A[i]/est_A[i])**2) for i in range(N)]
    err_fc_from_Afc=[np.sqrt((est_fc[i]*err_Afc[i]/est_Afc[i])**2+(est_fc[i]*err_A[i]/est_A[i])**2) for i in range(N)]
    
    # Compute significances for fs, fc data
    
    sig_fs = [est_fs[i]/err_fs[i] for i in range(N)] 
    sig_fc= [est_fc[i]/err_fc[i] for i in range(N)] 
    
    # Next compute errors on angle and unbiased strength
    
    def amplitude_err(fs,fc,sig_fs,sig_fc): # error in amplitude 
    	return np.sqrt(((sig_fs*fs)**2+(sig_fc*fc)**2)/(fs**2+fc**2))
    
    def angle_err(fs,fc,sig_fs,sig_fc): # error in angle in degrees
    	return 1./4*np.sqrt(((fs*sig_fc)**2+(fc*sig_fs)**2)/(fs**2+fc**2))*180./np.pi
    
    # These are the unbiased estimates	
    unbiased_err_str = [amplitude_err(est_fs[i],est_fc[i],err_fs[i],err_fc[i]) for i in range(N)]
    unbiased_err_ang = [angle_err(est_fs[i],est_fc[i],err_fs[i],err_fc[i]) for i in range(N)]
    
    # Compute significance of strength and angle
    sig_unbiased_str = [est_str[i]/unbiased_err_str[i] for i in range(N)]
    sig_unbiased_ang = [est_ang[i]/unbiased_err_ang[i] for i in range(N)]
     
    # Also read in coordinates of patches
    import pickle
    full_ras=pickle.load(open(a.root_dir+str(map_size)+'deg/fvsmapRas.pkl','rb'))
    full_decs=pickle.load(open(a.root_dir+str(map_size)+'deg/fvsmapDecs.pkl','rb'))
    goodMap=pickle.load(open(a.root_dir+str(map_size)+'deg/fvsgoodMap.pkl','rb'))
    ras=[full_ras[i] for i in range(len(full_ras)) if goodMap[i]!=False]
    decs=[full_decs[i] for i in range(len(full_decs)) if goodMap[i]!=False]
    ras=ras[:N]
    decs=decs[:N] # remove any extras
    
    # Now plot on grid
    import astropy.coordinates as coords
    import astropy.units as u
    import matplotlib.pyplot as plt

    ra_deg=coords.Angle(ras*u.degree) # convert to correct format
    ra_deg=ra_deg.wrap_at(180*u.degree)
    dec_deg=coords.Angle(decs*u.degree)

    # Make output directory
    import os
    outDir=a.root_dir+'SkyPolMapsUnbiased%sdeg/' %(map_size)
    
    if not os.path.exists(outDir):
    	os.mkdir(outDir)
    	
    # Plot maps
    datSet=[est_str,est_ang,est_A,est_fs,est_fc,\
    sig_fs,sig_fc,\
    sig_unbiased_str,sig_unbiased_ang,unbiased_err_str,unbiased_err_ang,\
    unbiased_str,mean_str,est_log_A]
    
    #datSet=[pol_str,pol_ang,str_err,ang_err,sig_str,sig_ang] # all datasets
    names=['Polarisation Strength','Polarisation Angle','Monopole Amplitude',\
    'f_s','f_c','f_s Significance','f_c Significance',\
    'Unbiased Strength Significance','Unbiased Angle Significance',\
    'Unbiased Strength Error','Unbiased Angle Error',\
    'Strength - mean(isotropic strength)','Mean Isotropic Strength','log Amplitude Estimation']
    #names=['Polarisation Strength', 'Polarisation Angle', \
    #'Strength Gaussian Error','Angle Gaussian Error',
    #'Polarisation Strength Significance','Polarisation Angle Significance']
    fileStr=['est_str','est_ang','est_A','est_fs','est_fc','sig_fs','sig_fc',\
    'sig_unbiased_str','sig_unbiased_ang','unbiased_err_str','unbiased_err_ang',\
    'str_mean_sub','mean_iso_str','est_log_A']
    #fileStr=['pol_str','pol_ang','str_err','ang_err','sig_str','sig_ang']
    
    # Size of points in plot
    if flatSky:
    	s_dot=120
    else:
    	if map_size==5:
    		s_dot=50 
        elif map_size==3:
    		s_dot=30
    	
    
    for i in range(len(datSet)):
        fig=plt.figure()
        if flatSky:
        	fig.add_subplot(111)
        	plt.scatter(ras,decs,c=datSet[i],marker='o',s=s_dot)
        else:
            	fig.add_subplot(111,projection='mollweide')
            	plt.scatter(ra_deg.radian,dec_deg.radian,c=datSet[i],marker='o',s=s_dot)
            	plt.axis('off')
   	#if i == 2:
   	#	plt.scatter(ra_deg.radian,dec_deg.radian,c=datSet[i],marker='o',vmax=0.5,s=s_dot)
   	#elif i ==4:
   	#	plt.scatter(ra_deg.radian,dec_deg.radian,c=datSet[i],marker='o',vmax=10,s=s_dot)
   	#else:
    	plt.colorbar()
    	plt.title(names[i])
    	plt.savefig(outDir+str(map_size)+'deg_'+fileStr[i]+'.png',bbox_inches='tight')
    	plt.clf()
    	plt.close()


