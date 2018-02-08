import numpy as np
from hades.params import BICEP
a=BICEP()

from flipper import *

def padded_estimates(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,lMin=a.lMin,lMax=a.lMax,KKmethod=a.KKmethod,rot=0.):
	""" Compute the estimated angle, amplitude and polarisation fraction with noise, using zero-padding.
	Noise model is from Hu & Okamoto 2002 and errors are estimated using MC simulations.
	
	Input: map_id (tile number)
	padding_ratio (ratio of real-space padded tile width to initial tile width)
	map_size (tile width in degrees)
	sep (separation of tile centres in degrees)
	N_sims (number of MC simulations)
	noise_power (noise power in microK-arcmin)
	FWHM (noise FWHM in arcmin)
	slope (fiducial slope of C_l isotropic dust dependance)
	lMin / lMax (range of ell values to apply the estimators over)
	KKmethod (Boolean, controlling which SNR to use (see KKtest.py))
	rot (angle to rotate by before applying estimators - for testing)
	NB: if rot!=0, only ang, A, frac can be used
	
	Output: Estimated data for A, fs, fc and errors."""
	
	# First compute high-resolution B-mode map from padded-real space map with desired padding ratio
	from .PaddedPower import MakePaddedPower
	Bpow=MakePaddedPower(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep)
	
	# Input directory:
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
	
	# Compute the noise power map using the B-mode map as a template
	from .NoisePower import noise_map
	noiseMap=noise_map(powMap=Bpow.copy(),noise_power=noise_power,FWHM=FWHM,windowFactor=Bpow.windowFactor)
	
	# Compute total map
	totMap=Bpow.copy()
	totMap.powerMap=Bpow.powerMap+noiseMap.powerMap
	
	# Apply the KK estimators
	from .KKtest import zero_estimator
	A_est,fs_est,fc_est,Afs_est,Afc_est=zero_estimator(totMap.copy(),lMin=lMin,lMax=lMax,slope=slope,factor=1e-10,FWHM=FWHM,noise_power=noise_power,KKmethod=KKmethod,rot=rot)
	# (Factor is expected monpole amplitude (to speed convergence))
	
	# Compute anisotropy fraction and angle
	ang_est=-rot+0.25*180./np.pi*(np.arctan(Afs_est/Afc_est)) # in degrees
	frac_est=np.sqrt(fs_est**2.+fc_est**2.)
		
	## Run MC Simulations	
	# First compute 1D power spectrum by binning in annuli
	from hades.PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totMap.copy(),0.8*a.lMin,1.*a.lMax,0.8*a.l_step,binErr=False,windowFactor=Bpow.windowFactor) 
	# gives central binning l and mean power in annulus using window function corrections (from unpaddded map)
	
	# Compute univariate spline model fit to 1D power spectrum
	from scipy.interpolate import UnivariateSpline
	spline_fun = UnivariateSpline(np.log10(l_cen),np.log10(mean_pow),k=4) # compute spline of log data
	
	def model_power(ell):
		return 10.**spline_fun(np.log10(ell)) # this estimates 1D spectrum for any ell
	
	# Initialise arrays
	A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC=[],[],[],[],[],[],[]
	
	from hades.NoisePower import single_MC
	
	for n in range(N_sims): # for each MC map
		MC_map=single_MC(totMap.copy(),model_power,windowFactor=Bpow.windowFactor) # create random map from isotropic spectrum
		output=zero_estimator(MC_map.copy(),lMin=lMin,lMax=lMax,slope=slope,factor=1e-10,FWHM=FWHM,noise_power=noise_power,KKmethod=KKmethod) 
		# compute MC anisotropy parameters  
		A_MC.append(output[0])
		fs_MC.append(output[1])
		fc_MC.append(output[2])
		Afs_MC.append(output[3])
		Afc_MC.append(output[4])
		epsilon_MC.append(np.sqrt(output[1]**2.+output[2]**2.))
		ang_MC.append(0.25*180./np.pi*np.arctan(output[3]/output[4]))
		
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
	fracdat=[frac_est,frac_mean,frac_std]
	angdat=[ang_est,ang_mean,ang_std]
	
	# Return all output
	return Adat,fsdat,fcdat,Afsdat,Afcdat,fracdat,angdat
	

def MakePaddedPower(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,freq=a.freq):
    """ Function to create 2D B-mode power map from real space map padded with zeros.
    Input: map_id (tile number)
    map_size (in degrees)
    sep (separation of map centres)
    padding_ratio (ratio of padded map width to original (real-space) map width)
    freq (experiment frequency (calibrated for 100-353 GHz))
   
    Output: B-mode map in power-space   
    """
    import flipperPol as fp
    
    inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
    
    # Read in original maps from file
    Tmap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    maskMap=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    
    # Compute window factor <W^2> for UNPADDED window (since this is only region with data)
    #windowFactor=np.mean(maskMap.data**2.)
    
    # Compute zero-padded maps (including mask map)
    from .PaddedPower import zero_padding
    zTmap=zero_padding(Tmap,padding_ratio)
    zQmap=zero_padding(Qmap,padding_ratio)
    zUmap=zero_padding(Umap,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    
    # Compute window factor for new map
    windowFactor=np.mean(zWindow.data**2.)
    
    # Define mod(l) and ang(l) maps needed for fourier transforms
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary

    # Create pure T,E,B maps using 'hybrid' method to minimize E->B leakage
    fT,fE,fB=fp.fftPol.TQUtoPureTEB(zTmap,zQmap,zUmap,zWindow,modL,angL,method='hybrid')

    # Transform into power space
    _,_,_,_,_,_,_,_,BB=fp.fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT,fE,fB)
    
    # Now account for power loss due to padding:
    #BB.powerMap*=zTmap.powerFactor # no effect here
    
    # Rescale to correct amplitude using dust SED
    from .PowerMap import dust_emission_ratio
    dust_intensity_ratio=dust_emission_ratio(freq)
    
    BB.powerMap*=dust_intensity_ratio**2. # square since applied to power-maps
    
    # Account for window factor
    BB.powerMap/=windowFactor
    BB.windowFactor=windowFactor # store window factor
    
    return BB
    
	
def MakePowerAndFourierMaps(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,freq=a.freq,fourier=True,power=True,returnMask=False):
    """ Function to create 2D B-mode power map from real space map padded with zeros.
    Input: map_id (tile number)
    map_size (in degrees)
    sep (separation of map centres)
    padding_ratio (ratio of padded map width to original (real-space) map width)
    freq (experiment frequency (calibrated for 100-353 GHz))
    fourier (return fourier space map?)
    power (return power map?)
    returnMask (return real-space mask window?)
   
    Output: B-mode map in power-space , B-mode map in Fourier-space  
    """
    import flipperPol as fp
    
    inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
    
    # Read in original maps from file
    Tmap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    maskMap=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    
    
    # Compute zero-padded maps (including mask map)
    from .PaddedPower import zero_padding
    zTmap=zero_padding(Tmap,padding_ratio)
    zQmap=zero_padding(Qmap,padding_ratio)
    zUmap=zero_padding(Umap,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    
    # Compute window factor <W^2> for padded window (since this is only region with data)
    windowFactor=np.mean(zWindow.data**2.)

    # Define mod(l) and ang(l) maps needed for fourier transforms
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary

    # Create pure T,E,B maps using 'hybrid' method to minimize E->B leakage
    _,_,fB=fp.fftPol.TQUtoPureTEB(zTmap,zQmap,zUmap,zWindow,modL,angL,method='hybrid')
    # Rescale to correct amplitude using dust SED
    from .PowerMap import dust_emission_ratio
    dust_intensity_ratio=dust_emission_ratio(freq)
    
    fB.kMap*=dust_intensity_ratio # apply dust-reduction factor 
    #fE.kMap*=dust_intensity_ratio
    #fT.kMap*=dust_intensity_ratio  

    # Account for window factor - this accounts for power loss due to padding
    fB.kMap/=np.sqrt(windowFactor)
    #fE.kMap/=np.sqrt(windowFactor)
    #fT.kMap/=np.sqrt(windowFactor)
    
    if power:
    	# Transform into power space
    	BB=fftTools.powerFromFFT(fB)
    	#_,_,_,_,_,_,_,_,BB=fp.fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT,fE,fB)
    	# Now account for power loss due to padding:
    	#BB.powerMap*=zTmap.powerFactor # no effect here
	#BB.powerMap*=dust_intensity_ratio**2. # square since applied to power-maps
    
    	#BB.powerMap/=windowFactor
    	#BB.windowFactor=windowFactor # store window factor
    if returnMask:
        if fourier and power:
    		return fB,BB,zWindow
    	elif fourier:
    		return fB,zWindow
    	elif power:
    		return BB,zWindow
    else:
    	if fourier and power:
		return fB,BB
	elif fourier:
		return fB
	elif power:
	    	return BB
    
def zero_padding(tempMap,padding_factor):
	""" Pad the real-space map with zeros.
	Padding_factor is ratio of padded map width to original map width.
	NB: WCS data is NOT changed by the zero-padding, so will be inaccurate if used.
	(this doesn't affect any later processes)"""
	
	if padding_factor==1.: # for no padding for convenience
		tempMap.powerFactor=1.
		return tempMap
	else:
		zeroMap=tempMap.copy() # unpadded map template
		oldNx=tempMap.Nx
		oldNy=tempMap.Ny # old Map dimensions
		
		# Apply padding
		paddingY=int(oldNy*(padding_factor-1.)/2.)
		paddingX=int(oldNx*(padding_factor-1.)/2.) # no. zeros to add to each edge of map
		zeroMap.data=np.lib.pad(tempMap.data,((paddingY,paddingY),(paddingX,paddingX)),'constant') # pads with zeros by default
		
		# Reconfigure other parameters
		zeroMap.Ny=len(zeroMap.data)
		zeroMap.Nx=len(zeroMap.data[0]) # map dimensions
		zeroMap.area*=(zeroMap.Ny/oldNy)*(zeroMap.Nx/oldNx) # rescale area
		zeroMap.x1-=oldNx*zeroMap.pixScaleX*180./np.pi # change width in degrees
		zeroMap.x0+=oldNx*zeroMap.pixScaleX*180./np.pi
		zeroMap.y0-=oldNy*zeroMap.pixScaleY*180./np.pi
		zeroMap.y1+=oldNy*zeroMap.pixScaleY*180./np.pi # signs to fit with flipper conventions
		
		# Define 'power factor'
		# Power-space maps must be multiplied by this factor to have correct power
		#zeroMap.powerFactor=zeroMap.area/tempMap.area
	
		return zeroMap
	
# Default parameters
map_size=a.map_size
nmin = 0
nmax = 1e5#1399#3484
cores = 42


if __name__=='__main__':
     """ Batch process to use all available cores to compute the KK estimators and Gaussian errors using the iterator function below.
    Inputs are min and max file numbers. Output is saved as npy file"""

     import tqdm
     import sys
     import numpy as np
     import multiprocessing as mp
     	
     # Parameters if input from the command line
     if len(sys.argv)>=2:
     	map_size=int(sys.argv[1])
     if len(sys.argv)>=3:
         nmin = int(sys.argv[2])
     if len(sys.argv)>=4:
         nmax = int(sys.argv[3])
     if len(sys.argv)==5:
         cores = int(sys.argv[4])
         
     print 'Starting estimations for map width = %s degrees' %map_size
     
     # Compute map IDs with non-trivial data
     all_file_ids=np.arange(nmin,nmax+1)
     import pickle
     goodMaps=pickle.load(open(a.root_dir+str(map_size)+'deg'+str(a.sep)+'/fvsgoodMap.pkl','rb'))
     
     file_ids=[int(all_file_ids[i]) for i in range(len(all_file_ids)) if goodMaps[i]!=False] # just for correct maps
     params=[[file_id, map_size] for file_id in file_ids]
     # Start the multiprocessing
     p = mp.Pool(processes=cores)
     
     # Define iteration function
     from hades.PaddedPower import iterator
     
     # Display progress bar with tqdm
     r = list(tqdm.tqdm(p.imap(iterator,params),total=len(file_ids)))
     
     np.save(a.root_dir+'BICEPnoNoisePaddedMCEstimates%sdeg%s.npy' %(map_size,a.sep),np.array(r))
     

def iterator(params):
	""" To run the iterations"""
	from hades.PaddedPower import padded_estimates
	out = padded_estimates(int(params[0]),map_size=params[1])
	print('%s map complete' %params[0])
	return out


def reconstructor(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power):
	""" Reconstruct and plot data, in PaddedMaps***/ subdirectory."""
	import os
	
	# First create data from batch files if available
	batchDir=a.root_dir+'BatchData/%sdeg_%ssep_%sFWHM_%snoise_power/' %(map_size,sep,FWHM,noise_power) # for batch files
	if os.path.exists(batchDir):
		from hades.batch_maps import reconstruct_array
		reconstruct_array(map_size=map_size,sep=sep,FWHM=FWHM,noise_power=noise_power)
	else:
		return Exception('Data not yet created')
	
	# File directory
	fileDir=a.root_dir+'Maps/%sdeg_%ssep_%sFWHM_%snoise_power/' %(map_size,sep,FWHM,noise_power)
	
	dat = np.load(fileDir+'data.npy')
	N = len(dat)
	
	# Construct A,fs,fc arrays
	A=[d[0][0] for d in dat]
	fs=[d[1][0] for d in dat]
	fc=[d[2][0] for d in dat]
	fs_err=[d[1][2] for d in dat]
	fc_err=[d[2][2] for d in dat]
	eps=[d[5][0] for d in dat]
	ang=[d[6][0] for d in dat]
	eps_mean=[d[5][1] for d in dat]
	logA=[np.log10(d[0][0]) for d in dat]
	ratio=[eps[i]/eps_mean[i] for i in range(len(dat))]
	
	f_err=[np.mean([d[1][2],d[2][2]]) for d in dat] # as fs_err=fc_err
	
	# Compute angle error
	ang_err=[f_err[i]/(4.*eps[i])*180./np.pi for i in range(len(A))]
	
    	# Compute chi-squared probability
	def chi2_prob(eps,err):
		return np.exp(-eps**2./(2.*err**2.))
	
	rand_prob=[chi2_prob(eps[i],f_err[i]) for i in range(len(f_err))]
	rho_log_prob=[-eps[i]**2./(2.*f_err[i]**2.)*np.log10(np.e) for i in range(len(rand_prob))]
	
	dat_set=[A,fs,fc,fs_err,fc_err,eps,ang,eps_mean,logA,f_err,ang_err,rand_prob,rho_log_prob,ratio]
	names=['Monopole Amplitude','fs','fc','fs_err','fc_err','Anisotropy Fraction','Anisotropy Angle',\
	'Mean MC Anisotropy Fraction','log Monopole Amplitude','f_err','Angle Error','Chi-squared Probability of Isotropy',\
	'rho: log Isotropy Probability','Ratio of Tile Epsilon to Mean Isotropic Epsilon (epsilon=anisotropy fraction)']	
	name_str=['A','fs','fc','fs_err','fc_err','epsilon','angle','mean_mc_epsilon','logA','sigma_f','angle_err','rand_prob','log_rho_prob','epsilon_ratio']
	
	# Load coordinates of patch centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,N)
	
    	# Now plot on grid:
    	import matplotlib.pyplot as plt
    	import cmocean
    	
    	for j in range(len(names)):
    		plt.figure()
    		if names[j]=='Angle Error':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,vmax=22.5)
    		elif names[j]=='rho: log Isotropy Probability':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,vmin=-5)
    		elif names[j]=='Anisotropy Angle':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,cmap=cmocean.cm.phase)
    		else:
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80)
    		plt.title(names[j])
    		plt.colorbar()
    		plt.savefig(fileDir+name_str[j]+'.png',bbox_inches='tight')
    		plt.clf()
    		plt.close()
    		
def isotropic_reconstructor(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power):
	""" Reconstruct and plot data, in PaddedMaps***/ subdirectory."""
	import os
	
	# First create data from batch files if available
	batchDir=a.root_dir+'BatchDataIsotropic/%sdeg_%ssep_%sFWHM_%snoise_power/' %(map_size,sep,FWHM,noise_power) # for batch files
	if os.path.exists(batchDir):
		from hades.batch_maps import reconstruct_array
		reconstruct_array(map_size=map_size,sep=sep,FWHM=FWHM,noise_power=noise_power)
	else:
		return Exception('Data not yet created')
	
	# File directory
	fileDir=a.root_dir+'MapsIsotropic/%sdeg_%ssep_%sFWHM_%snoise_power/' %(map_size,sep,FWHM,noise_power)
	
	dat = np.load(fileDir+'data.npy')
	N = len(dat)
	
	# Construct A,fs,fc arrays
	A=[d[0][1] for d in dat]
	fs=[d[1][1] for d in dat]
	fc=[d[2][1] for d in dat]
	fs_err=[d[1][2] for d in dat]
	fc_err=[d[2][2] for d in dat]
	eps=[d[5][1] for d in dat]
	ang=[d[6][1] for d in dat]
	eps_mean=[d[5][1] for d in dat]
	logA=[np.log10(d[0][1]) for d in dat]
	ratio=[eps[i]/eps_mean[i] for i in range(len(dat))]
	
	f_err=[np.mean([d[1][2],d[2][2]]) for d in dat] # as fs_err=fc_err
	
	# Compute angle error
	ang_err=[f_err[i]/(4.*eps[i])*180./np.pi for i in range(len(A))]
	
    	# Compute chi-squared probability
	def chi2_prob(eps,err):
		return np.exp(-eps**2./(2.*err**2.))
	
	rand_prob=[chi2_prob(eps[i],f_err[i]) for i in range(len(f_err))]
	rho_log_prob=[-eps[i]**2./(2.*f_err[i]**2.)*np.log10(np.e) for i in range(len(rand_prob))]
	
	dat_set=[A,fs,fc,fs_err,fc_err,eps,ang,eps_mean,logA,f_err,ang_err,rand_prob,rho_log_prob,ratio]
	names=['Monopole Amplitude','fs','fc','fs_err','fc_err','Anisotropy Fraction','Anisotropy Angle',\
	'Mean MC Anisotropy Fraction','log Monopole Amplitude','f_err','Angle Error','Chi-squared Probability of Isotropy',\
	'rho: log Isotropy Probability','Ratio of Tile Epsilon to Mean Isotropic Epsilon (epsilon=anisotropy fraction)']	
	name_str=['A','fs','fc','fs_err','fc_err','epsilon','angle','mean_mc_epsilon','logA','sigma_f','angle_err','rand_prob','log_rho_prob','epsilon_ratio']
	
	# Load coordinates of patch centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,N)
	
    	# Now plot on grid:
    	import matplotlib.pyplot as plt
    	import cmocean
    	
    	for j in range(len(names)):
    		plt.figure()
    		if names[j]=='Angle Error':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,vmax=22.5)
    		elif names[j]=='rho: log Isotropy Probability':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,vmin=-5)
    		elif names[j]=='Anisotropy Angle':
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80,cmap=cmocean.cm.phase)
    		else:
    			plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80)
    		plt.title(names[j])
    		plt.colorbar()
    		plt.savefig(fileDir+name_str[j]+'.png',bbox_inches='tight')
    		plt.clf()
    		plt.close()
    		

