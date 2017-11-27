from flipper import *
import numpy as np

def est_and_err(map_id,map_size=3,N_sims=100,l_step=100):
    """ Compute estimated angle, amplitude and polarisation of map compared to Gaussian error from MC simulations.
    Input - map_id, map_size
    N_sims = no. of MC sims
    l_step = bin width for computing slope of B-space map

    Out: best estimator and error as a list in order (strength,angle,amplitude)
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
    l_min=50#l_step # don't include l=0
    l_max=2000 # max/min of map

    # Compute power
    #print('Computing Full Power Spectrum')
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
    p_str,p_ang,A=angle_estimator(map_id,map_size,map=Bmap,l_step=l_step,lMin=l_min,lMax=l_max,slope=slope)
    best_preds=[p_str,p_ang,A]
    #print l_bin,pow_mean
    
    # Compute MC statistics for errors
    #print('Computing MC Statistics')
    means,stds=error_wrapper(map_id,l_bin,np.pi*np.array(pow_mean),map_size=map_size,N_sims=N_sims,l_step=l_step,slope=slope)
	# must multiply pow_mean by pi to get correct Gaussian maps

    best_estim=[]
    for i in range(len(best_preds)):
        best_estim.append([best_preds[i],stds[i],means[i]])

    return best_estim

def error_wrapper(map_id,l_bin,Cl,map_size=3,N_sims=100,l_step=50,slope=None):
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
    if map_size==3:
        indir='/data/ohep2/sims/3deg/'
    elif map_size==5:
        indir='/data/ohep2/sims/5deg/'
    elif map_size==10:
        indir='/data/ohep2/sims/simdata/'
    else:
        return Exception('Incorrect map size')
        
    # Load in real space map (just used as a template)
    Tmap=liteMap.liteMapFromFits(indir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')

    # Load in window
    window=liteMap.liteMapFromFits(indir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')

    # Initialise arrays
    angs,strs,amps=[],[],[]
    
    # Compute N_sims realisations:
    for n in range(N_sims):
    	#if n % 10 ==0:
        #	print('%s out of %s completed' %(n,N_sims))
          
        # Compute estimators
        p_str,p_ang,A_est,l_bin,Cl=error_estimator(map_id,map_size,l_step,Tmap,window,l_bin,Cl,slope)
        strs.append(p_str) # polarisation strength
        amps.append(A_est) # isotropic amplitude
        angs.append(p_ang) # polarisation angles

    # Compute mean and standard deviations
    means=[np.mean(strs),np.mean(angs),np.mean(amps)]
    stdevs=[np.std(strs),np.std(angs),np.std(amps)]
    
    # Catch any errors:
    if np.isnan(stdevs).any(): # double l range 
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


    Out: strength, angle and amplitude from KK estimator
    """
    import numpy as np
    # Compute random map from the given power spectrum
    MCmap=Tmap.copy()
    		
    l_bin,Cl=makeGaussian(MCmap,l_bin,Cl)
    l_min=np.min(l_bin)-l_step/2.
    l_max=np.max(l_bin)+l_step/2. # new max/min of array
    
    #print(l_min,l_max)
    #MCmap.fillWithGaussianRandomField(np.array(l_bin),np.array(Cl),bufferFactor=2)
    # make 2x size map + cut out to avoid periodic boundary condition erorrs

    # Create power space map
    MCpower=Power2DMap(MCmap,window)
    # Now apply statistical estimator - this RECOMPUTES slope for each
    from .KKtest import angle_estimator
    p_str,p_ang,A_est=angle_estimator(map_id,map_size=map_size,l_step=l_step,lMin=l_min,lMax=l_max,map=MCpower,slope=slope)

    return p_str,p_ang,A_est,l_bin,Cl

def Power2DMap(Bmap,window):
    """ Compute 2D power map of Bmode map from real space map + window function. This is taken from fftPol code, just for only one map"""
    Bmap.data*=window.data # multiply by window
    fftB=fftTools.fftFromLiteMap(Bmap) # compute Fourier Transform

    # Now compute power map
    BB_power = fftTools.powerFromFFT(fftB,fftB)

    return BB_power
    
def plot_est_and_err(map_size):
    """ PPlot the KK estimated angle maps and errors using the Gaussian errors. Here we plot the significance of the detection at each position.
    Input: map_size = 3,5,10. 
    
    Output - 6 maps saved in SkyPolMaps/ directory"""
    
    import numpy as np

    # Read in data
    data=np.load('MCestimates'+str(map_size)+'deg.npy')
    
    pol_str=[d[0][0] for d in data] # polarisation strength
    pol_ang=[d[1][0] for d in data] # polarisation angle
    str_err=[d[0][1] for d in data] # strength gaussian error
    ang_err=[d[1][1] for d in data] # angle gaussian error
    
    # Compute significances
    sig_str = [d[0][0]/d[0][1] for d in data] # strength
    sig_ang= [d[1][0]/d[1][1] for d in data] # angle
    
    N = len(data)#  no. patches
    
    # Also read in coordinates of patches
    if map_size==10:
        coords=np.loadtxt('/data/ohep2/sims/simdata/fvsmapRAsDecs.txt')
  	# Create lists of patch centres
    	ras=np.zeros(N)
    	decs=np.zeros(N)
    	for i in range(N):
        	ras[i]=coords[i][1]
        	decs[i]=coords[i][2]
    elif map_size==5 or map_size==3:
    	import pickle
    	ras=pickle.load(open('/data/ohep2/sims/'+str(map_size)+'deg/fvsmapRas.pkl','rb'))
    	decs=pickle.load(open('/data/ohep2/sims/'+str(map_size)+'deg/fvsmapDecs.pkl','rb'))
    	ras=ras[:N]
    	decs=decs[:N] # remove any extras
    else:
       	return Exception('Invalid map_size')

    # Now plot on grid
    import astropy.coordinates as coords
    import astropy.units as u
    import matplotlib.pyplot as plt

    ra_deg=coords.Angle(ras*u.degree) # convert to correct format
    ra_deg=ra_deg.wrap_at(180*u.degree)
    dec_deg=coords.Angle(decs*u.degree)

    # Make output directory
    import os
    outDir='/data/ohep2/SkyPolMaps/'
    
    if not os.path.exists(outDir):
    	os.mkdir(outDir)
    	
    # Plot maps
    datSet=[pol_str,pol_ang,str_err,ang_err,sig_str,sig_ang] # all datasets
    names=['Polarisation Strength', 'Polarisation Angle', \
    'Strength Gaussian Error','Angle Gaussian Error',
    'Polarisation Strength Significance','Polarisation Angle Significance']
    fileStr=['pol_str','pol_ang','str_err','ang_err','sig_str','sig_ang']
    
    for i in range(len(datSet)):
        fig=plt.figure()
    	fig.add_subplot(111,projection='mollweide')
    	plt.scatter(ra_deg.radian,dec_deg.radian,c=datSet[i],marker='o',s=30)
    	plt.colorbar()
    	plt.axis('off')
    	plt.title(names[i])
    	plt.savefig(outDir+str(map_size)+'deg_'+fileStr[i]+'.png')
    	plt.clf()

