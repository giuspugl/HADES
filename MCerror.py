from flipper import *
import numpy as np

def est_and_err(map_id,map_size=5,N_sims=100,l_step=50):
    """ Compute estimated angle, amplitude and polarisation of map compared to Gaussian error from MC simulations.
    Input - map_id, map_size
    N_sims = no. of MC sims
    l_step = bin width for computing slope of B-space map

    Out: best estimator and error as a list in order (strength,angle,amplitude)
    """
    from .KKtest import angle_estimator

    # Find best estimate of angle + polarisation strength
    p_str,p_ang,A,slope=angle_estimator(map_id,map_size,l_step=l_step,lMin=100,lMax=2000,returnSlope=True)
    best_preds=[p_str,p_ang,A]
    
    # Compute MC statistics for errors    
    means,stds=error_wrapper(map_id,map_size=map_size,N_sims=N_sims,l_step=l_step,slope=slope,A=A)
    best_estim=[]
    for i in range(len(best_preds)):
        best_estim.append([best_preds[i],stds[i],means[i]])

    return best_estim

def error_wrapper(map_id,map_size=5,N_sims=100,l_step=50,slope=None,A=None):
    """ Computes many random Monte Carlo simulations of Gaussian field realisation of Cl spectrum to test estimations. This calls error_estimator to perform the analysis.
    In: map_id==map number
    map_size = size in degrees (3,5,10 only)
    N_sims = no. of estimates
    l_step = step-size for slope estimation
    slope,A = slope and amplitude if already computed (else recomputed)

    Out: mean + std of pol. strength, pol. angle + isotropic amplitude
    """
    from .PowerMap import RescaledPlot

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

    # Compute slope + A for real map if not already computed
    if slope==None and A==None:
        _,slope,A=RescaledPlot(map_id,l_min=100,l_max=2000,l_step=l_step,\
                                         map_size=map_size, rescale=True,\
                                         returnMap=True,save=False,\
                                         show=False,saveFit=False,showFit=False)
    
    # Initialise arrays
    angs,strs,amps=[],[],[]
    
    # Compute N_sims realisations:
    for n in range(N_sims):
        print('%s out of %s completed' %(n,N_sims))
          
        # Compute estimators
        p_str,p_ang,A_est=error_estimator(map_id,map_size,l_step,Tmap,window,slope,A)
        strs.append(p_str) # polarisation strength
        amps.append(A_est) # isotropic amplitude
        angs.append(p_ang) # polarisation angles

    # Compute mean and standard deviations
    means=[np.mean(strs),np.mean(angs),np.mean(amps)]
    stdevs=[np.std(strs),np.std(angs),np.std(amps)]
    return means,stdevs 

def error_estimator(map_id,map_size,l_step,Tmap,window,slope,A):
    """ This function computes a random Gaussian field and applies statistical estimators to it to find error in predictions.
    Inputs:real space Bmap
    slope=best fit slope of power spectrum map
    A=amplitude of Bmap
    map_id, map_size,l_step (as before)
    """
    # define Cl predictions
    l = np.array(range(1,20000))
    Cl = A*(l**(-slope))

    MCmap=Tmap.copy()
    MCmap.fillWithGaussianRandomField(l,Cl,bufferFactor=3)
    # make 3x size map + cut out to avoid periodic boundary condition erorrs

    # Create power space map
    MCpower=Power2DMap(MCmap,window)

    # Now apply statistical estimator
    from .KKtest import angle_estimator
    p_str,p_ang,A_est=angle_estimator(map_id,map_size=map_size,l_step=l_step,lMin=100,lMax=2000,slope=slope,map=MCpower)

    return p_str,p_ang,A_est

def Power2DMap(Bmap,window):
    """ Compute 2D power map of Bmode map from real space map + window function. This is taken from fftPol code, just for only one map"""
    Bmap.data*=window.data # multiply by window
    fftB=fftTools.fftFromLiteMap(Bmap) # compute Fourier Transform

    # Now compute power map
    BB_power = fftTools.powerFromFFT(fftB,fftB)

    return BB_power
