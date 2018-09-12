import numpy as np
from hades.params import BICEP
a=BICEP()
from flipper import *

def ffp10_lensing(delensing_fraction=a.delensing_fraction,camb_spectrum=a.camb_spectrum):
    """ Return a spline fit to the ffp10 lensing C_l^BB spectrum. This is NOT identical to the CAMB spectrum used otherwise.
    If camb_spectrum=True; this uses the CAMB spectrum instead for rescaling."""
    if camb_spectrum:
        from .NoisePower import lensed_Cl
        return lensed_Cl(delensing_fraction,ffp10_spectrum=False)
    dat=np.load(a.hades_dir+'CAMB_Profiles/ClLensTrue.npz') # load ClBB spectrum
    lensB=dat['ClBB']
    ellsB=dat['ell']
    
    # Create spline fit
    from scipy.interpolate import UnivariateSpline
    spl=UnivariateSpline(np.log(ellsB),np.log(lensB),k=5)
    def spline(ell):
        return np.exp(spl(np.log(ell)))*delensing_fraction
        
    return spline

def lens_ratio_correction(lensPower,lensFourier,delensing_fraction=a.delensing_fraction,lMax=a.lMax,l_step=a.l_step):
    """Rescale the Fourier lens map using the power lensing map to ensure that it reproduces the correct
    spectrum for ell, to avoid excess of power at high ell, as observed in cut-outs."""
    # First compute the binned power from the maps
    from .PowerMap import oneD_binning
    llF,ppF=oneD_binning(lensPower,1,lMax,l_step,exactCen=False)
    
    from .lens_power import ffp10_lensing
    ideal_lens=ffp10_lensing(delensing_fraction) # expected power 
    
    # Create spline fit to ratio:
    from scipy.interpolate import UnivariateSpline
    spl=UnivariateSpline(np.log(llF),np.log(ideal_lens(llF)/ppF),k=5)
    def ratio(ell):
            output=np.exp(spl(np.log(ell)))
            output[ell==0.]=1. # avoid infinities in log
            return output

    # Now rescale the Fourier map by sqrt(ratio)
    fourierScaled=lensFourier.copy()
    fourierScaled.kMap=fourierScaled.kMap*np.sqrt(ratio(fourierScaled.modLMap))
    
    return fourierScaled
        
def MakeFourierLens(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,\
            fourier=True,power=False,flipU=True,delensing_fraction=a.delensing_fraction,lMax=a.lMax,l_step=a.l_step):
    """ Function to create 2D B-mode fourier map from FFP10 lensed scalar map.
    Yanked from PaddedPower.MakePowerAndFourierMaps
    Input: map_id (tile number)
    map_size (in degrees)
    sep (separation of map centres)
    padding_ratio (ratio of padded map width to original (real-space) map width)
    fourier (return fourier space map?)
    power (return power map?)
    delensing_fraction (how much (uniform) delensing is applied)

    Output: lensing B-mode map in fourier [and power]-space  
    """
    raise Exception('Depracated')
    import flipperPol as fp
    
    inDir='LensTest/%sdeg%s/' %(map_size,sep)
    
    # Read in original maps from file
    Tmap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    if flipU:
        Umap.data*=-1.
    maskMap=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    
    # Compute zero-padded maps (including mask map)
    from .PaddedPower import zero_padding
    zTmap=zero_padding(Tmap,padding_ratio)
    zQmap=zero_padding(Qmap,padding_ratio)
    zUmap=zero_padding(Umap,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    
    del Tmap,Qmap,Umap
    
    # Compute window factor <W^2> for padded window (since this is only region with data)
    windowFactor=np.mean(zWindow.data**2.)

    # Define mod(l) and ang(l) maps needed for fourier transforms
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary

    # Create pure T,E,B maps using 'hybrid' method to minimize E->B leakage
    _,_,fB=fp.fftPol.TQUtoPureTEB(zTmap,zQmap,zUmap,zWindow,modL,angL,method='hybrid')
    
    del zTmap,zQmap,zUmap,modL,angL
    
    # Rescale to correct delensing fraction
    fB.kMap*=np.sqrt(delensing_fraction)

    # Account for window factor - this accounts for power loss due to padding
    fB.kMap/=np.sqrt(windowFactor)
    #fE.kMap/=np.sqrt(windowFactor)
    #fT.kMap/=np.sqrt(windowFactor)
    
    # Mask central pixel
    index=np.where(fB.modLMap==min(fB.modLMap.ravel())) # central pixel
    rest_index=np.where(fB.modLMap!=min(fB.modLMap.ravel())) 
    index2=np.where(fB.modLMap==min(fB.modLMap[rest_index].ravel())) # next to centre
    fB.kMap[index]=np.mean(fB.kMap[index2]) # replace pixel
    del index,rest_index,index2
    
    # Transform into power space
    BB=fftTools.powerFromFFT(fB)
    
    # Now account for extra power at high ell:
    from .lens_power import lens_ratio_correction
    fBcorr=lens_ratio_correction(BB,fB,delensing_fraction=delensing_fraction,lMax=lMax,l_step=l_step)
    
    #del zWindow,maskMap
    if power:
        BBcorr=fftTools.powerFromFFT(fBcorr)
    if fourier and power:
        return fBcorr,BBcorr
    elif fourier:
        return fBcorr
    elif power:
        del fBcorr
        return BBcorr
    
def MakeFourierLens2(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,\
            fourier=True,power=False,delensing_fraction=a.delensing_fraction):
    """ Function to create 2D B-mode fourier map from FFP10 lensed scalar map.
    Yanked from PaddedPower.MakePowerAndFourierMaps
    Input: map_id (tile number)
    map_size (in degrees)
    sep (separation of map centres)
    padding_ratio (ratio of padded map width to original (real-space) map width)
    fourier (return fourier space map?)
    power (return power map?)
    delensing_fraction (how much (uniform) delensing is applied)

    Output: lensing B-mode map in fourier [and power]-space  
    """
    import flipperPol as fp
    
    lDir=a.full_lens_dir+'%sdeg%s/' %(map_size,sep)
    lTmap=liteMap.liteMapFromFits(lDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    lQmap=liteMap.liteMapFromFits(lDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    lUmap=liteMap.liteMapFromFits(lDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    lUmap.data*=-1. # for U-flip convention

    maskMap=liteMap.liteMapFromFits(lDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    
    # Compute zero-padded maps (including mask map)
    from .PaddedPower import zero_padding
    zTmap=zero_padding(lTmap,padding_ratio)
    zQmap=zero_padding(lQmap,padding_ratio)
    zUmap=zero_padding(lUmap,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    
    del lTmap,lQmap,lUmap
    
    # Compute window factor <W^2> for padded window (since this is only region with data)
    windowFactor=np.mean(zWindow.data**2.)
    
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary
    _,_,fB=fp.fftPol.TQUtoPureTEB(zTmap,zQmap,zUmap,zWindow,modL,angL,method='standard') # use standard since no E-modes present
    
    del zTmap,zQmap,zUmap,modL,angL
    
    # Rescale to correct delensing fraction
    fB.kMap*=np.sqrt(delensing_fraction)

    # Account for window factor - this accounts for power loss due to padding
    fB.kMap/=np.sqrt(windowFactor)
    #fE.kMap/=np.sqrt(windowFactor)
    #fT.kMap/=np.sqrt(windowFactor)
    
    if power:
        BBcorr=fftTools.powerFromFFT(fB)
    if fourier and power:
        return fB,BBcorr
    elif fourier:
        return fB
    elif power:
        del fB
        return BBcorr
    
