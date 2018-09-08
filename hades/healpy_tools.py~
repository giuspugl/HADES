import numpy as np
from flipper import *
from hades.params import BICEP
a=BICEP()

def create_map(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,fourier=True,power=True,\
		returnMasks=False,FWHM=a.FWHM,noise_power=a.noise_power,delensing_fraction=a.delensing_fraction,\
		f_dust=a.f_dust,f_noise=1.):
	""" Function to create 2D B-mode power map from real space map padded with zeros. 
   	This uses presaved cut-outs of separate full-sky dust, lensing + noise maps.
   	Input: map_id (tile number)
   	map_size (in degrees)
    	sep (separation of map centres)
    	padding_ratio (ratio of padded map width to original (real-space) map width)
    	fourier (return fourier space map?)
    	power (return power map?)
    	returnMasks (return real-space mask windows?)
    	FWHM,noise_power,delensing_fraction -> noise + lensing parameters
    	f_noise - scaling ratio for noise - testing only

    	Output: B-mode map in power-space , B-mode map in Fourier-space  
    	"""
    
	if noise_power!=1.:
		raise Exception('Only S4 noise map is implemented')
	if FWHM!=1.5:
		raise Exception('Only S4 noise map is implemented')
		
	if padding_ratio!=1.:
		raise Exception('Window function effects not yet removed')
		
	import flipperPol as fp
	root_dir='/data/ohep2/HealpyTest/'
	folder='%sdeg%s/' %(map_size,sep)
		
    	# Read in original maps from file
    	lensBmap=liteMap.liteMapFromFits(root_dir+'Lens/'+folder+'fvsmapB_'+str(map_id).zfill(5)+'.fits')
    	noiseBmap=liteMap.liteMapFromFits(root_dir+'Noise/'+folder+'fvsmapB_'+str(map_id).zfill(5)+'.fits')
    	dustBmap=liteMap.liteMapFromFits(root_dir+'Dust/'+folder+'fvsmapB_'+str(map_id).zfill(5)+'.fits')
    	maskMap=liteMap.liteMapFromFits(root_dir+'Dust/'+folder+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    	
    	# Now compute total Bmap
    	Bmap=dustBmap.copy()
    	Bmap.data=dustBmap.data*f_dust+lensBmap.data*np.sqrt(delensing_fraction)+noiseBmap.data*f_noise
    	
    	# Compute zero-padded maps (including mask map)	
    	from .PaddedPower import zero_padding
    	zBmap=zero_padding(Bmap,padding_ratio)
    	zWindow=zero_padding(maskMap,padding_ratio)
    	
    	# Now multiply by the mask:
    	#zBmap.data*=zWindow.data
    
    	# Compute window factor <W^2> for padded window (since this is only region with data)
    	windowFactor=np.mean(zWindow.data**2.)

    	# Define mod(l) and ang(l) maps needed for fourier transforms
    	modL,angL=fp.fftPol.makeEllandAngCoordinate(zBmap) # choice of map is arbitary

   	fB=fftTools.fftFromLiteMap(zBmap)
   	
    	# Account for window factor - this accounts for power loss due to padding
    	#fB.kMap/=np.sqrt(windowFactor)
    
    	# Mask central pixel
    	index=np.where(fB.modLMap==min(fB.modLMap.ravel())) # central pixel
    	rest_index=np.where(fB.modLMap!=min(fB.modLMap.ravel())) 
    	index2=np.where(fB.modLMap==min(fB.modLMap[rest_index].ravel())) # next to centre
    	fB.kMap[index]=np.mean(fB.kMap[index2]) # replace pixel
    	del index,rest_index,index2
    
    	if power:
    		# Transform into power space
    		BB=fftTools.powerFromFFT(fB)
    		#_,_,_,_,_,_,_,_,BB=fp.fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT,fE,fB)
    	# Now account for power loss due to padding:
    	#BB.powerMap*=zTmap.powerFactor # no effect here
	#BB.powerMap*=dust_intensity_ratio**2. # square since applied to power-maps
    
    	#BB.powerMap/=windowFactor
    	#BB.windowFactor=windowFactor # store window factor
    	if returnMasks:
        	if fourier and power:
    			return fB,BB,zWindow,maskMap
    		elif fourier:
    			return fB,zWindow,maskMap
    		elif power:
    			del fB
    			return BB,zWindow,maskMap
    	else:
    		del zWindow,maskMap
    		if fourier and power:
			return fB,BB
		elif fourier:
			return fB
		elif power:
			del fB
	    	return BB
    
def MakeCombinedFourierMaps(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,freq=a.freq,fourier=True,power=True,returnMasks=False,flipU=a.flipU,\
				dust_dir='/data/ohep2/CleanWidePatch/',root_dir=a.root_dir,delensing_fraction=a.delensing_fraction,f_noise=1.,f_dust=a.f_dust,\
				FWHM=a.FWHM,noise_power=a.noise_power,healpy_flipU=True,unPadded=a.unPadded):
    """ Function to create 2D B-mode power map from real space map padded with zeros.
    Input: map_id (tile number)
    map_size (in degrees)
    sep (separation of map centres)
    padding_ratio (ratio of padded map width to original (real-space) map width)
    freq (experiment frequency (calibrated for 100-353 GHz))
    fourier (return fourier space map?)
    power (return power map?)
    returnMasks (return real-space mask windows?)

    Output: B-mode map in power-space , B-mode map in Fourier-space  
    """
    if noise_power!=1.:
    	raise Exception('Only S4 Noise implemented')
    if FWHM!=1.5:
    	raise Exception('Only S4 Noise implemented')
    if unPadded:
    	raise Exception('unPadded is not appropriate for lensing modes')
    import flipperPol as fp
    
    # 1. DUST MAPS
    # Read in original maps from file
    inDir=dust_dir+'%sdeg%s/' %(map_size,sep)    
    Tmap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    if flipU: Umap.data*=-1.
    maskMap=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    
    # Rescale to correct amplitude using dust SED
    if a.rescale_freq:
    	from .PowerMap import dust_emission_ratio
    	dust_intensity_ratio=dust_emission_ratio(freq)
    	Tmap.data*=dust_intensity_ratio # apply dust-reduction factor 
    	Qmap.data*=dust_intensity_ratio 
    	Umap.data*=dust_intensity_ratio  
    
    # 2. NOISE MAPS
    nDir=root_dir+'Noise/%sdeg%s/' %(map_size,sep)
    nTmap=liteMap.liteMapFromFits(nDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    nQmap=liteMap.liteMapFromFits(nDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    nUmap=liteMap.liteMapFromFits(nDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    if healpy_flipU: nUmap.data*=-1. # for U-flip convention
    
    # 3. LENSING MAPS
    lDir=root_dir+'Lens/%sdeg%s/' %(map_size,sep)
    lTmap=liteMap.liteMapFromFits(lDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    lQmap=liteMap.liteMapFromFits(lDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    lUmap=liteMap.liteMapFromFits(lDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    if healpy_flipU: lUmap.data*=-1. # for U-flip convention
   
    # 4. COMPUTE CONTAMINATION MAPS
    conT=Tmap.copy();conQ=Qmap.copy();conU=Umap.copy()
    conT.data=f_noise*nTmap.data+lTmap.data*np.sqrt(delensing_fraction)
    conQ.data=f_noise*nQmap.data+lQmap.data*np.sqrt(delensing_fraction)
    conU.data=f_noise*nUmap.data+lUmap.data*np.sqrt(delensing_fraction)
    
    # Compute zero-padded maps (including mask map)
    from .PaddedPower import zero_padding
    zTmap=zero_padding(conT,padding_ratio)
    zQmap=zero_padding(conQ,padding_ratio)
    zUmap=zero_padding(conU,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    
    # Compute window factor <W^2> for padded window (since this is only region with data)
    windowFactor=np.mean(zWindow.data**2.)

    # Define mod(l) and ang(l) maps needed for fourier transforms
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary
    _,_,fBcon=fp.fftPol.TQUtoPureTEB(zTmap,zQmap,zUmap,zWindow,modL,angL,method='standard') # use standard since no E-modes present
    
    # 5. COMPUTE DUST MAPS
    zTmapD=zero_padding(Tmap,padding_ratio)
    zQmapD=zero_padding(Qmap,padding_ratio)
    zUmapD=zero_padding(Umap,padding_ratio)
     # Create pure T,E,B maps using 'hybrid' method to minimize E->B leakage
    _,_,fBdust=fp.fftPol.TQUtoPureTEB(zTmapD,zQmapD,zUmapD,zWindow,modL,angL,method='hybrid')
    
    del Tmap,Qmap,Umap,nTmap,nQmap,nUmap,lTmap,lQmap,lUmap,conT,conQ,conU
    
    # 6. COMPUTE TOTAL MAP
    fB=fBdust.copy()
    fB.kMap=fBdust.kMap*f_dust+fBcon.kMap
    
    del zTmap,zQmap,zUmap,modL,angL,zTmapD,zQmapD,zUmapD
    
    # Account for window factor - this accounts for power loss due to padding
    fB.kMap/=np.sqrt(windowFactor)
    #fE.kMap/=np.sqrt(windowFactor)
    #fT.kMap/=np.sqrt(windowFactor)
    
    # Mask central pixel
    if False:
    	index=np.where(fB.modLMap==min(fB.modLMap.ravel())) # central pixel
    	rest_index=np.where(fB.modLMap!=min(fB.modLMap.ravel())) 
    	index2=np.where(fB.modLMap==min(fB.modLMap[rest_index].ravel())) # next to centre
    	fB.kMap[index]=np.mean(fB.kMap[index2]) # replace pixel
    	del index,rest_index,index2
    
    if power:
    	# Transform into power space
    	BB=fftTools.powerFromFFT(fB)
    	#_,_,_,_,_,_,_,_,BB=fp.fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT,fE,fB)
    	# Now account for power loss due to padding:
    	#BB.powerMap*=zTmap.powerFactor # no effect here
	#BB.powerMap*=dust_intensity_ratio**2. # square since applied to power-maps
    
    	#BB.powerMap/=windowFactor
    	#BB.windowFactor=windowFactor # store window factor
    if returnMasks:
        if fourier and power:
    		return fB,BB,zWindow,maskMap
    	elif fourier:
    		return fB,zWindow,maskMap
    	elif power:
    		del fB
    		return BB,zWindow,maskMap
    else:
    	del zWindow,maskMap
    	if fourier and power:
		return fB,BB
	elif fourier:
		return fB
	elif power:
		del fB
	    	return BB
    
    
    
def create_good_map_ids(root_dir=a.root_dir,sep=a.sep,map_size=a.map_size):
	""" Create a file with the list of only good map ids in"""
	import pickle
	
	# Load in good maps
	goodMaps=pickle.load(open(root_dir+'Noise/'+str(map_size)+'deg'+str(sep)+'/fvsgoodMap.pkl','rb'))
	all_file_ids=np.arange(0,len(goodMaps))
	
	goodIds=[int(file_id) for file_id in all_file_ids if goodMaps[file_id]!=False] # just for correct maps
	
	np.save(root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep),goodIds)

