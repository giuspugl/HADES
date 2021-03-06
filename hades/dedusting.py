from hades.params import BICEP
a=BICEP()
import numpy as np
from flipper import *
import flipperPol as fp

if __name__=='__main__':
	""" This is the iterator for batch processing the map creation through HTCondor. Each map is done separately, and argument is map_id."""
	import time
	start_time=time.time()
	import sys
	import pickle
	sys.path.append('/data/ohep2/')
	sys.path.append('/home/ohep2/Masters/')
	import os
	
	batch_id=int(sys.argv[1]) # batch_id number
	
	# First load good IDs:
	goodFile=a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep)
	if a.useQU:
		t='QU'
	else:
		t='I'
	outDir=a.root_dir+'Dedusting%s/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(t,a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
	import os
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	
	if a.remakeErrors:
		if os.path.exists(outDir+'%s.npy' %batch_id):
			print 'output exists; exiting'
			sys.exit()
	
	if batch_id<110: # create first time
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids()
		print 'creating good IDs'
		
	goodIDs=np.load(goodFile)
	
	
	if batch_id>len(goodIDs)-1:
		print 'Process %s terminating' %batch_id
		sys.exit() # stop here
	
	map_id=goodIDs[batch_id] # this defines the tile used here
	
	print '%s starting for map_id %s' %(batch_id,map_id)

		
	from hades.dedusting import compute_angle
	angle,ratio=compute_angle(map_id)
		
	# Save output to file
	if not os.path.exists(outDir): # make directory
		os.makedirs(outDir)
		
	np.save(outDir+'A%s.npy' %batch_id, angle) # save output
	np.save(outDir+'R%s.npy' %batch_id, ratio) # save output
	
	print "Job %s complete in %s seconds" %(batch_id,time.time()-start_time)
	
	if batch_id==len(goodIDs)-2:
		if a.send_email:
			from hades.NoiseParams import sendMail
			sendMail('Angle + Ratio Maps')




def compute_angle(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,sep=a.sep,freq=a.freq,\
                  f_dust=a.f_dust,lMax=a.lMax,lMin=a.lMin,l_step=a.l_step,FWHM=a.FWHM,noise_power=a.noise_power,\
                  slope=a.slope,delensing_fraction=a.delensing_fraction,useQU=a.useQU,N_bias=a.N_bias):
    """Compute the polarisation angle for a specific tile, creating a model B-power spectrum + cross-spectra
    in order to find the angle including the ambiguity in sin(2alpha), cos(2alpha) due to initial computation
    of sin(4alpha), cos(4alpha).
    
    Returns angle in degrees.
    """

    # Step 1, create actual B-mode map
    lCut=int(1.35*lMax) # maximum ell for Fourier space maps

    # First compute B-mode map from padded-real space map with desired padding ratio. Also compute the padded window function for later use
    from hades.PaddedPower import MakePowerAndFourierMaps,DegradeMap,DegradeFourier
    fBdust,padded_window,unpadded_window=MakePowerAndFourierMaps(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq,fourier=True,power=False,returnMasks=True,flipU=a.flipU)

    # Also compute unpadded map to give binning values without bias
    unpadded_fBdust=MakePowerAndFourierMaps(map_id,padding_ratio=1.,map_size=map_size,freq=freq,fourier=True,power=False,returnMasks=False,flipU=a.flipU)
    unpadded_fBdust=DegradeFourier(unpadded_fBdust,lCut) # remove high ell pixels

    fBdust=DegradeFourier(fBdust,lCut) # discard high-ell pixels
    padded_window=DegradeMap(padded_window.copy(),lCut) # remove high-ell data
    unpadded_window=DegradeMap(unpadded_window.copy(),lCut)

    unpadded_fBdust.kMap*=f_dust
    fBdust.kMap*=f_dust

    wCorrection = np.mean(padded_window.data**2.)**2./np.mean(padded_window.data**4.)

    from hades.NoisePower import noise_model,lensed_Cl,r_Cl
    Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction) # function for lensed Cl

    def total_Cl_noise(l):
        return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)

    from hades.PaddedPower import fourier_noise_map
    ellNoise=np.arange(5,lCut) # ell range for noise spectrum

    from hades.RandomField import fill_from_model
    #fourierNoise=fourier_noise_map

    from hades.PaddedPower import fourier_noise_test
    fourierNoise,unpadded_noise=fourier_noise_test(padded_window,unpadded_window,ellNoise,total_Cl_noise(ellNoise),padding_ratio=padding_ratio,unpadded=False,log=True)

    totFmap=fBdust.copy()
    totFmap.kMap+=fourierNoise.kMap# for total B modes
    unpadded_totFmap=unpadded_fBdust.copy()
    unpadded_totFmap.kMap+=unpadded_noise.kMap

    fBtrue=totFmap.copy()

    # Step 2: Compute the I map
    inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
    Tmap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
    Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
    Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
    Umap.data*=-1.
    QUmap=Qmap.copy()
    QUmap.data=np.sqrt(Qmap.data**2.+Umap.data**2.)
    if useQU:
    	scaling=np.mean(QUmap.data**4.)
    else:
    	scaling=np.mean(Tmap.data**4.)
    
    maskMap=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
    from hades.PaddedPower import zero_padding
    zTmap=zero_padding(Tmap,padding_ratio)
    zQUmap=zero_padding(QUmap,padding_ratio)
    zWindow=zero_padding(maskMap,padding_ratio)
    # Compute window factor <W^2> for padded window (since this is only region with data)
    windowFactor=np.mean(zWindow.data**2.)

    # Define mod(l) and ang(l) maps needed for fourier transforms
    modL,angL=fp.fftPol.makeEllandAngCoordinate(zTmap) # choice of map is arbitary
    # Create pure T,E,B maps using 'hybrid' method to minimize E->B leakage
    zTmap.data*=zWindow.data
    zQUmap.data*=zWindow.data
    fT=fftTools.fftFromLiteMap(zTmap)
    fQU=fftTools.fftFromLiteMap(zQUmap)

    # Rescale to correct amplitude using dust SED
    from hades.PowerMap import dust_emission_ratio
    dust_intensity_ratio=dust_emission_ratio(freq)

    fT.kMap*=dust_intensity_ratio # apply dust-reduction factor 
    fT.kMap/=np.sqrt(windowFactor)
    fQU.kMap*=dust_intensity_ratio
    fQU.kMap/=np.sqrt(windowFactor)
    fImap=DegradeFourier(fT,lCut)
    fQUmap=DegradeFourier(fQU,lCut)

    # Step 3: Compute angle estimate
    powBtrue=fftTools.powerFromFFT(fBtrue)
    unpadded_powBtrue=fftTools.powerFromFFT(unpadded_totFmap)
    from hades.KKdebiased import derotated_estimator
    output=derotated_estimator(powBtrue,map_id,lMin=lMin,lMax=lMax,FWHM=FWHM,noise_power=noise_power,delensing_fraction=delensing_fraction,slope=slope)
    A,fs,fc,Afs,Afc,_=output
    HexPow2=Afs**2.+Afc**2.
    
    if a.debias_dedust:
    	from .RandomField import padded_fill_from_Cell
	bias_data=np.zeros(N_bias)
	
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)
	
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(unpadded_powBtrue.copy(),lMin*padding_ratio,lCut,l_step*padding_ratio,binErr=False,exactCen=a.exactCen,\
					C_ell_model=analytic_model,params=[A,slope]) 
	#l_cen,mean_pow=oneD_binning(totPow.copy(),lMin,lCut,l_step,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope])
	# gives central binning l and mean power in annulus using window function corrections 
	
	# Create spline fit
	from scipy.interpolate import UnivariateSpline
	spl=UnivariateSpline(l_cen,np.log(mean_pow),k=5)
	def spline(ell):
		return np.exp(spl(ell))
	#del l_cen,mean_pow
	
	# Precompute useful data:
	from hades.RandomField import precompute
	precomp=precompute(padded_window.copy(),spline,lMin=lMin,lMax=lMax)
	
	for n in range(N_bias):
		if n%100==0:
			print 'Computing bias sim %s of %s' %(n+1,N_bias)
		fBias=padded_fill_from_Cell(padded_window.copy(),l_cen,mean_pow,lMin=lMin,unPadded=a.unPadded,precomp=precomp)#,padding_ratio=padding_ratio)
		bias_cross=fftTools.powerFromFFT(fBias.copy(),totFmap.copy()) # cross map
		bias_self=fftTools.powerFromFFT(fBias.copy()) # self map
		# First compute estimators on cross-spectrum
		cross_ests=derotated_estimator(bias_cross.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,\
						factor=A,FWHM=FWHM,noise_power=noise_power,\
						rot=a.rot,delensing_fraction=delensing_fraction,useTensors=a.useTensors,\
						debiasAmplitude=False,rot_average=a.rot_average,KKdebiasH2=False) # NB: CHANGE DEBIAS_AMPLITUDE parameter here
		self_ests=derotated_estimator(bias_self.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,\
						factor=A,FWHM=FWHM,noise_power=noise_power,\
						rot=a.rot,delensing_fraction=delensing_fraction,useTensors=a.useTensors,\
						debiasAmplitude=True,rot_average=a.rot_average,KKdebiasH2=a.KKdebiasH2)
		bias_data[n]=(-1.*(self_ests[3]**4.+self_ests[4]**2.)+4.*(cross_ests[3]**2.+cross_ests[4]**2.))*wCorrection
	# Now compute the mean bias - this debiases the DATA only
	bias=np.mean(bias_data)
	del bias_self,bias_cross
	
    HexPow2-=bias	
    
    norm=np.sqrt(Afs**2.+Afc**2.)
    fsbar,fcbar=Afs/norm,Afc/norm

    sin2a=fsbar/np.sqrt(2.*(fcbar+1.))
    cos2a=np.sqrt((1.+fcbar)/2.)

    # Step 4: Compute B estimate
    angleMap=fImap.thetaMap*np.pi/180.
    fB_est=fImap.copy()
    if useQU:
    	baseMap=fQUmap.copy()
    else:
    	baseMap=fImap.copy()
    fB_est.kMap=baseMap.kMap*(sin2a*np.cos(2.*angleMap)-cos2a*np.sin(2.*angleMap))

    # Step 5: Now compute cross coefficient
    crossPow=fftTools.powerFromFFT(fB_est,fBtrue)
    estPow=fftTools.powerFromFFT(fB_est,fB_est)

    from hades.PowerMap import oneD_binning
    lC,pC=oneD_binning(crossPow,lMin,lMax/2.,l_step,exactCen=False)
    lE,pE=oneD_binning(estPow,lMin,lMax/2.,l_step,exactCen=False)
    lB,pB=oneD_binning(powBtrue,lMin,lMax/2.,l_step,exactCen=False)
    #rho=np.array(pC)/np.sqrt(np.array(pB)*np.array(pE))
    ratio=np.array(pC)/np.array(pE)
    sign=np.sign(np.mean(ratio))

    # Step 6: Now compute the actual angle
    alpha0=0.25*np.arctan2(fsbar,fcbar) # range is [-pi/4,pi/4]
    if sign==-1.0:
        alpha0+=np.pi/2.
       
    # Step 7: Compute the ratio of H^2/<I^4> for rescaling
    ratio=(np.abs(HexPow2)/scaling)**0.25
    
    alpha_deg=alpha0*180./np.pi
    print 'MapID: %s Angle: %.2f Ratio: %.2e' %(map_id,alpha_deg,ratio)
    
    return alpha_deg,ratio
