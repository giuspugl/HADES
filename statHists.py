import numpy as np
from hades.params import BICEP
a=BICEP()
from flipper import *
import flipperPol as fp

if __name__=='__main__':
	# This computes the histogram data for A, Afs, ang, H2
	map_id=146
	map_size=a.map_size
	sep=a.sep
	N_sims=a.N_sims
	N_bias=a.N_bias
	noise_power=a.noise_power
	FWHM=a.FWHM
	slope=a.slope
	l_step=a.l_step
	lMin=a.lMin
	lMax=a.lMax
	rot=a.rot
	freq=a.freq
	delensing_fraction=a.delensing_fraction
	useTensors=a.useTensors
	f_dust=a.f_dust
	rot_average=a.rot_average
	useBias=a.useBias
	padding_ratio=a.padding_ratio

	lCut=int(1.7*lMax) # maximum ell for Fourier space maps
	
	# First compute B-mode map from padded-real space map with desired padding ratio. Also compute the padded window function for later use
	from hades.PaddedPower import MakePowerAndFourierMaps,DegradeMap,DegradeFourier
	fBdust,padded_window,unpadded_window=MakePowerAndFourierMaps(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq,fourier=True,power=False,returnMasks=True)
	
	# Also compute unpadded map to give binning values without bias
	unpadded_fBdust=MakePowerAndFourierMaps(map_id,padding_ratio=1.,map_size=map_size,freq=freq,fourier=True,power=False,returnMasks=False)
	unpadded_fBdust=DegradeFourier(unpadded_fBdust,lCut) # remove high ell pixels
	
	fBdust=DegradeFourier(fBdust,lCut) # discard high-ell pixels
	padded_window=DegradeMap(padded_window.copy(),lCut) # remove high-ell data
	unpadded_window=DegradeMap(unpadded_window.copy(),lCut)
	
	unpadded_fBdust.kMap*=f_dust
	fBdust.kMap*=f_dust

	# Compute <W^2>^2 / <W^4> - this is a necessary correction for the H^2 quantities (since 4-field quantities)
	wCorrection = np.mean(padded_window.data**2.)**2./np.mean(padded_window.data**4.)

	# Input directory:
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)

	# First compute the total noise (instrument+lensing+tensors)
	from hades.NoisePower import noise_model,lensed_Cl,r_Cl
	Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction) # function for lensed Cl

	if useTensors: # include r = 0.1 estimate
		Cl_r_func=r_Cl()
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)+Cl_r_func(l)
	else:
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)

	# Now create a fourier space noise map	
	from hades.PaddedPower import fourier_noise_map
	ellNoise=np.arange(5,3000) # ell range for noise spectrum

	from hades.RandomField import fill_from_model
	#fourierNoise=fourier_noise_map

	from hades.PaddedPower import fourier_noise_test
	fourierNoise,unpadded_noise=fourier_noise_test(padded_window,unpadded_window,ellNoise,total_Cl_noise(ellNoise),padding_ratio=padding_ratio,unpadded=False)

	# Compute total map
	totFmap=fBdust.copy()
	totFmap.kMap+=fourierNoise.kMap# for total B modes
	unpadded_totFmap=unpadded_fBdust.copy()
	unpadded_totFmap.kMap+=unpadded_noise.kMap

	# Now convert to power-space
	totPow=fftTools.powerFromFFT(totFmap) # total power map
	Bpow=fftTools.powerFromFFT(fBdust) # dust only map
	unpadded_totPow=fftTools.powerFromFFT(unpadded_totFmap)
				
	# Compute true amplitude using ONLY dust map
	from hades.KKdebiased import derotated_estimator
	p=derotated_estimator(Bpow.copy(),map_id,lMin=lMin,lMax=lMax,slope=slope,factor=None,FWHM=0.,\
			noise_power=1.e-400,rot=rot,delensing_fraction=0.,useTensors=False,debiasAmplitude=False,rot_average=rot_average)
	trueA=p[0]
	del Bpow		

	# Compute rough semi-analytic C_ell spectrum
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)

	# Compute anisotropy parameters
	A_est,fs_est,fc_est,Afs_est,Afc_est,finalFactor=derotated_estimator(totPow.copy(),map_id,lMin=lMin,\
		lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,rot=rot,\
		delensing_fraction=delensing_fraction,useTensors=useTensors,debiasAmplitude=True,rot_average=rot_average)
	# (Factor is expected monopole amplitude (to speed convergence))

	## Run MC Simulations	

	# Compute 1D power spectrum by binning in annuli
	from hades.PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(unpadded_totPow.copy(),10,3000,l_step*padding_ratio,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 

	from hades.RandomField import padded_fill_from_Cell
	#all_sims=[]
	bias=0.
		
	def run_MC(n):
		# Create the map with a random implementation of Cell
		fourier_MC_map=padded_fill_from_Cell(padded_window.copy(),l_cen,mean_pow,lMin=lMin)
		#fourier_MC_map=fourier_MC_map.trimAtL(1.5*lMax)
		MC_map=fftTools.powerFromFFT(fourier_MC_map.copy()) # create power domain map
		
		# Now use the estimators on the MC sims
		output=derotated_estimator(MC_map.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=finalFactor,FWHM=FWHM,noise_power=noise_power,\
			rot=rot, delensing_fraction=delensing_fraction,useTensors=a.useTensors,\
			debiasAmplitude=True,rot_average=rot_average) 
		
		# Compute MC anisotropy parameters  
		A_MC=output[0]
		#fs_MC=output[3]/output[0]
		#fc_MC=output[4]/output[0]
		Afs_MC=output[3] # these are fundamental quantities here
		Afc_MC=output[4]
		#epsilon_MC[n]=np.sqrt((output[3]**2.+output[4]**2.)*wCorrection)/output[0] # NOT corrected for bias in <H^2>
		ang_MC=0.25*180./np.pi*np.arctan(output[3]/output[4]) # NB: this is not corrected for bias
		HexPow2_MC=(output[3]**2.+output[4]**2.)*wCorrection 
		#HexPow2_MC-=np.mean(HexPow2_MC)*np.ones_like(HexPow2_MC) # remove the bias (i.e. mean of H^2 from all sims)
		del MC_map
		return A_MC,Afs_MC,Afc_MC,ang_MC,HexPow2_MC,wCorrection
		
	import multiprocessing as mp
	import tqdm
	p=mp.Pool()
	N=int(100000)
	#from hades.statHists import run_MC
	out=list(tqdm.tqdm(p.imap_unordered(run_MC,range(N)),total=N))
	A=[o[0] for o in out]
	Afs=[o[1] for o in out]
	Afc=[o[2] for o in out]
	ang=[o[3] for o in out]
	h2=[o[4] for o in out]
	wCorrection=[o[5] for o in out]

	np.savez('/data/ohep2/CleanWidePatch/StatData3.npz',A=A,Afs=Afs,ang=ang,H2=h2,Afc=Afc,wCorrection=wCorrection)
	print 'Process complete'
	

