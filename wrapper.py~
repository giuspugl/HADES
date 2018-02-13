import numpy as np
from hades.params import BICEP
a=BICEP()

from flipper import *

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
	
	outDir=a.root_dir+'BatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
	
	if a.remakeErrors:
		if os.path.exists(outDir+'%s.npy' %batch_id):
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

		
	# Now run the estimation
	from hades.wrapper import best_estimates
	output=best_estimates(map_id)
	
	# Save output to file
	if not os.path.exists(outDir): # make directory
		os.makedirs(outDir)
		
	np.save(outDir+'%s.npy' %batch_id, output) # save output
	
	print "Job %s complete in %s seconds" %(batch_id,time.time()-start_time)
	
	if batch_id==len(goodIDs)-2:
		if a.send_email:
			from hades.NoiseParams import sendMail
			sendMail('Single Map')
	
def std_approx(data):
	"""Approximate the standard deviation by the 15.865, 84.135 percentiles. This reduces errors from some large data points."""		
	if True:
		import numpy as np
		return np.std(data)
	# LEGACY:
	if False:
		import numpy as np
		hi,lo=np.percentile(data,[84.135,15.865])
		return (hi-lo)/2.
	

def best_estimates(map_id,padding_ratio=a.padding_ratio,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,lMin=a.lMin,lMax=a.lMax,KKmethod=a.KKmethod,rot=a.rot,freq=a.freq,\
	delensing_fraction=a.delensing_fraction,debiasA=a.debiasA,useTensors=a.useTensors):
	""" Compute the estimated angle, amplitude and polarisation fraction with noise, using zero-padding.
	Noise model is from Hu & Okamoto 2002 and errors are estimated using MC simulations, which are all saved.
	
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
	rot (angle to rotate by before applying estimators)
	freq (desired map frequency; 150 GHz for BICEP, 353 GHz for Vansyngel)
	delensing_fraction (efficiency of delensing; i.e. 0.1=90% removed)
	debiasA (Boolean, whether to remove bias of monopole amplitude using noise-only sims)
	useTensors (Boolean, whether to include tensor noise from IGWs with r = 0.1)
	
	Output: First 6 values: [estimate,isotropic mean, isotropic stdev] for {A,Afs,Afc,fs,fc,str,ang}
	7th: full data for N_sims as a sequence of 7 lists for each estimate (each of length N_sims)
	8th: full data for N_sims for the monopole bias term
	9th: [estimate,isotropic mean, isotropic stdev] for Hexadecapole power
	"""
	
	# First compute high-resolution B-mode map from padded-real space map with desired padding ratio
	from .PaddedPower import MakePaddedPower
	Bpow=MakePaddedPower(map_id,padding_ratio=padding_ratio,map_size=map_size,sep=sep,freq=freq)
	
	# Input directory:
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
	
	# Compute the (noise + lensing) power map using the B-mode map as a template
	# First compute the total noise (instrument+lensing+tensors)
	from .NoisePower import noise_model,lensed_Cl,r_Cl
	Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction) # function for lensed Cl
	
	if useTensors: # include r = 0.1 estimate
		Cl_r_func=r_Cl()
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)+Cl_r_func(l)
	else:
		def total_Cl_noise(l):
			return Cl_lens_func(l)+noise_model(l,FWHM=FWHM,noise_power=noise_power)
		
	## Now compute the actual data estimated quantities
	
	from .RandomField import fill_from_model
	noiselensedMap=Bpow.copy() # template

	noiselensedMap.powerMap=fill_from_model(Bpow,total_Cl_noise)

	#from .NoisePower import noise_map	
	#noiselensedMap=noise_map(powMap=Bpow.copy(),noise_power=noise_power,FWHM=FWHM,\
	#,delensing_fraction=delensing_fraction)

	# Compute total map
	totMap=Bpow.copy()
	totMap.powerMap=Bpow.powerMap+noiselensedMap.powerMap
	
	# Apply the KK estimators
	from .KKtest import zero_estimator
	A_est,fs_est,fc_est,Afs_est,Afc_est=zero_estimator(totMap.copy(),map_id,lMin=lMin,\
		lMax=lMax,slope=slope,factor=None,FWHM=FWHM,noise_power=noise_power,\
		KKmethod=KKmethod,rot=rot,\
		delensing_fraction=delensing_fraction,useTensors=a.useTensors)
	# (Factor is expected monopole amplitude (to speed convergence))
	# NB: Afs, Afc are the fundamental quantities here - all others must be bias-corrected
	
	# Define function to estimate standard deviation from percentiles (avoids outliers, assuming Gaussianity)
	from .wrapper import std_approx
	
	# Compute monopole bias from noise-only MC sims
	if debiasA:
		noiseA=np.zeros(N_sims)
		from .KKtest import A_estimator
		from .RandomField import fill_from_model
		for n in range(N_sims):
			if n%50==0:
				print 'MapID %s: Starting noise-only simulation %s of %s' %(map_id,n+1,N_sims)
			# Create random noise+lensing-only sims
			noiselensedMap=Bpow.copy()
			noiselensedMap.powerMap=fill_from_model(Bpow,total_Cl_noise)
			# Now compute estimated amplitude
			noiseA[n]=A_estimator(noiselensedMap.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=A_est,FWHM=FWHM,noise_power=noise_power,KKmethod=KKmethod,\
			rot=rot,delensing_fraction=delensing_fraction,useTensors=a.useTensors)
		A_bias=np.mean(noiseA)
		A_bias_err=std_approx(noiseA)
	else:
		A_bias=0. #  for compatibility
		A_bias_err=0.
		noiseA=0.
	
			
	print 'Estimated debiased amplitude: %.3e, Bias term: %.3e +- %.3e' %(A_est-A_bias,A_bias,A_bias_err)
	
	# Now correct terms for bias
	A_est_biased=A_est # needed for later KK test 
	A_est = A_est-A_bias
	fs_est=Afs_est/A_est
	fc_est=Afc_est/A_est
	
	## Run MC Simulations	
	
	# Compute rough semi-analytic C_ell spectrum
	def analytic_model(ell,A_est,slope):
		"""Use the estimate for A to construct analytic model.
		NB: This is just used for finding the centres of the actual binned data.
		"""
		return total_Cl_noise(ell)+A_est*ell**(-slope)
	
	# Compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totMap.copy(),0.8*a.lMin,1.*a.lMax,0.8*a.l_step,binErr=False,exactCen=a.exactCen,C_ell_model=analytic_model,params=[A_est,slope]) 
	
	# gives central binning l and mean power in annulus using window function corrections (from unpaddded map)
	
	# Compute univariate spline model fit to 1D power spectrum
	#from scipy.interpolate import UnivariateSpline
	#spline_fun = UnivariateSpline(np.log10(l_cen),np.log10(mean_pow),k=4) # compute spline of log data
	
	#def model_power(ell):
	#	return 10.**spline_fun(np.log10(ell)) # this estimates 1D spectrum for any ell
	
	# Initialise arrays
	A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow_MC=[[] for _ in range(8)]
	
	#from hades.NoisePower import single_MC
	from .RandomField import fill_from_Cell
	MC_map=Bpow.copy()
	
	for n in range(N_sims): # for each MC map
		if n%50==0:
			print('MapID %s: Starting simulation %s of %s' %(map_id,n+1,N_sims))
		MC_map.powerMap=fill_from_Cell(totMap,l_cen,mean_pow)
		#MC_map=single_MC(totMap.copy(),model_power) # create random map from isotropic spectrum
		output=zero_estimator(MC_map.copy(),map_id,lMin=lMin,lMax=lMax,\
			slope=slope,factor=A_est_biased,FWHM=FWHM,noise_power=noise_power,\
			KKmethod=KKmethod,rot=rot,\
			delensing_fraction=delensing_fraction,useTensors=a.useTensors) 
		# compute MC anisotropy parameters  
		A_MC.append(output[0]-A_bias)
		fs_MC.append(output[3]/(output[0]-A_bias))
		fc_MC.append(output[4]/(output[0]-A_bias))
		Afs_MC.append(output[3])
		Afc_MC.append(output[4])
		epsilon_MC.append(np.sqrt((output[3]**2.+output[4]**2.)/((output[0]-A_bias)**2.)))#fc_MC[-1]**2.+fs_MC[-1]**2.))
		ang_MC.append(0.25*180./np.pi*np.arctan(output[3]/output[4]))
		HexPow_MC.append(np.sqrt(output[3]**2.+output[4]**2.))
		
	allMC=[A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC,HexPow_MC]
	
	# Compute anisotropy fraction and angle from data
	ang_est=0.25*180./np.pi*(np.arctan(Afs_est/Afc_est)) # in degrees
	frac_est=np.sqrt(fs_est**2.+fc_est**2.) # already corrected for rotation
	HexPow_est=np.sqrt(Afs_est**2.+Afc_est**2.) # biased hexadecapolar power
	
	# Compute means and standard deviations
	A_mean=np.mean(A_MC)
	A_std=std_approx(A_MC)
	fc_mean=np.mean(fc_MC)
	fs_mean=np.mean(fs_MC)
	fc_std=std_approx(fc_MC)
	fs_std=std_approx(fs_MC)
	frac_mean=np.mean(epsilon_MC)
	frac_std=std_approx(epsilon_MC) # NB: frac, ang, HexPow are NOT Gaussian so these have limited use
	ang_mean=np.mean(ang_MC)
	ang_std=std_approx(ang_MC)
	HexPow_mean=np.mean(HexPow_MC)
	HexPow_std=std_approx(HexPow_MC)
	Afs_mean=np.mean(Afs_MC)
	Afc_mean=np.mean(Afc_MC)
	Afs_std=std_approx(Afs_MC)
	Afc_std=std_approx(Afc_MC)
	
	# Regroup data
	Adat=[A_est,A_mean,A_std]
	fsdat=[fs_est,fs_mean,fs_std]
	fcdat=[fc_est,fc_mean,fc_std]
	Afsdat=[Afs_est,Afs_mean,Afs_std]
	Afcdat=[Afc_est,Afc_mean,Afc_std]
	fracdat=[frac_est,frac_mean,frac_std]
	angdat=[ang_est,ang_mean,ang_std]
	HexPowdat=[HexPow_est,HexPow_mean,HexPow_std]
	
	# Return all output
	return Adat,fsdat,fcdat,Afsdat,Afcdat,fracdat,angdat,allMC,noiseA,HexPowdat
	
def stats_and_plots(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,\
	freq=a.freq,delensing_fraction=a.delensing_fraction,makePlots=False):
	""" Function to create plots for each tile.
	MakePlots command creates plots of epsilon histogram in the Maps/HistPlots/ directory.
	Other plots are saved in the Maps/ directory """
	import warnings # catch rogue depracation warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
	
	print 'DEPRACATED: USE hexPow_stats'
	
	import matplotlib.pyplot as plt
	from scipy.stats import percentileofscore
	import os
	
	# Import good map IDs
	goodMaps=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	
	# Define arrays
	A,Afs,Afc,fs,fc,ang,frac,probA,probP,logA,epsDeb,HPow,HPowDeb,HPowSig=[np.zeros(len(goodMaps)) for _ in range(14)]
	A_err,Af_err,f_err,ang_err,frac_err,frac_mean=[np.zeros(len(goodMaps)) for _ in range(6)]
	
	# Define output directories:
	outDir=a.root_dir+'Maps/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
	histDir=a.root_dir+'Maps/HistPlots/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
	
	if not os.path.exists(histDir):
		os.makedirs(histDir)
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	
	# Iterate over maps:
	for i in range(len(goodMaps)):
		map_id=goodMaps[i] # map id number
		if i%100==0:
			print 'loading %s of %s' %(i+1,len(goodMaps))
		# Load in data from tile
		data=np.load(a.root_dir+'BatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,i))
		
		# Load in data
		A[i],fs[i],fc[i],Afs[i],Afc[i],frac[i],ang[i]=[d[0] for d in data[:7]]
		if A[i]>0:
			logA[i]=np.log10(A[i])
		else:
			logA[i]=logA[i-1] # to avoid errors
		A_err[i],fs_err,fc_err,Afs_err,Afc_err,frac_err[i]=[d[2] for d in data[:6]]
		frac_mean[i]=data[5][1]
		
		# Compute debiased H
		HPow[i]=data[9][0]
		HPowDeb[i]=data[9][0]-data[9][1]
		HPowSig[i]=(data[9][0]-data[9][1])/data[9][2]
		
		# Compute other errors
		f_err[i]=np.mean([fs_err,fc_err])
		Af_err[i]=np.mean([Afs_err,Afc_err])
		ang_err[i]=f_err[i]/(4*frac[i])*180./np.pi
		
		# Creat epsilon plot
		eps=data[7][5] # all epsilon data
		eps_est=data[5][0]
		epsDeb[i]=eps_est-np.mean(eps)
						
		percentile=percentileofscore(eps,eps_est,kind='mean') # compute percentile of estimated data
		probP[i]=percentile
		sigma_f=np.mean([data[1][2],data[2][2]]) # mean of fs,fc errors
		def eps_CDF(eps):	
			""" CDF of epsilon modified chi-squared distribution)"""
			return 1-np.exp(-eps**2./(2.*sigma_f**2.))
		def eps_PDF(eps):
			""" PDF of epsilon modified chi-squared distribution"""
			return eps/(sigma_f**2.)*np.exp(-eps**2./(2.*sigma_f**2.)) 
		# Compute analytic CDF percentile:
		probA[i]=100.*eps_CDF(eps_est)
		
		
		# Repeat this analysis for hexadecapolar power H
		H_MC=data[7][7] # all H data
						
		percentile=percentileofscore(eps,eps_est,kind='mean') # compute percentile of estimated data
		probP[i]=percentile
		sigma_f=np.mean([data[1][2],data[2][2]]) # mean of fs,fc errors
		def eps_CDF(eps):	
			""" CDF of epsilon modified chi-squared distribution)"""
			return 1-np.exp(-eps**2./(2.*sigma_f**2.))
		def eps_PDF(eps):
			""" PDF of epsilon modified chi-squared distribution"""
			return eps/(sigma_f**2.)*np.exp(-eps**2./(2.*sigma_f**2.)) 
		# Compute analytic CDF percentile:
		probA[i]=100.*eps_CDF(eps_est)
		
		if makePlots:
			if i%10==0:
				print 'Creating plot %s of %s' %(i+1,len(goodMaps))
			y,x,_=plt.hist(eps,bins=30,range=[0,1],normed=True,alpha=0.5) # create histogram of random values
			ydat=np.linspace(0,max(y)*2,100)
			xdat=np.ones_like(ydat)*eps_est
			plt.plot(xdat,ydat,c='r',ls='--') # plot estimated value
			plt.ylabel('Epsilon PDF')
			plt.xlabel('Epsilon')
			plt.xlim(0,1)
			epsdat=np.linspace(0,1,100)
			plt.plot(epsdat,eps_PDF(epsdat),c='k') # predicted PDF
			plt.title('Tile %s // Percentile: %.2f // Analytic Percentile: %.2f' %(map_id,probP[i],probA[i]))
			plt.ylim(0,max(eps_PDF(epsdat)))
			# now save output
			plt.savefig(histDir+'%s.png' %map_id,bbox_inches='tight')
			plt.clf()
			plt.close()
	
	## Now compute the whole patch maps
	# Dataset:
	dat_set=[epsDeb,A,fs,fc,Afs,Afc,frac,ang,A_err,Af_err,f_err,frac_mean,frac_err,ang_err,probA,probP,logA,HPow,HPowDeb,HPowSig]
	names=[r'Debiased Anisotropy Fraction',r'Monopole amplitude',r'$f_s$',r'$f_c$',r'$Af_s$',r'$Af_c$',r'Anisotropy Fraction, $\epsilon$',r'Anisotropy Angle, $\alpha$',r'MC error for $A$',r'MC error for $Af$',r'MC error for $f$',r'MC mean anisotropy fraction',r'MC error for anisotropy fraction',r'MC error for angle',r'Epsilon Isotropic Percentile, $\rho$, (Analytic)',r'Epsilon Isotropic Percentile, $\rho$, (Statistical)',r'$\log_{10}(A)$',r'Hexadecapole Biased Power',r'Hexadecapole Debiased Power',r'Hexadecapole Detection Significance']
	file_str=['eps_deb','A','fs','fc','Afs','Afc','epsilon','angle','A_err','Af_err','f_err',\
	'epsilon_MC_mean','epsilon_err','ang_err','prob_analyt','prob_stat','logA','HexPow','HexPowDeb','HexPowSig']
	
	# Load coordinates of map centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,len(goodMaps))
	# Load in border of BICEP region if necessary:
	border=False
	if a.root_dir=='/data/ohep2/WidePatch/' or a.root_dir=='/data/ohep2/CleanWidePatch/' or a.root_dir=='/data/ohep2/FFP8/':
		border=True # for convenience
		from hades.plotTools import BICEP_border
		temp=BICEP_border(map_size,sep)
		if temp!=None:
			edge_ra,edge_dec=temp
			# to only use cases where border is available
		else:
			border=False
	if border!=False:
		border_coords=[edge_ra,edge_dec]
	else:
		border_coords=None # for compatibility
	
	# Now plot on grid:
	import cmocean # for angle colorbar
	for j in range(len(names)):
		print 'Generating patch map %s of %s' %(j+1,len(names))
		cmap='jet'
		minMax=None
		if file_str[j]=='angle':
			cmap=cmocean.cm.phase
		if file_str[j]=='epsilon':
			vmin=min([0,min(dat_set[j])])
			vmax=min([1.,np.percentile(dat_set[j],95)])
			minMax=[vmin,vmax]
		if file_str[j]=='fs' or file_str[j]=='fc':
			minMax=[-1,1]
		from hades.plotTools import skyMap
		# Create plot
		skyMap(dat_set[j],ra,dec,cbar_label=names[j],cmap=cmap,minMax=minMax,\
			border=border_coords,outFile=outDir+file_str[j]+'.png')
		if False: # old plotting regime depracated
			plt.figure()
			if file_str[j]=='angle':
				plt.scatter(ra,dec,c=dat_set[j],marker='o',\
				s=80,cmap=cmocean.cm.phase)
			else:
				plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80)
			if border:
				plt.plot(edge_ra,edge_dec,c='k') # plot border
			plt.title(names[j])
			plt.colorbar()
			plt.savefig(outDir+file_str[j]+'.png',bbox_inches='tight')
			plt.clf()
			plt.close()
	
	
def hexPow_stats(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,\
	freq=a.freq,delensing_fraction=a.delensing_fraction,makePlots=False):
	""" Function to create plots for each tile. These use hexadecapolar power mostly
	Plots are saved in the CorrectedMaps/ directory """
	
	
	import matplotlib.pyplot as plt
	from scipy.stats import percentileofscore
	import os
	
	# Import good map IDs
	goodDir=a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep)
	if not os.path.exists(goodDir):
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids()
		print 'creating good IDs'
	goodMaps=np.load(goodDir)
	
	# Define arrays
	A,Afs,Afc,fs,fc,ang,frac,probA,probP,logA,epsDeb,HPow,HPowDeb,HPowSig,HprobA,HprobP,logHPowMean=[np.zeros(len(goodMaps)) for _ in range(17)]
	logHexPow,logHexPowDeb,A_err,Af_err,f_err,ang_err,frac_err,frac_mean=[np.zeros(len(goodMaps)) for _ in range(8)]
	
	# Define output directories:
	outDir=a.root_dir+'CorrectedMaps/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
	
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	
	# Iterate over maps:
	for i in range(len(goodMaps)):
		map_id=goodMaps[i] # map id number
		if i%100==0:
			print 'loading %s of %s' %(i+1,len(goodMaps))
		# Load in data from tile
		data=np.load(a.root_dir+'HexBatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,i))
		
		# Load in data
		A[i],fs[i],fc[i],Afs[i],Afc[i],frac[i],ang[i]=[d[0] for d in data[:7]] # NB: only A, Afs, Afc, ang are useful
		# NB: A is already debiased
		if A[i]>0:
			logA[i]=np.log10(A[i])
		else:
			logA[i]=1000. # to avoid errors - treat these separately
			
		A_err[i],fs_err,fc_err,Afs_err,Afc_err,frac_err[i]=[d[2] for d in data[:6]]
		frac_mean[i]=data[5][1]
		
		# Compute debiased H
		HPow[i]=data[9][0]
		HPow_MC=data[7][7]
		logHPowMean[i]=np.log10(np.mean(HPow_MC))
		HPowDeb[i]=data[9][0]-data[9][1]
		HPowSig[i]=(data[9][0]-data[9][1])/data[9][2] # not very useful
		logHexPow[i]=np.log10(HPow[i])
		if HPowDeb[i]>0.:
			logHexPowDeb[i]=np.log10(HPowDeb[i])
		else:
			logHexPowDeb[i]=1000.
		
		# Compute other errors
		f_err[i]=np.mean([fs_err,fc_err]) # not useful
		Af_err[i]=np.mean([Afs_err,Afc_err])
		ang_err[i]=Af_err[i]/(4*HPow[i])*180./np.pi # error in angle
		
		# Compute statistics for Hexadecapolar power
		HprobP[i]=percentileofscore(HPow_MC,HPow[i],kind='mean') # statistical percentile of estimated data
		def H_CDF(H,sigma_Af):
			"""CDF of HexPow modified chi2 distribution"""
			return 1 - np.exp(-H**2./(2.*sigma_Af**2.))
		#def H_PDF(H,sigma_Af):
		#	"""PDF of HexPow modified Chi2 distribution"""
		#	return H/(sigma_Af**2.)*np.exp(-H**2./(2.*sigma_Af**2.))
		
		# Compute analytic CDF percentile:
		HprobA[i]=100.*H_CDF(HPow[i],Af_err[i])
		
		
		# Now repeat for epsilon - NB: this uses the debiased monopole amplitudes implicitly
		eps=data[7][5] # all epsilon data
		eps_est=data[5][0]
		epsDeb[i]=eps_est-np.mean(eps)
						
		percentile=percentileofscore(eps,eps_est,kind='mean') # compute percentile of estimated data
		probP[i]=percentile
		sigma_f=np.mean([data[1][2],data[2][2]]) # mean of fs,fc errors
		def eps_CDF(eps):	
			""" CDF of epsilon modified chi-squared distribution)"""
			return 1-np.exp(-eps**2./(2.*sigma_f**2.))
		def eps_PDF(eps):
			""" PDF of epsilon modified chi-squared distribution"""
			return eps/(sigma_f**2.)*np.exp(-eps**2./(2.*sigma_f**2.)) 
		# Compute analytic CDF percentile:
		probA[i]=100.*eps_CDF(eps_est)
		
		
	## Now compute the whole patch maps
	# First define dataset:
	dat_set = [HPow,HPowDeb,HPowSig,logHPowMean,logHexPow,logHexPowDeb,A,frac,epsDeb,ang,ang_err,HprobA,HprobP,probA,probP,logA]
	names = [r'Hexadecapole Amplitude',r'Hexadecapole Debiased Amplitude',r'Hexadecapole "Significance"',r'log(Hexadecapole MC Isotropic Mean)',\
		r'log(Hexadecapole Amplitude)',r'Log(Hexadecapole Debiased Amplitude)',r'Debiased Monopole Amplitude',r'Anisotropy Fraction',r'Debiased Epsilon',\
		r'Anisotropy Angle',r'Anisotropy Angle Error',r'Hexadecapole Percentile (Analytic)',r'Hexadecapole Percentile (Statistical)',\
		r'Epsilon Percentile (Analytic)',r'Epsilon Percentile (Statistical)','log(Monopole Amplitude)']
	file_str =['hexPow','hexPowDeb','hexPowSig','HexPowMeanMC','logHexPow','logHexPowDeb','Adebiased','eps','epsDeb','ang','angErr','hexProbAna','hexProbStat',\
		'epsProbAna','epsProbStat','logA']
	
	if len(names)!=len(file_str) or len(names)!=len(dat_set):
		raise Exception('Wrong length of files') # for testing
		
	# Load coordinates of map centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,len(goodMaps))
	# Load in border of BICEP region if necessary:
	border=False
	if a.root_dir=='/data/ohep2/WidePatch/' or a.root_dir=='/data/ohep2/CleanWidePatch/' or a.root_dir=='/data/ohep2/FFP8/':
		border=True # for convenience
		from hades.plotTools import BICEP_border
		temp=BICEP_border(map_size,sep)
		if temp!=None:
			edge_ra,edge_dec=temp
			# to only use cases where border is available
		else:
			border=False
	if border!=False:
		border_coords=[edge_ra,edge_dec]
	else:
		border_coords=None # for compatibility
	
	# Now plot on grid:
	from hades.plotTools import skyMap
	import cmocean # for angle colorbar
	for j in range(len(names)):
		print 'Generating patch map %s of %s' %(j+1,len(names))
		cmap='jet'
		minMax=None
		if file_str[j]=='ang':
			cmap=cmocean.cm.phase
		if file_str[j]=='angErr':
			minMax=[0.,45.]
		if file_str[j]=='epsDeb':
			minMax=[-5,5]
		if file_str[j]=='eps':
			vmin=min(dat_set[j])
			vmax=min([1.,np.percentile(dat_set[j],95)])
			minMax=[vmin,vmax]
		decA,raA=dec.copy(),ra.copy()
		if file_str[j]=='logA':
			ids=np.where(dat_set[j]!=1000.)
			dat_set[j]=dat_set[j][ids]
			raA=raA[ids]
			decA=decA[ids]
		if file_str[j]=='logHexPowDeb':
			ids=np.where(dat_set[j]!=1000.)
			dat_set[j]=dat_set[j][ids]
			raA=raA[ids]
			decA=decA[ids]
		# Create plot
		skyMap(dat_set[j],raA,decA,cbar_label=names[j],cmap=cmap,minMax=minMax,\
			border=border_coords,outFile=outDir+file_str[j]+'.png')
	print 'Plotting complete'

def hex_patch_anisotropy(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,
	freq=a.freq,delensing_fraction=a.delensing_fraction,N_sims=a.N_sims,suffix='',folder=None,root_dir=a.root_dir):
	"""Compute the global hexadecapole anisotropy over the patch, summing the epsilon values weighted by the S/N.
	The estimate is assumed Gaussian by Central Limit Theorem.
	Errors are obtained by computing estimate for many MC sims
	"""
	raise Exception('Now depracated. Use patch_hexadecapole in hex_wrap.py')
	# Load array of map ids
	import os
	
	goodDir=root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep)
	if not os.path.exists(goodDir):
		from hades.batch_maps import create_good_map_ids
		create_good_map_ids(map_size=map_size,sep=sep,root_dir=root_dir)
		print 'creating good IDs'
	goodMaps=np.load(root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	
	if a.I2SNR:
		I2dir = root_dir+'%sdeg%s/meanI2.npy' %(map_size,sep)
		QUdir = root_dir+'%sdeg%s/meanQU.npy' %(map_size,sep)
		import os
		if not os.path.exists(I2dir):
			raise Exception('Must compute <I^2> data by running batchI2.py')
		I2data=np.load(I2dir)
		QUdata=np.load(QUdir)
	
	# Initialize variables
	hex_patch_num=0.
	A_patch_num=0. # for mean amplitude shift
	norm=0. # normalisation
	hex_patch_MC_num=np.zeros(N_sims)
	A_patch_MC_num=np.zeros(N_sims)
	
	for i in range(len(goodMaps)):
		# Load dataset
		if root_dir=='/data/ohep2/liteBIRD/':
			ij=goodMaps[i]
		else:
			ij=i # for compatibility
		if folder==None:
			folder='HexBatchData'
		datPath=root_dir+folder+'/f%s_ms%s_s%s_fw%s_np%s_d%s/%s%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,ij,suffix)
		data=np.load(datPath)
		A=data[0][1]	
		A_est=data[0][0]
		A_MC=data[7][0]
		A_eps=data[0][2]	
		hex_est=data[9][0]
		hex_MC=data[7][7]
		hex_eps=data[9][2]
		trueA=data[10]
		
		# Compute contribution to mean epsilon
		if a.I2SNR:
			SNR=(A/hex_eps)**2.#(data[0][1]/data[0][2])**2.#/hex_eps**2.#I2data[i]**2./hex_eps**2.
		else:
			SNR=1.#(trueA/hex_eps)**2.
		hex_patch_num+=SNR*hex_est
		A_patch_num+=SNR*A_est
		norm+=SNR
		for j in range(N_sims):
			hex_patch_MC_num[j]+=SNR*hex_MC[j]
			A_patch_MC_num[j]+=SNR*A_MC[j]
			
	# Compute mean epsilon + MC values
	hex_patch=hex_patch_num/norm
	hex_patch_MC=hex_patch_MC_num/norm
	A_patch=A_patch_num/norm
	A_patch_MC=A_patch_MC_num/norm
	
	# Compute mean and standard deviation
	MC_mean=np.mean(hex_patch_MC)
	MC_std=np.std(hex_patch_MC)
	A_MC_mean=np.mean(A_patch_MC)
	A_MC_std=np.mean(A_patch_MC)
	
	# Compute significance of detection
	sigmas=(hex_patch-MC_mean)/MC_std
	sigmasA=(A_patch-A_MC_mean)/A_MC_std
	
	# Now plot
	import matplotlib.pyplot as plt
	y,x,_=plt.hist(hex_patch_MC,bins=30,normed=True)
	plt.ylabel('PDF')
	plt.xlabel('Patch Averaged Hexadecapole Amplitude')
	plt.title('%.2f Sigma // Patch Averaged Hexadecapole Amplitude // %s patches & %s sims' %(sigmas,len(goodMaps),N_sims))
	xpl=np.ones(100)*hex_patch
	ypl=np.linspace(0,max(y),100)
	plt.plot(xpl,ypl,ls='--',c='r')
	plt.ylim(0,max(y))
	outDir=root_dir+'PatchHexPowImproved/'
	import os
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	plt.savefig(outDir+'hist_f%s_ms%s_s%s_fw%s_np%s_d%s.png' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction),bbox_inches='tight')
	plt.clf()
	plt.close()
	
	print sigmas,sigmasA
	return sigmas,sigmasA
	
def epsilon_patch_anisotropy(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,noise_power=a.noise_power,
	freq=a.freq,delensing_fraction=a.delensing_fraction,N_sims=a.N_sims):
	"""Compute the global anisotropy over the patch, summing the epsilon values weighted by the S/N.
	The estimate is assumed Gaussian by Central Limit Theorem.
	Errors are obtained by computing estimate for many MC sims
	"""
	# Load array of map ids
	goodMaps=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))
	
	# Initialize variables
	epsilon_patch_num=0.
	norm=0. # normalisation
	epsilon_patch_MC_num=np.zeros(N_sims)
	
	for i in range(len(goodMaps)):
		# Load dataset
		datPath=a.root_dir+'BatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,i)
		data=np.load(datPath)		
		eps_est=data[5][0]
		eps_MC=data[7][5]
		sigma_eps=data[5][2]
		
		# Compute contribution to mean epsilon
		SNR=1.#/(sigma_eps**2.)
		epsilon_patch_num+=SNR*eps_est
		norm+=SNR
		for j in range(N_sims):
			epsilon_patch_MC_num[j]+=SNR*eps_MC[j]
			
	# Compute mean epsilon + MC values
	epsilon_patch=epsilon_patch_num/norm
	epsilon_patch_MC=epsilon_patch_MC_num/norm
	
	# Compute mean and standard deviation
	MC_mean=np.mean(epsilon_patch_MC)
	from .wrapper import std_approx
	MC_std=std_approx(epsilon_patch_MC)
	
	# Compute significance of detection
	sigmas=(epsilon_patch-MC_mean)/MC_std
	
	# Now plot
	import matplotlib.pyplot as plt
	y,x,_=plt.hist(epsilon_patch_MC,bins=30,normed=True)
	plt.ylabel('PDF')
	plt.xlabel('Patch Averaged Epsilon')
	plt.title('%.2f Sigma // Patch Averaged Epsilon// %s patches & %s sims' %(sigmas,len(goodMaps),N_sims))
	xpl=np.ones(100)*epsilon_patch
	ypl=np.linspace(0,max(y),100)
	plt.plot(xpl,ypl,ls='--',c='r')
	plt.ylim(0,max(y))
	outDir=a.root_dir+'PatchEpsilon/'
	import os
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	plt.savefig(outDir+'hist_f%s_ms%s_s%s_fw%s_np%s_d%s.png' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction),bbox_inches='tight')
	plt.clf()
	plt.close()
		
def map_plot(data,map_size=a.map_size,sep=a.sep):
	""" Simple wrapper to plot a patch map for input data array"""
	# Load array of map ids
	goodMaps=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep))

	# Load coordinates of map centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,len(goodMaps))
	
	# Now plot
	import matplotlib.pyplot as plt
	plt.figure()
	plt.scatter(ra,dec,c=data,marker='o',s=80)
	plt.colorbar()
	plt.show()
	
