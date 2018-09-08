from hades.params import BICEP
a=BICEP()
import numpy as np
from flipper import *


if __name__=="__main__":
	import multiprocessing as mp
	
	p=mp.Pool()
	
	indices=np.arange(500)
	
	from hades.GaussianityTest import MC_variations
	import tqdm
	
	out=list(tqdm.tqdm(p.imap_unordered(MC_variations,indices),total=len(indices)))
	
	np.save('BICEP2/MCvariations2.npy',out)
	

def MC_variations(dummy,map_id=18388,padding_ratio=a.padding_ratio,map_size=a.map_size,\
	sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,\
	slope=a.slope,lMin=a.lMin,lMax=a.lMax,KKmethod=a.KKmethod):
	"""Compute MC values of estimated quantities (yanked from PaddedPower.padded_estimates)"""
	
	print 'Starting %s' %dummy
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
	#A_est,fs_est,fc_est,Afs_est,Afc_est=zero_estimator(totMap.copy(),lMin=lMin,lMax=lMax,slope=slope,factor=1e-10,FWHM=FWHM,noise_power=noise_power,KKmethod=KKmethod)
	# (Factor is expected monpole amplitude (to speed convergence))
	
	# Compute anisotropy fraction and angle
	#ang_est=0.25*180./np.pi*np.arctan(Afs_est/Afc_est) # in degrees
	#frac_est=np.sqrt(fs_est**2.+fc_est**2.)
		
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
		
	return A_MC,fs_MC,fc_MC,Afs_MC,Afc_MC,epsilon_MC,ang_MC
	
def reconstructor():
	""" Reconstruct and output the MC data"""
	dat=np.load('BICEP2/MCvariations.npy')
	
	A,fs,fc,Afs,Afc,epsilon,ang=[],[],[],[],[],[],[]
	for d in dat:
		for i in range(len(d[0])):
			A.append(d[0][i])
			fs.append(d[1][i])
			fc.append(d[2][i])
			Afs.append(d[3][i])
			Afc.append(d[4][i])
			epsilon.append(d[5][i])
			ang.append(d[6][i])
	import os
	if os.path.exists('BICEP2/MCvariations2.npy'):
		dat2=np.load('BICEP2/MCvariations2.npy')
	
		for d in dat2:
			for i in range(len(d[0])):
				A.append(d[0][i])
				fs.append(d[1][i])
				fc.append(d[2][i])
				Afs.append(d[3][i])
				Afc.append(d[4][i])
				epsilon.append(d[5][i])
				ang.append(d[6][i])
	return A,fs,fc,Afs,Afc,epsilon,ang


def kamerud78(r,mu_x,mu_y,sig_x,sig_y):
	""" Kamerud 1978 distribution PDF for X/Y for X,Y normally independently distributed.
	This uses form found in CG Ciao 2006 (https://www.emis.de/journals/HOA/ADS/Volume2006/78375.pdf)
	"""
	import numpy as np
	w=r*sig_y/sig_x
	
	s=(w**2.+1)**(-0.5)
	
	k=(mu_x/sig_x*w+mu_y/sig_y)*s**2.
	M=-0.5*(mu_y/sig_y*w-mu_x/sig_x)**2. * s**2.
	
	from scipy.stats import norm
	phi=norm.cdf(-k/s)
	
	Q = k*s*np.sqrt(2*np.pi)*(1.-2.*phi) + 2*s**2.*np.exp(-k**2./(2*s**2.))
	
	g=1./(2.*np.pi)*Q*np.exp(M)
	
	f=sig_y/sig_x*g
	
	return f
