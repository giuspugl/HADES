import numpy as np
from hades.params import BICEP
a=BICEP()
from flipper import *

# Default parameters
nmin = 0
nmax = 1e5#1399#3484
cores = 42

if __name__=='__main__':
     """ Batch process to use all available cores to compute the KK estimators and Gaussian errors using the est_and_err function im MCerror
    Inputs are min and max file numbers. Output is saved as npy file"""

     import tqdm
     import sys
     import numpy as np
     import multiprocessing as mp
     	
     # Parameters if input from the command line
     if len(sys.argv)>=2:
         nmin = int(sys.argv[1])
     if len(sys.argv)>=3:
         nmax = int(sys.argv[2])
     if len(sys.argv)==4:
         cores = int(sys.argv[3])
     
     # Compute map IDs with non-trivial data
     all_file_ids=np.arange(nmin,nmax+1)
     import pickle
     goodMaps=pickle.load(open(a.root_dir+str(a.map_size)+'deg'+str(a.sep)+'/fvsgoodMap.pkl','rb'))
     
     file_ids=[int(all_file_ids[i]) for i in range(len(all_file_ids)) if goodMaps[i]!=False] # just for correct maps
     
     # Start the multiprocessing
     p = mp.Pool(processes=cores)
     
     # Define iteration function
     from hades.RotationNoise import iterator
     
     # Display progress bar with tqdm
     r = list(tqdm.tqdm(p.imap(iterator,file_ids),total=len(file_ids)))
     
     if not a.NoiseAnalysis:
     	# Save output
     	np.save(a.root_dir+'%sdeg%s/RotationalChiMCestimates%sdeg%s.npy' %(a.map_size,a.sep,a.map_size,a.sep),np.array(r))
     if a.NoiseAnalysis:
     	import os
     	outDir=a.root_dir+'%sdeg%s/RotationalNoiseAnalysis/' %(a.map_size,a.sep)
     	if not os.path.exists(outDir):
     		os.mkdir(outDir)
     	if a.ComparisonSetting=='FWHM':
     		np.save(outDir+'FWHM%.2f.npy' %a.FWHM,np.array(r))
     	else:
     		np.save(outDir+'NoisePower%.2f.npy' %a.noise_power,np.array(r))


def iterator(map_id):
	""" To run the iterations"""
	from hades.RotationNoise import rotation_est
	out = rotation_est(int(map_id))
	print('%s map complete' %map_id)
	return out


def rotation_est(map_id,map_size=a.map_size,sep=a.sep,N_sims=a.N_sims,noise_power=a.noise_power,FWHM=a.FWHM,slope=a.slope):
	""" Compute the estimated angle, amplitude and polarisation strength in the presence of noise, following Hu & Okamoto 2002 noise model. Error is from MC simulations. This averages the estimated quantities over map rotations to avoid errors.
	Output: list of data for A, fs, fc (i.e. output[0]-> A etc.), with structure [map estimate, MC_standard_deviation, MC_mean]
	"""
	# First calculate the B-mode map (noiseless)
	from .PowerMap import MakePower
	Bpow=MakePower(map_id,map_size=map_size,map_type='B')
	
	# Load the relevant window function
	inDir=a.root_dir+'%sdeg%s/' %(map_size,sep)
	mask=liteMap.liteMapFromFits(inDir+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits')
	
	# Compute mean square window function
	windowFactor=np.mean(mask.data**2.)
	
	# Now compute the noise power-map
	from .NoisePower import noise_map
	noiseMap=noise_map(powMap=Bpow.copy(),noise_power=a.noise_power,FWHM=a.FWHM,windowFactor=windowFactor)
	
	# Compute total map
	totMap=Bpow.copy()
	totMap.powerMap=Bpow.powerMap+noiseMap.powerMap
	
	# Initially using NOISELESS estimators
	from .KKtest import rotation_estimator
	est_data=rotation_estimator(totMap.copy(),slope=slope) # compute anisotropy parameters
		
	## Run MC Simulations	
	# First compute 1D power spectrum by binning in annuli
	from .PowerMap import oneD_binning
	l_cen,mean_pow = oneD_binning(totMap.copy(),0.8*a.lMin,1.2*a.lMax,a.l_step,binErr=False,windowFactor=windowFactor) # gives central binning l and mean power in annulus using window function corrections
	
	# Compute univariate spline model fit to 1D power spectrum
	from scipy.interpolate import UnivariateSpline
	spline_fun = UnivariateSpline(np.log10(l_cen),np.log10(mean_pow),k=4) # compute spline of log data
	
	def model_power(ell):
		return 10.**spline_fun(np.log10(ell)) # this estimates 1D spectrum for any ell
	
	# Now run MC simulation N_sims times
	ang_MC,frac_MC=[],[]
	
	from hades.NoisePower import single_MC
	
	for n in range(N_sims): # for each MC map
		MC_map=single_MC(totMap.copy(),model_power,windowFactor=windowFactor) # create random map from isotropic spectrum
		p=rotation_estimator(MC_map.copy(),slope=slope) # compute MC anisotropy parameters  
		frac_MC.append(p[1])
		ang_MC.append(p[2])
	
	# Compute mean and standard deviation of MC statistics
	frac_mean=np.mean(frac_MC)
	ang_mean=np.mean(ang_MC)
	sigma_frac=np.std(frac_MC)
	
	
	# Regroup output (as described above)
	output = [est_data[0],est_data[1],est_data[2],frac_mean,ang_mean,sigma_frac]		
	
	return output
	
def chi2_reconstructor(map_size=a.map_size,sep=a.sep):
	""" Plotting code for the BICEP patch using the chi-squared statistics, for rotationally averaged KK derived quantities. 
	Inputs: map_size and centre separation.
	Outputs: plots saved in RotChiSkyMaps/ subdirectory."""
	
	# Load in dataset
	dat = np.load(a.root_dir+'%sdeg%s/RotationalChiMCestimates%sdeg%s.npy' %(map_size,sep,map_size,sep))
	N = len(dat)
	
	# Construct A,fs,fc arrays
	A=[d[0] for d in dat]
	frac=[d[1] for d in dat]
	ang=[d[2] for d in dat]
	MC_frac_mean=[d[3] for d in dat]
	MC_ang_mean=[d[4] for d in dat]
	sigma_frac=[d[5] for d in dat] # = sigma_f_s = sigma_f_c
	
	# Compute angle error
	sigma_ang=[sigma_frac[i]/(4.*frac[i])*180./np.pi for i in range(len(A))]
	
	# Create log monopole amplitude data 
	log_A_est=np.log10(A)
	
    	# Compute chi-squared probability
	def chi2_prob(eps,err):
		return 1.-np.exp(-eps**2./(2.*err**2.))
	
	rand_prob=[chi2_prob(frac[i],sigma_frac[i]) for i in range(len(sigma_frac))]
	
	dat_set=[A,frac,ang,MC_frac_mean,MC_ang_mean,sigma_frac,sigma_ang,log_A_est,rand_prob]
	
	# Load coordinates of patch centres
	from .NoisePower import good_coords
	ra,dec=good_coords(map_size,sep,N)
	
    	# Make output directory
    	import os
    	outDir=a.root_dir+'RotChiSkyMaps%sdeg%s/' %(map_size,sep)
    
    	if not os.path.exists(outDir):
    		os.mkdir(outDir)
    
    	# Now plot on grid:
    	import matplotlib.pyplot as plt
    	names = ['Monopole Amplitude','Anisotropy Fraction','Polarisation Angle','MC Mean Fraction',\
    	'MC Mean Angle','fs,fc Error','Angle Error','log Anisotropy Fraction','Chi-Squared Probability of Randomness',]
    	name_str=['A_est','frac_est','ang_est','MC_frac_mean','MC_ang_mean','fsfcError','AngleError','logA','chi2_prob']
    	
    	for j in range(len(names)):
    		plt.figure()
    		plt.scatter(ra,dec,c=dat_set[j],marker='o',s=80)
    		plt.title(names[j])
    		plt.colorbar()
    		plt.savefig(outDir+name_str[j]+'.png',bbox_inches='tight')
    		plt.clf()
    		plt.close()
    		
