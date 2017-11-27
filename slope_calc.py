import numpy as np
slope_width=[2,6]
amp_width=[-9,-5]
nburn=100 # burn in steps
nrun=10000 # mcmc steps
	
def find_slope(l_bin,l_step,pow_mean,pow_err):
	""" This function calculates the slope of a plot using MCMC.
	Inputs: binned l values, bin width, mean and error in binned power spectrum."""
	
	import emcee
	
	ndim, nwalkers = 2, 8 # just two parameters here
	
	p0=[np.array([np.random.uniform(slope_width[0],slope_width[1]),np.random.uniform(amp_width[0],amp_width[1])]) for _ in range(nwalkers)]
	# Choosing flat priors (both broad)
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[l_bin,l_step,pow_mean,pow_err])
	
	# Burn in steps
	print('start burn in')
	pos, prob, state = sampler.run_mcmc(p0, nburn)
	#print pos
	sampler.reset()
	
	print('run MCMC')
	sampler.run_mcmc(pos, nrun)

	return sampler.flatchain[8000:]

def lnprior(param):
	""" Log prior used here - top hat function"""
	slope=param[0]
	logA=param[1]
	
	if (slope < slope_width[1]) and (slope > slope_width[0]) and \
	(logA < amp_width[1]) and (logA > amp_width[0]):
		return float(np.log(1./((slope_width[1]-slope_width[0])*(amp_width[1]-amp_width[0]))))
	else:
		return -1.0e6 #arbitrary large -ve value

def lnprob(param, l_bin, l_step,pow_mean,pow_err):
	""" This is log of likelihood function - taken here as a Gaussian.
	We use log10(A) here for faster computations"""
	
	slope=param[0]
	A=np.power(10,param[1]) # amplitude
	
	logL=0. # initialise
	
	for i in range(len(l_bin)): # i.e. each input data point
		pred=pow_model(l_bin[i],A,slope) #prediction
		err=np.sqrt((0.5*l_step)**2+pow_err[i]**2) # summing errors in quadrature
		logL-= (pow_mean[i]-pred)**2 / (2*(err**2)) + 0.5*np.log(np.pi*2*(err**2))
	
	# This is just log(product of gaussian likelihoods)	
	return logL+lnprior(param)
	
def pow_model(l,A,slope):
	""" This is model fit, using a power law slope and amplitude A"""
	return A*(l**(-slope))
