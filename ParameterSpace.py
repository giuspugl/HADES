from hades.params import BICEP
a=BICEP()
import numpy as np

def param_space(lMin):
    """Code to investigate dependence of angle and phase output on noise parameters and lMin"""
    # Define noise parameters used
    noise_pows=a.param_space_noise_powers
    fwhms=a.param_space_FWHMs
    
    # Initialise arrays
    all_angs=np.zeros((len(noise_pows),len(fwhms)))
    all_eps=np.zeros_like(all_angs)
    all_A=np.zeros_like(all_angs)
    all_A_err=np.zeros_like(all_angs)
    all_fs=np.zeros_like(all_angs)
    all_fs_err=np.zeros_like(all_angs)
    all_fc=np.zeros_like(all_angs)
    all_fc_err=np.zeros_like(all_angs)
    all_eps_err=np.zeros_like(all_angs)
    all_angs_err=np.zeros_like(all_angs)
    
    # Iterate over parameters
    count=0 # for output
    for i,n_p in enumerate(noise_pows):
        for j,fw in enumerate(fwhms):
            count+=1
            print 'Computing sample %s of %s' %(count,len(noise_pows)*len(fwhms))
            from .PaddedPower import padded_estimates
            out=padded_estimates(18388,lMin=lMin,noise_power=n_p,FWHM=fw)
            all_fs[i,j]=out[1][0]
            all_fc[i,j]=out[2][0]
            all_fs_err[i,j]=out[1][2]
            all_fc_err[i,j]=out[2][2]
            all_angs[i,j]=out[6][0]
            all_eps[i,j]=out[5][0]
            all_angs_err[i,j]=180./pi*np.mean([out[1][2],out[2][2]])/(4.*out[5][0]) # i.e. sigma_f/(4*epsilon)
            all_eps_err[i,j]=out[5][2]
            all_A[i,j]=out[0][0]
            all_A_err[i,j]=out[0][2]
    np.savez(a.root_dir+'lMin%sPaddedParamSpace.npz' %lMin,noise_power=noise_pows,FWHM=fwhms,\
    		fs=all_fs,fs_err=all_fs_err,fc=all_fc,fc_err=all_fc_err,\
    		A=all_A,A_err=all_A_err,ang=all_angs,frac=all_eps,\
    		ang_err=all_angs_err,frac_err=all_eps_err)

def plotter(map_size=a.map_size,sep=a.sep):
	""" Plot parameter space for previous defined data"""
	import matplotlib.pyplot as plt
	import cmocean
	
	outDir=a.root_dir+'%sdeg%sParamSpace/' %(map_size,sep)
	import os
	if not os.path.exists(outDir):
		os.mkdir(outDir)
	
	names=['fs','fc','fs_err','fc_err','ang','ang_err','frac','frac_err','A','A_err'] # keys for npz file
		
	for i,lMin in enumerate(a.param_space_lMin):
		data=np.load(a.root_dir+'lMin%sPaddedParamSpace.npz' %lMin)
		if i==0: # same for all data sets
			fw=data['FWHM']
			n_p=data['noise_power']
			FW,NP=np.meshgrid(fw,n_p)
		for j in range(len(names)):
			if names[j]=='ang':
				plt.scatter(FW,NP,c=data[names[j]],marker='o',s=600,edgecolor='none',cmap=cmocean.cm.phase,alpha=0.9)
			elif names[j]=='ang_err':
				plt.scatter(FW,NP,c=data[names[j]]*180./np.pi,marker='o',s=600,edgecolor='none')
			else:
				plt.scatter(FW,NP,c=data[names[j]],marker='o',s=600,edgecolor='none',alpha=0.9)
			plt.colorbar()
			plt.xlabel('FWHM / arcmin')
			plt.ylabel('Noise Power / uK-arcmin')
			plt.title('Parameter space for %s using lMin = %s' %(names[j],lMin))
			plt.savefig(outDir+'%s_lMin=%s.png' %(names[j],lMin),bbox_inches='tight')
			plt.clf()
			plt.close()
		data.close()
		
if __name__=='__main__':
	"""Multiprocess the param_space function above for predefine lMin range."""
	import multiprocessing as mp
	p=mp.Pool()
	import tqdm
	lMin=a.param_space_lMin
	
	from hades.ParameterSpace import param_space
	
	r=list(tqdm.tqdm(p.imap(param_space,lMin),total=len(lMin)))
	
	print 'Complete'
	
