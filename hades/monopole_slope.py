import numpy as np
from hades.params import BICEP
a=BICEP()

def runner(map_id):
	from hades.PaddedPower import MakePowerAndFourierMaps
	Bpow=MakePowerAndFourierMaps(map_id,fourier=False,power=True)
	
	from scipy.optimize import curve_fit
	def model(logell,logA,slope):
    		return logA-slope*logell
    	filt=np.where((Bpow.modLMap.ravel()>90.)&(Bpow.modLMap.ravel()<2000.))
	modL=Bpow.modLMap.ravel()[filt]
	power=Bpow.powerMap.ravel()[filt]
	try:
		params,_=curve_fit(model,np.log10(modL),np.log10(power),p0=[-11.,2.42])
	except RuntimeError:
		print 'could not compute'
		params=[0,0]
	A=10.**params[0];slope=params[1]
	return [A,slope]
	
if __name__=='__main__':
	import multiprocessing as mp
	p=mp.Pool()
	goodIDs=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep))
	import tqdm
	outs=list(tqdm.tqdm(p.imap_unordered(runner,goodIDs),total=len(goodIDs)))
	A=[ou[0] for ou in outs if ou[0]!=0]
	slope=[ou[1] for ou in outs if ou[1]!=0]
	np.savez(a.root_dir+'MonopoleSlopes.npz',A=A,slope=slope)
	print 'complete'
	

