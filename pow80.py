import numpy as np
from hades.params import BICEP
a=BICEP()

def runner(map_id):
	from hades.PaddedPower import MakePowerAndFourierMaps
	powB=MakePowerAndFourierMaps(map_id,fourier=False,power=True,root_dir=a.root_dir)
	filt=np.where((powB.modLMap.ravel()<180.)&(powB.modLMap.ravel()>0.))
	
	logL=-1.*np.log(powB.modLMap.ravel()[filt]/80.)
	logpow=np.log(powB.powerMap.ravel()[filt])

	from scipy.stats import linregress
	outs=linregress(logL,logpow)
	pow80_regress=np.exp(outs[1])
	
	idx=np.where((powB.modLMap.ravel()<120.)&(powB.modLMap.ravel()>70.))	
	pow85=np.mean(powB.powerMap.ravel()[idx])
	
	return [pow80_regress,pow85]
	
if __name__=='__main__':
	print a.root_dir
	import multiprocessing as mp
	p=mp.Pool()
	goodIDs=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep))
	import tqdm
	outs=list(tqdm.tqdm(p.imap_unordered(runner,goodIDs),total=len(goodIDs)))
	pow80_regress=[ou[0] for ou in outs]
	pow85=[ou[1] for ou in outs]
	np.savez(a.root_dir+'Pow80.npz',regression=pow80_regress,single=pow85)
	print 'complete'
	

