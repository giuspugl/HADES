import numpy as np
from hades.params import BICEP
a=BICEP()

def create_hyperparams():
	ELL,SLOPE,MS=np.meshgrid(a.hyp_lMin,a.hyp_slope,a.hyp_map_size)
	np.savez(a.root_dir+'hyperparams.npz',lMin=ELL.ravel(),slope=SLOPE.ravel(),map_size=MS.ravel())
	
if __name__=='__main__':
	import sys
	import os
	import time
	index = int(sys.argv[1])
	print index
	start_time=time.time()
	
	params=np.load(a.root_dir+'hyperparams.npz')
	map_size=params['map_size'][index]
	slope=params['slope'][index]
	lMin=params['lMin'][index]
	params.close()
	
	from hades.PaddedPower import padded_estimates
	
	output=padded_estimates(18388,slope=slope,lMin=lMin,map_size=map_size)
	
	# Save output to file
	outDir=a.root_dir+'BatchData/HyperparamsBICEPNoise/'
	
	if not os.path.exists(outDir): # make directory
		os.makedirs(outDir)
		
	np.save(outDir+'%s.npy' %index, output) # save output
	
	print "Job %s complete in %s seconds" %(index,time.time()-start_time)
	
def reconstruct():
	inDir=a.root_dir+'BatchData/HyperparamsBICEPNoise/'
	params=np.load(a.root_dir+'hyperparams.npz')
	eps=[]
	lMin=[]
	slope=[]
	map_size=[]
	for i in range(48):
		import os
		mapDir=inDir+'%s.npy' %i
		if os.path.exists(mapDir):
			d=np.load(mapDir)
			eps.append(d[5][0])
			lMin.append(params['lMin'][i])
			slope.append(params['slope'][i])
			map_size.append(params['map_size'][i])
			
	return lMin,slope,map_size,eps
