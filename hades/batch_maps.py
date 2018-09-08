import numpy as np
from hades.params import BICEP
a=BICEP()

def create_good_map_ids(root_dir=a.root_dir,sep=a.sep,map_size=a.map_size):
	""" Create a file with the list of only good map ids in"""
	import pickle
	
	# Load in good maps
	goodMaps=pickle.load(open(root_dir+str(map_size)+'deg'+str(sep)+'/fvsgoodMap.pkl','rb'))
	all_file_ids=np.arange(0,len(goodMaps))
	
	goodIds=[int(file_id) for file_id in all_file_ids if goodMaps[file_id]!=False] # just for correct maps
	
	np.save(root_dir+'%sdeg%sGoodIDs.npy' %(map_size,sep),goodIds)


if __name__=='__main__':
	""" This is the iterator for batch processing the map creation through HTCondor. Each map is done separately, and argument is map_id.
	
	MOVED TO WRAPPER.PY NOW!!!!"""
	import time
	start_time=time.time()
	import sys
	import pickle
	sys.path.append('/data/ohep2/')
	sys.path.append('/home/ohep2/Masters/')
	import os
	
	raise Exception('UsingDepracatedVersion. Use wrapper.py')
	
	# First load good IDs:
	goodIDs=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep))
	
	batch_id=int(sys.argv[1]) # batch_id number
	
	if batch_id>len(goodIDs):
		print 'Process %s terminating' %batch_id
		sys.exit() # stop here
	
	map_id=goodIDs[batch_id] # this defines the tile used here
	
	print '%s starting for map_id %s' %(batch_id,map_id)
	
	# Now run the estimation
	from hades.wrapper import best_estimates
	output=best_estimates(map_id)
	
	# Save output to file
	outDir=a.root_dir+'BatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
	
	if not os.path.exists(outDir): # make directory
		os.makedirs(outDir)
		
	np.save(outDir+'%s.npy' %batch_id, output) # save output
	
	print "Job %s complete in %s seconds" %(batch_id,time.time()-start_time)
	
def reconstruct_array(map_size=a.map_size,sep=a.sep,FWHM=a.FWHM,\
	noise_power=a.noise_power,freq=a.freq,delensing_fraction=a.delensing_fraction):
	""" Code to reconstruct the array from the batch processing"""
	
	import os
	
	inDir=a.root_dir+'BatchData/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction) # input folder
	
	total_list=[]
	i=0  # iterator to read in all files
	while os.path.exists(inDir+'%s.npy' %i):
		total_list.append(np.load(inDir+'%s.npy' %i))
		i+=1
	# Now save output
	outDir=a.root_dir+'Maps/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
	
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	
	np.save(outDir+'data.npy',total_list)
	print 'Batch reconstruction complete'
	
