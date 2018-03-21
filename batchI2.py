import numpy as np
from flipper import *
from hades.params import BICEP
a=BICEP()


if __name__=='__main__':
     """ Batch process to use all available cores to compute the mean I^2 value for each submap in the patch.
    Inputs are number of CPU cores. Output is saved as meanI2.npy file in the relevant data directory"""

     import tqdm
     import sys
     import multiprocessing as mp
     	
     cores = 42
     
     # Parameters if input from the command line
     if len(sys.argv)>=2:
         cores = int(sys.argv[1])
     else:
     	cores=42
        
     # Compute map IDs with non-trivial data
     file_ids=np.load(a.root_dir+'%sdeg%sGoodIDs.npy' %(a.map_size,a.sep))
     
     from hades.batchI2 import I_strength
     
     # Start the multiprocessing
     p = mp.Pool(processes=cores)
     
     # Display progress bar with tqdm
     r = list(tqdm.tqdm(p.imap(I_strength,file_ids),total=len(file_ids)))
     
     Idat=[rd[0] for rd in r]
     QUdat=[rd[1] for rd in r]
     
     np.save(a.root_dir+'%sdeg%s/meanI2.npy' %(a.map_size,a.sep),np.array(Idat))
     np.save(a.root_dir+'%sdeg%s/meanQU.npy' %(a.map_size,a.sep),np.array(QUdat))
     
def I_strength(map_id,map_size=a.map_size,sep=a.sep,root_dir=a.root_dir):
	""" Function to compute <I^2> for a map patch, from the cut-out T maps (unsmoothed)."""
	
	inDir=root_dir+'%sdeg%s/' %(map_size,sep)
	
	# Load maps
	Imap=liteMap.liteMapFromFits(inDir+'fvsmapT_'+str(map_id).zfill(5)+'.fits')
	Qmap=liteMap.liteMapFromFits(inDir+'fvsmapQ_'+str(map_id).zfill(5)+'.fits')
	Umap=liteMap.liteMapFromFits(inDir+'fvsmapU_'+str(map_id).zfill(5)+'.fits')
	
	
	# Calculate power
	meanI2 = np.mean(Imap.data.ravel()**2.)
	meanQU = np.mean(Qmap.data.ravel()**2.+Umap.data.ravel()**2.)
	
	return [meanI2,meanQU]
