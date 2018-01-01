import numpy as np
from hades.params import BICEP
a=BICEP()

def BICEP_border(map_size,sep):
    """ Compute RA/dec coordinates of edge of BICEP region.
    Output is RA,dec in degrees. 
     """
    import pickle
    
    map_dir='BICEP2/'+'%sdeg%s/' %(map_size,sep)
    import os
    if not os.path.exists(map_dir+'fvsmapRas.pkl'):
    	print 'no border'
    	return None
    
    full_ras=pickle.load(open(map_dir+'fvsmapRas.pkl','rb'))
    full_decs=pickle.load(open(map_dir+'fvsmapDecs.pkl','rb'))
    goodMap=pickle.load(open(map_dir+'fvsgoodMap.pkl','rb'))
    ra=[full_ras[i] for i in range(len(full_ras)) if goodMap[i]!=False]
    dec=[full_decs[i] for i in range(len(full_decs)) if goodMap[i]!=False]
    for i in range(len(ra)):
        if ra[i]>180.:
            ra[i]-=360.
    
    DECs=np.unique(dec)
    N=2*len(DECs)
    edge_ra,edge_dec=[np.zeros(N+1) for _ in range(2)]
    for j,D in enumerate(DECs):
        RAs=[ra[i] for i in range(len(ra)) if dec[i]==D]
        edge_ra[j]=min(RAs)
        edge_ra[N-1-j]=max(RAs)
        edge_dec[j]=D
        edge_dec[N-1-j]=D
    edge_ra[N]=edge_ra[0]
    edge_dec[N]=edge_dec[0]
    
    return edge_ra,edge_dec
