import numpy as np
import multiprocessing as mp
import os
import tqdm
import sys
from hades.params import BICEP
a=BICEP()

def runner(index):
    string = 'python -u '+a.hades_dir+'hades/full_lens_wrap.py ' + str(index)
    os.system(string)
    return None
    
if len(sys.argv) > 1:
    N_samples=int(sys.argv[1]) # batch_id number
else:
    print('Please enter lower bound to no. of tiles as a command line argument')
    exit()
    
p=mp.Pool(6)
indices=np.arange(N_samples)
#p.map(runner,indices)
out=list(tqdm.tqdm(p.imap_unordered(runner,indices),total=len(indices)))

p.close()
p.join()

print 'complete'
