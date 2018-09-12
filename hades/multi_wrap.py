import numpy as np
import multiprocessing as mp
import os
import tqdm
from hades.params import BICEP
a=BICEP()

def runner(index):
    string = 'python -u '+a.hades_dir+'hades/full_lens_wrap.py ' + str(index)
    os.system(string)
    return None
    
p=mp.Pool(6)
indices=np.arange(500)
#p.map(runner,indices)
out=list(tqdm.tqdm(p.imap_unordered(runner,indices),total=len(indices)))

p.close()
p.join()

print 'complete'
