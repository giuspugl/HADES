

import numpy as np

ell = np.arange(10000)

toWrite = np.transpose([ell, np.ones(len(ell))])

np.savetxt('../data/identityBeam.txt', toWrite) 


                       
