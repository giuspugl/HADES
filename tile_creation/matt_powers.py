




import sys
sys.path.append('/scratch/r/rbond/engelen/aveTools')


from flipperPol import *
from flipper import *
from scipy import *


npatches = 4

powers = [None]  * npatches




mattloc = '/scratch/r/rbond/engelen/coadd_maps/140215/'

patchnames = ['deep1', 'deep2', 'deep5', 'deep6']




for p in [0,1, 2, 3]:

    zz = aveTools.allpowers('../data/mapTcuts_' + patchnames[p] + '_' , '.fits'
