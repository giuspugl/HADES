
from flipper import *
from flipperPol import *
import healpy
import aveTools
import pickle
import scipy.ndimage.filters


p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])


import matplotlib.pyplot as plt

tqus = ['T', 'Q', 'U']

goodMap = pickle.load( open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'r'))
mapRas = pickle.load( open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'r'))
mapDecs = pickle.load( open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'r'))

nMaps = len(goodMap)
# nGoodMaps = np.sum(goodMap)

cosineApod = p['cosineApod']


def myTaper(indata):

    smoothed = scipy.ndimage.filters.gaussian_filter(indata, sigma = 60)

    bitmask = zeros(indata.shape)
    bitmask[np.where(smoothed > .99)] = 1.

    output = scipy.ndimage.filters.gaussian_filter(bitmask, sigma = 60)
    
    return output




firstTime = True
flipU = True

doAll = True

if doAll:
    powers = aveTools.onedl(nMaps)
    # for  mapnum in range(50):
    for mapnum in ([50]):
        if goodMap[mapnum]:
            tquMaps = [None] * 3
            print 'mapnum', mapnum


            for pol , tqu in enumerate(tqus):

                filename = p['workDir'] + p['basename'] + 'map%s_%05i.fits'%(tqu, mapnum)

                tquMaps[pol] = liteMap.liteMapFromFits(filename)
                if firstTime:

                    taper = liteMapPol.initializeCosineWindow(tquMaps[0],\
                                                              cosineApod['lenApod'],\
                                                              cosineApod['pad'])  # taper weight map

                    firstTime = False
                    
            if flipU:
                tquMaps[2].data *= -1

            maskFilename = p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%( mapnum)
            mask = liteMap.liteMapFromFits(maskFilename)

            
            


            smoothedEdges = myTaper(mask.data) 

            # maskCopy = mask.copy()
            # mask.data *= smoothedEdges * taper.data 1
            mask.data = smoothedEdges * taper.data # 2 
            # mask.data *=  taper.data 


            if True:
                powers[mapnum] = \
                    aveTools.allpowers(*(tquMaps), window = mask, binFile = p['binFile'])




            pickle.dump(powers, open(p['workDir'] + p['basename'] + 'PowersSandbox.pkl', "wb"))

ells = arange(10000)

fells = zeros(len(ells))

fells[where( ells > 2500) ] = 1.
fells[where(ells > 3000)] = 0.
filteredMaps = [None] * 3
for pol , tqu in enumerate(tqus):
    
    filteredMaps[pol] = tquMaps[pol].filterFromList([ells, fells])


figure('temp')
clf()


for pol , tqu in enumerate(tqus):
    subplot(1, 3, pol + 1)
    imshow(filteredMaps[pol].data, clim = [-.01,.01])

    title(tqu)
    colorbar()
    
show()



figure('mapcutout', figsize = [20,7])
clf()


for pol , tqu in enumerate(tqus):
    subplot(1, 3, pol + 1)
    imshow(tquMaps[pol].data, cmap = 'RdBu')

    title(tqu)
    colorbar()
    
show()
savefig('../plot/mapcutout.pdf')

# nx = 10
# ny = 10
# rangeT = [0,10]
# rangeP = [0, 1]
# ranges = [rangeT, rangeP, rangeP]
