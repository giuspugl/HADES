



from flipper import *

import healpy
import aveTools
import pickle
p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])


import matplotlib.pyplot as plt

tqus = ['T', 'Q', 'U']

nx = 10
ny = 10
rangeT = [0,10]
rangeP = [0, 1]

goodMap = pickle.load( open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'r'))
mapRas = pickle.load( open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'r'))
mapDecs = pickle.load( open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'r'))

nMaps = len(goodMap)
# nGoodMaps = np.sum(goodMap)
ranges = [rangeT, rangeP, rangeP]
for pol , tqu in enumerate(tqus):

    plt.figure(pol, figsize = (20,20))
    runningCount = 0
    for  mapnum in range(nMaps):

        if goodMap[mapnum]:
        
            filename = p['workDir'] + p['basename'] + 'map%s_%05i.fits'%(tqu, mapnum)
            map =liteMap.liteMapFromFits(filename)

            runningCount += 1
            plt.subplot(nx, ny, runningCount)
            plt.axis('off')
            plt.imshow(map.data, clim = ranges[pol])
            plt.title('(%3.1f,%3.1f),  %i' %(mapRas[mapnum], mapDecs[mapnum], mapnum), fontsize = 9)

    plt.savefig('../plot/quickPlot_%s.pdf'  % tqu)
# show()


