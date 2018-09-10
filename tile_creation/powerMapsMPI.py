import sys
sys.path.append('/data/ohep2/EngelenTools/') 

from flipper import *
from flipperPol import *
import healpy
import aveTools
import pickle
import scipy.ndimage.filters

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])



import matplotlib.pyplot as plt

tqus = ['T', 'Q', 'U']

goodMap = pickle.load( open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'r'))
mapRas = pickle.load( open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'r'))
mapDecs = pickle.load( open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'r'))

nMaps = len(goodMap)

nMaps = len(mapRas)#1400
print "hack, setting nMaps to %i" % nMaps

# nGoodMaps = np.sum(goodMap)

cosineApod = p['cosineApod']


def myTaper(indata):

    smoothed = scipy.ndimage.filters.gaussian_filter(indata, sigma = 60)

    bitmask = numpy.zeros(indata.shape)
    bitmask[numpy.where(smoothed > .99)] = 1.

    output = scipy.ndimage.filters.gaussian_filter(bitmask, sigma = 60)
    
    return output




firstTime = True

doAll = True

iStop = nMaps


iStart = 0
delta = (iStop - iStart)/size
if delta == 0:
	raise ValueError, 'Too many processors for too small a  loop!'

iMin = iStart+rank*delta
iMax = iStart+(rank+1)*delta

if iMax>iStop:
    iMax = iStop
elif (iMax > (iStop - delta)) and iMax <iStop:
    iMax = iStop

if doAll:
    powers = aveTools.onedl(nMaps)

    for mapnum in xrange(iMin, iMax):
    # for  mapnum in range(nMaps):
    # for mapnum in ([36]):
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


            if p['flipU']:
                tquMaps[2].data *= -1

            if p['applyPerPatchMask']:
                maskFilename = p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%( mapnum)
                mask = liteMap.liteMapFromFits(maskFilename)
            else:
                mask = tquMaps[0].copy()
                mask.data[:] = 1.
            
            


            smoothedEdges = myTaper(mask.data) 

            # maskCopy = mask.copy()
            # mask.data *= smoothedEdges * taper.data 1
            #mask.data = smoothedEdges * taper.data # 2 
            mask.data *=  taper.data 


            mask.writeFits(p['workDir'] + p['basename'] + 'mapMaskSmoothed_%05i.fits'%mapnum, overWrite = True)

            if True:
                powers[mapnum] = \
                    aveTools.allpowers(*(tquMaps), window = mask, binFile = p['binFile'])


if rank > 0:
    comm.send(powers[iMin:iMax], dest = 0)
    print 'rank %i of %i: sending data, length', len(powers[iMin:iMax])
else:
    for i in range(1, size):
        inData = comm.recv(source = i)  #do this in two steps, because the length of the data received here is unpredictable for i == size-1
        powers[iStart + i * delta : iStart + i * delta + len(inData)] = inData
        print 'rank %i of %i: received data ' % (rank, size) 
                
        # pickle.dump(powers, open(p['workDir'] + p['basename'] + 'PowersSandbox.pkl', "wb"))
    pickle.dump(powers, open(p['workDir'] + p['basename'] + 'Powers.pkl', "wb"))

            


# nx = 10
# ny = 10
# rangeT = [0,10]
# rangeP = [0, 1]
# ranges = [rangeT, rangeP, rangeP]
