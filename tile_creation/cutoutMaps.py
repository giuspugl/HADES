

from flipper import *

import healpy
import aveTools
import numpy as np
p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])

Ra0Array = p['Ra0Array']
Ra1Array =  p['Ra1Array']
Dec0Array =  p['Dec0Array']
Dec1Array =  p['Dec1Array']
buffer = p['buffer']

import sys
sys.path.append('/scratch2/r/rbond/engelen/') 
import liteMapNickHand



doAll = True
if doAll:
    print 'loading data'
    healpixMapT,healpixMapQ,healpixMapU, mapHeader = healpy.fitsfunc.read_map(p['workDir'] + p['basename'] + '_map_gal.fits', field = (0,1,2),\
                                                         h = True)

    if p['cutoutMask']:
        mask, maskHeaderMask = healpy.fitsfunc.read_map(p['workDir'] + 'mask_map_gal.fits', h = True)
    print 'done'



decVals = np.linspace(-90 + p['mapWidthsDeg'] / 2,  90 - p['mapWidthsDeg'] / 2, num = 180 / (p['mapWidthsDeg'] - 1))

nDecs = len(decVals)

raVals = aveTools.onedl(nDecs)

nPoints = np.zeros(nDecs)

for i, decVal in enumerate(decVals):

    raVals[i] = np.arange(0, 360, p['mapWidthsDeg'] / np.cos(decVal * np.pi / 180))
    nPoints[i] = len(raVals[i])





# m = liteMap.makeEmptyCEATemplateAdvanced(Ra0Array[i] - buffer, Dec0Array[i] - buffer, \
#                                              Ra1Array[i] + buffer, Dec1Array[i] + buffer)

i = 0

m = liteMap.makeEmptyCEATemplateAdvanced(Ra0Array[i] - buffer, Dec0Array[i] - buffer, \
                                             Ra1Array[i] + buffer, Dec1Array[i] + buffer)
tempfilename = p['workDir'] + 'templatemap.fits'
m.writeFits(tempfilename, overWrite = True)

# m.loadDataFromHealpixMap(mapT)


# zz.loadDataFromHealpixMap(mapT)


mapRas = np.array([item for sublist in raVals for item in sublist])

mapDecs = [[decVals[i]] * n for i,n in enumerate(nPoints)]
mapDecs = np.array([item for sublist in mapDecs for item in sublist])

nMaps = len(mapDecs)
if nMaps != len(mapRas):
    raise valueError

goodMap = aveTools.onedl(nMaps)
if p['cutoutMask']:

    roundMask = np.round(mask)
else: #just set to ones.
    roundMask = np.ones(len(healpixMapT))

mapsT = aveTools.onedl(nMaps)
mapsQ = aveTools.onedl(nMaps)
mapsU = aveTools.onedl(nMaps)

centerPixes = healpy.ang2pix(p['mapNside'], np.pi * (90 - mapDecs) / 180., np.pi * mapRas / 180.)

for i in np.arange(nMaps):
    
    if roundMask[centerPixes[i]] == 1 : 
        goodMap[i] = True

        print 'good data at ra = %3.1f, dec = %3.1f, pixel %i' %(mapRas[i], mapDecs[i], i)

        mapsT = liteMapNickHand.getEmptyMapAtLocation(tempfilename, mapRas[i], mapDecs[i])
        mapsT.loadDataFromHealpixMap(healpixMapT)

        
        mapsQ = mapsT.copy()
        mapsQ.loadDataFromHealpixMap(healpixMapQ)

        mapsU = mapsT.copy()
        mapsU.loadDataFromHealpixMap(healpixMapU)

        if p['cutoutMask']:
            maskCutout = mapsT.copy()
            maskCutout.loadDataFromHealpixMap(mask)

        mapsT.writeFits(p['workDir'] + p['basename'] + 'mapT_%05i.fits'%i, overWrite = True)
        mapsQ.writeFits(p['workDir'] + p['basename'] + 'mapQ_%05i.fits'%i, overWrite = True)
        mapsU.writeFits(p['workDir'] + p['basename'] + 'mapU_%05i.fits'%i, overWrite = True)
        if p['cutoutMask']:
            maskCutout.writeFits(p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%i, overWrite = True)


    else:
        goodMap[i] = False

import pickle
pickle.dump(goodMap, open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'w'))
pickle.dump(mapRas, open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'w'))
pickle.dump(mapDecs, open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'w'))

