

from flipper import *

import healpy
import aveTools

p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])

Ra0Array = p['Ra0Array']
Ra1Array =  p['Ra1Array']
Dec0Array =  p['Dec0Array']
Dec1Array =  p['Dec1Array']
buffer = p['buffer']

rotateMask = False

mapT,mapQ,mapU, mapHeader = healpy.fitsfunc.read_map(p['mapDir'] + p['mapName'], field = (0,1,2),\
                                                     h = True)
mapCelestial = [mapT, mapQ, mapU]

if rotateMask:
    maskCelestial, maskHeaderMask = healpy.fitsfunc.read_map(p['mapDir'] + p['maskName'], h = True)


i = 0



nTQUs = 3

mapGalactic = aveTools.onedl(nTQUs)

for i in range(nTQUs):


    mapGalactic[i] = aveTools.rotateHealpixGtoC(mapCelestial[i], actuallyDoCtoG = True)
    if rotateMask:
        maskGalactic = aveTools.rotateHealpixGtoC(maskCelestial, actuallyDoCtoG = True)


healpy.fitsfunc.write_map(p['workDir'] + p['basename'] + '_map_gal.fits', mapGalactic, coord = 'G') 


# m = liteMap.makeEmptyCEATemplateAdvanced(Ra0Array[i] - buffer, Dec0Array[i] - buffer, \
#                                              Ra1Array[i] + buffer, Dec1Array[i] + buffer)


# m.loadDataFromHealpixMap(mapCelestial)




