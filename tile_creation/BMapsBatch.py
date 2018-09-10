if __name__=='__main__':
	from flipper import *

	import healpy
	import sys
	sys.path.append('/data/ohep2/EngelenTools/') 

	import aveTools
	import numpy as np
	p = flipperDict.flipperDict()
	p.read_from_file(sys.argv[1])

	Ra0Array = p['Ra0Array']
	Ra1Array =  p['Ra1Array']
	Dec0Array =  p['Dec0Array']
	Dec1Array =  p['Dec1Array']
	buffer = p['buffer']

	import os
	if not os.path.exists(p['mapDir']):
		os.makedirs(p['mapDir'])


	#import sys
	#sys.path.append('/data/ohep2/EngelenTools/') 
	import liteMapNickHand



	doAll = True
	if doAll:
	    print 'loading data'
	    healpixMapB, mapHeader = healpy.fitsfunc.read_map(p['workDir']+p['mapfilename'],h=True)#']'](p['workDir'] + p['basename'] + '_map_gal.fits', field = (0,1,2),\
	                                                     #    h = True)

	    if p['cutoutMask']:
	        mask, maskHeaderMask = healpy.fitsfunc.read_map(p['workDir'] + p['maskName'], h = True)
	    print 'done'



	decVals = np.linspace(-90 + p['mapWidthsDeg'] / 2.,  90 - p['mapWidthsDeg'] / 2., num = (180/(p['mapWidthsDeg'])) ) # num = 180 / (p['mapWidthsDeg'] - 1)) - Philcox edit

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

	mapDecs = [[decVals[int(i)]] * int(n) for i,n in enumerate(nPoints)]
	mapDecs = np.array([item for sublist in mapDecs for item in sublist])

	nMaps = len(mapDecs)
	if nMaps != len(mapRas):
	    raise valueError
	print(nMaps)
	goodMap = aveTools.onedl(nMaps)
	if p['cutoutMask']:
	    roundMask = np.round(mask)
	else: #just set to ones
	    roundMask = np.ones(len(healpixMapB))

	mapsT = aveTools.onedl(nMaps)
	mapsQ = aveTools.onedl(nMaps)
	mapsU = aveTools.onedl(nMaps)

	centerPixes = healpy.ang2pix(p['mapNside'], np.pi * (90 - mapDecs) / 180., np.pi * mapRas / 180.)

	def mapCreate(i): 
		#print i
		if roundMask[centerPixes[i]]==1:
			good_map = True
			print 'good data at ra = %3.1f, dec=%3.1f pixel %i' %(mapRas[i],mapDecs[i],i)
			
			mapsB = liteMapNickHand.getEmptyMapAtLocation(tempfilename,mapRas[i],mapDecs[i])
			mapsB.loadDataFromHealpixMap(healpixMapB)
			
	        
	        	if p['cutoutMask']:
	        		maskCutout = mapsB.copy()
	            		maskCutout.loadDataFromHealpixMap(mask)

	        	mapsB.writeFits(p['workDir'] + p['basename'] + 'mapB_%05i.fits'%i, overWrite = True)
	        	if p['cutoutMask']:
	            		maskCutout.writeFits(p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%i, overWrite = True)

		else:
	        	good_map = False
	        #print 'done'
	    	return good_map

	print 'starting multiprocessing'
	
	import multiprocessing as mp
	pq=mp.Pool()
	import tqdm

	iterVals=np.arange(nMaps)

	
	#print mapCreate(15)
	output=list(tqdm.tqdm(pq.imap(mapCreate,iterVals),total=len(iterVals)))
	print 'multiprocessing complete'


	for i in range(len(output)):
		goodMap[i]=output[i]


	import pickle
	pickle.dump(goodMap, open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'w'))
	pickle.dump(mapRas, open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'w'))
	pickle.dump(mapDecs, open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'w'))




