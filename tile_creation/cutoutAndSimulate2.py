if __name__=='__main__':
	reMake=True # whether to remake previously created data

	from flipper import *
	from flipperPol import *

	import healpy
	import sys
	sys.path.append('/data/ohep2/EngelenTools/') 
	
	sys.path.append('/data/ohep2/hades/')
	from hades.params import BICEP
	a=BICEP()
	from hades.padded_debiased_wrap import padded_wrap
	from hades.batchI2 import I_strength

	import aveTools
	import numpy as np
	p = flipperDict.flipperDict()
	p.read_from_file(sys.argv[1])

	Ra0Array = p['Ra0Array']
	Ra1Array =  p['Ra1Array']
	Dec0Array =  p['Dec0Array']
	Dec1Array =  p['Dec1Array']
	buffer = p['buffer']

	if p['mapWidthsDeg']!=a.sep:
		raise Exception('Inconsistent separation/map width size')
	import os
	if not os.path.exists(p['mapDir']):
		os.makedirs(p['mapDir'])

	#import sys
	#sys.path.append('/data/ohep2/EngelenTools/') 
	import liteMapNickHand

	doAll = True
	if doAll:
	    print 'loading data'
	    healpixMapT,healpixMapQ,healpixMapU, mapHeader = healpy.fitsfunc.read_map(p['workDir']+p['mapfilename'],field=(0,1,2),h=True)#']'](p['workDir'] + p['basename'] + '_map_gal.fits', field = (0,1,2),\
	                                                     #    h = True)

	    if p['cutoutMask']:
	        mask, maskHeaderMask = healpy.fitsfunc.read_map(p['workDir'] + p['maskName'], h = True,field=p['maskField'])
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
	    roundMask = np.ones(len(healpixMapT))

	mapsT = aveTools.onedl(nMaps)
	mapsQ = aveTools.onedl(nMaps)
	mapsU = aveTools.onedl(nMaps)

	centerPixes = healpy.ang2pix(p['mapNside'], np.pi * (90 - mapDecs) / 180., np.pi * mapRas / 180.)
	
	
	# Run prerequisite powerMaps functions
	import pickle
	
	import scipy.ndimage.filters
	#from mpi4py import MPI
	#comm=MPI.COMM_WORLD
	#rank=comm.Get_rank()
	#size=comm.Get_size()
	
	tqus=['T','Q','U']
	cosineApod=p['cosineApod']
	
	def myTaper(indata):

    		smoothed = scipy.ndimage.filters.gaussian_filter(indata, sigma = 60)

    		bitmask = numpy.zeros(indata.shape)
    		bitmask[numpy.where(smoothed > .99)] = 1.

    		output = scipy.ndimage.filters.gaussian_filter(bitmask, sigma = 60)
    
    		return output

	tquMaps=[None]*3
	print 'Creating cosine window'
	del mapHeader,mapsT,mapsQ,mapsU
	if p['cutoutMask']:
		del maskHeaderMask
		
	# Import the power map function
	sys.path.append('data/ohep2/py/')
	
	def createMask(mapnum):
		# mask creation code yanked from powerMaps.py
		tquMaps=[None]*3
		for pol, tqu in enumerate(tqus):
			filename=p['workDir']+p['basename'] + 'map%s_%05i.fits'%(tqu, mapnum)
			tquMaps[pol] = liteMap.liteMapFromFits(filename)        
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

	        powersCoeff = [0.]#aveTools.allpowers(*(tquMaps), window = mask, binFile = p['binFile'])
		del tquMaps,smoothedEdges
		if p['applyPerPatchMask']:
			del mask

		return powersCoeff

	def mapCreate(i,firstTime=False): 
		import time
		init_time=time.time()
		if roundMask[centerPixes[i]]==1:
			good_map = True
			#return [good_map,0,0]
			outDir=a.root_dir+'DebiasedBatchDataFull/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
			fileLoc=outDir+'%s.npy' %i
			
			if reMake:
				if os.path.exists(fileLoc) and not firstTime:
					print '%s already done' %i
					return [good_map,None,None]
			print 'good data at ra = %3.1f, dec=%3.1f pixel %i' %(mapRas[i],mapDecs[i],i)
			
			mapsT = liteMapNickHand.getEmptyMapAtLocation(tempfilename,mapRas[i],mapDecs[i])
			mapsT.loadDataFromHealpixMap(healpixMapT)
			
			mapsQ = mapsT.copy()
			mapsQ.loadDataFromHealpixMap(healpixMapQ)
			
			mapsU = mapsT.copy()
			mapsU.loadDataFromHealpixMap(healpixMapU)
	        	
	        	#del healpixMapT,healpixMapU,healpixMapQ
	        
	        	if p['cutoutMask']:
	        		maskCutout = mapsT.copy()
	            		maskCutout.loadDataFromHealpixMap(mask)

	        	mapsT.writeFits(p['workDir'] + p['basename'] + 'mapT_%05i.fits'%i, overWrite = True)
	        	mapsQ.writeFits(p['workDir'] + p['basename'] + 'mapQ_%05i.fits'%i, overWrite = True)
	        	mapsU.writeFits(p['workDir'] + p['basename'] + 'mapU_%05i.fits'%i, overWrite = True)
	        	if p['cutoutMask']:
	            		maskCutout.writeFits(p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%i, overWrite = True)
	            		del maskCutout
	            		
	            	if not firstTime:
	            	
	            		# Now compute the mask maps
	            		createMask(i)
	            	
	            		## Now compute the estimators on this map patch
	            		outDir=a.root_dir+'DebiasedBatchDataFull/f%s_ms%s_s%s_fw%s_np%s_d%s/' %(a.freq,a.map_size,a.sep,a.FWHM,a.noise_power,a.delensing_fraction)
	
				# Now compute mean I^2 and Q^2+U^2:
				I2,QU=I_strength(i,root_dir=a.root_dir,sep=a.sep,map_size=a.map_size)
				
				output=padded_wrap(i,map_size=a.map_size,sep=a.sep,N_sims=a.N_sims,N_bias=a.N_bias,noise_power=a.noise_power,FWHM=a.FWHM,\
					slope=a.slope,l_step=a.l_step,lMin=a.lMin,lMax=a.lMax,rot=a.rot,freq=a.freq,root_dir=a.root_dir,\
					delensing_fraction=a.delensing_fraction,useTensors=a.useTensors,f_dust=a.f_dust,\
					rot_average=a.rot_average,useBias=a.useBias,padding_ratio=a.padding_ratio,unPadded=a.unPadded,flipU=a.flipU)
	
	            		if not os.path.exists(outDir): # make directory
					os.makedirs(outDir)
		
				np.save(outDir+'%s.npy' %i, output) # save output
				print 'anisotropy estimators complete for map %s' %i
				del output
				# Delete old maps		
				
				del mapsQ,mapsU,mapsT
				if os.path.exists(p['workDir'] + p['basename'] + 'mapT_%05i.fits'%i):
					os.remove(p['workDir'] + p['basename'] + 'mapT_%05i.fits'%i)
					os.remove(p['workDir'] + p['basename'] + 'mapQ_%05i.fits'%i)
					os.remove(p['workDir'] + p['basename'] + 'mapU_%05i.fits'%i)
					os.remove(p['workDir'] + p['basename'] + 'mapMaskSmoothed_%05i.fits'%i)
					try:
						os.remove(p['workDir'] + p['basename'] + 'mapMask_%05i.fits'%i)
					except OSError:
						pass
				else:
					print 'No file to remove'
					
				print 'Map %s complete after %s seconds' %(i,time.time()-init_time)
		else:
	        	good_map = False
	        	I2=None
	        	QU=None
	        	print 'bad_map at index %s' %i
	        	
	        if firstTime:
	        	return good_map
	        else:
	        	return [good_map,I2,QU]

	print 'starting multiprocessing'
	
	# Compute first run separately
	mapCreate(0,firstTime=True)
	
	for pol,tqu in enumerate(tqus):
		filename=p['workDir']+p['basename'] + 'map%s_%05i.fits'%(tqu, 0)
		tquMaps[pol] = liteMap.liteMapFromFits(filename)
		taper = liteMapPol.initializeCosineWindow(tquMaps[0],\
					cosineApod['lenApod'],\
                                        cosineApod['pad'])  # taper weight map

	# Now run the multiprocessing
	print 'start multiprocessing'
	import multiprocessing as mp
	pq=mp.Pool()#processes=40)
	import tqdm

	iterVals=np.arange(nMaps)

	
	output=list(tqdm.tqdm(pq.imap(mapCreate,iterVals),total=len(iterVals)))
	print 'multiprocessing complete'
	#pq.close()
	#pq.join()
	

	Idat,QUdat=[],[]

	for i in range(len(output)):
		goodMap[i]=output[i][0]
		if goodMap[i]==True:
			Idat.append(output[i][1])
			QUdat.append(output[i][2])
		

	import pickle
	pickle.dump(goodMap, open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'w'))
	pickle.dump(mapRas, open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'w'))
	pickle.dump(mapDecs, open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'w'))


	# Save QU,<I2> data
	np.save(a.root_dir+'%sdeg%s/meanI2.npy' %(a.map_size,a.sep),np.array(Idat))
   	np.save(a.root_dir+'%sdeg%s/meanQU.npy' %(a.map_size,a.sep),np.array(QUdat))
   
