import healpy as hp
import numpy as np
type='TrueMaps'
from flipper import *
import liteMapNickHand,healpy

if True:
	healpixMapT,healpixMapQ,healpixMapU = healpy.fitsfunc.read_map('CleanWidePatch/DedustingMaps/%s.fits' %type,field=(0,1,2),h=False)

	Ra0,dec0,Ra1,dec1=300.,-70.,340.,-45.#270.,-70.,350.,-46.
	m = liteMap.makeEmptyCEATemplateAdvanced(Ra0,dec0,Ra1,dec1)
	tempfilename = 'CleanWidePatch/DedustingMaps/templatemap.fits'
	m.writeFits(tempfilename, overWrite = True)
	mapsT = liteMapNickHand.getEmptyMapAtLocation(tempfilename,np.mean([Ra0,Ra1]),np.mean([dec0,dec1]))
	mapsT.loadDataFromHealpixMap(healpixMapT)
	print('T done')
	mapsQ = mapsT.copy()
	mapsQ.loadDataFromHealpixMap(healpixMapQ)
	print('Q done')
	mapsU = mapsT.copy()
	mapsU.loadDataFromHealpixMap(healpixMapU)
	print('U done')

	mapsT.writeFits('CleanWidePatch/DedustingMaps/rT_%sCut.fits' %type,overWrite=True)
	mapsU.writeFits('CleanWidePatch/DedustingMaps/rU_%sCut.fits' %type,overWrite=True)
	mapsQ.writeFits('CleanWidePatch/DedustingMaps/rQ_%sCut.fits' %type,overWrite=True)
	print('Saved')
	import pickle
	window=mapsT.copy()
	window.data=np.ones_like(window.data)
	window=window.createGaussianApodization(pad=0,kern=100)
	window.writeFits('CleanWidePatch/DedustingMaps/Window.fits',overWrite=True)
	print('Window done')
#if True:
#	window=liteMap.
#import flipperPol as fp
#modL,angL=fp.fftPol.makeEllandAngCoordinate(mapsT)
#fT,fE,fB=fp.fftPol.TQUtoPureTEB(mapsT,mapsQ,mapsU,window,modL,angL,method='hybrid')
#print('Fourier Done')
#pickle.dump(open('CleanWidePatch/DedustingMaps/fB%s.pkl' %type,'wb'),fB)
