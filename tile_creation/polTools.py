from flipper import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2
from scipy import interpolate
from scipy import *
from mapTools import *
import scipy
import os
import random
import sys
import pickle
from scipy.interpolate import splrep,splev

import pdb
def makeEllandAngCoordinate(liteMap,bufferFactor=1):
    
    Ny = liteMap.Ny*bufferFactor
    Nx = liteMap.Nx*bufferFactor
    ly = numpy.fft.fftfreq(Ny,d = liteMap.pixScaleY)*(2*numpy.pi)
    lx = numpy.fft.fftfreq(Nx,d = liteMap.pixScaleX)*(2*numpy.pi)
    modLMap = numpy.zeros([Ny,Nx])
    angLMap = numpy.zeros([Ny,Nx])
    iy, ix = numpy.mgrid[0:Ny,0:Nx]
    modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
    #Trigonometric orientation
    angLMap[iy,ix]= numpy.arctan2(ly[iy],lx[ix])###original issue Thibaut version had -sign here
    
    return(modLMap,angLMap)

def smoothNoise(p2dNoise, pixelSmoothingScale=12, factor=5):
    binningFile = os.environ['LENSRECONSMC_DIR']+os.path.sep+'params/BIN_100_LOG'
    # ---- throw out outliers (do this in bins because median of the whole map
    # does not make much sense )
    binLo, binHi, BinCe = fftTools.readBinningFile(binningFile)
    modIntLMap = numpy.array(p2dNoise.modLMap + 0.5,dtype='int64')
    for ibin in xrange(len(binLo)):
        loc = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
        noiseInRing  =  p2dNoise.powerMap[loc]
        med = numpy.median(noiseInRing)
        noiseInRing[numpy.where(noiseInRing > factor*med)] = med
    # now smooth
    kernel_width = (pixelSmoothingScale,pixelSmoothingScale)
    
    p2dNoise.powerMap[:] = scipy.ndimage.gaussian_filter(p2dNoise.powerMap, kernel_width) 
    
    print "Done tempering Noise Map ..."
    return p2dNoise


def convertToEB(qMap,uMap,window=True,beam = True, beamFile = None):#beamWidth = 1.4):
	"""
	Analysing Data from T,Q,U fits to fft maps of T, E, B 
	@param	T	Tmap
	@param	Q	Qmap
	@param	U	Umap
	"""

        lmap, thetamap = makeEllandAngCoordinate(qMap)

	if window:
	    kQMap = fftTools.fftFromLiteMap(qMap)
	    kUMap = fftTools.fftFromLiteMap(uMap)
	else:
	    print "applying windows!"
	    kQMap = fftTools.fftFromLiteMap(qMap,applySlepianTaper=True,nresForSlepian=1.0)
	    kUMap = fftTools.fftFromLiteMap(uMap,applySlepianTaper=True,nresForSlepian=1.0)
	  
        kTMap = kQMap.copy()
        kEMap = kQMap.copy()
	kBMap = kUMap.copy()

        kEMap.kMap = numpy.cos(2.*thetamap)*kQMap.kMap+numpy.sin(2.*thetamap)*kUMap.kMap
	kBMap.kMap = -numpy.sin(2.*thetamap)*kQMap.kMap+numpy.cos(2.*thetamap)*kUMap.kMap
	
	eMap = qMap.copy()
	bMap = uMap.copy()

        if beam:   # deconvolve beam
            ell, f_ell = numpy.transpose(numpy.loadtxt(beamFile))[0:2,:]
            filt = 1./f_ell
            filt = makeTemplate(ell,filt,kEMap)
            kEMap.kMap *= filt
            kBMap.kMap *= filt

	eMap.data = numpy.real(numpy.fft.ifft2(kEMap.kMap))
        bMap.data = numpy.real(numpy.fft.ifft2(kBMap.kMap))
	
	return eMap, bMap


def initializeDerivativesWindowfuntions(liteMap):
	
    def matrixShift(l,row_shift,column_shift):	
        m1=numpy.hstack((l[:,row_shift:],l[:,:row_shift]))
        m2=numpy.vstack((m1[column_shift:],m1[:column_shift]))
        return m2
    delta=liteMap.pixScaleX
    Win=liteMap.data[:]
    
    dWin_dx=(-matrixShift(Win,-2,0)+8*matrixShift(Win,-1,0)-8*matrixShift(Win,1,0)+matrixShift(Win,2,0))/(12*delta)
    dWin_dy=(-matrixShift(Win,0,-2)+8*matrixShift(Win,0,-1)-8*matrixShift(Win,0,1)+matrixShift(Win,0,2))/(12*delta)
    d2Win_dx2=(-matrixShift(dWin_dx,-2,0)+8*matrixShift(dWin_dx,-1,0)-8*matrixShift(dWin_dx,1,0)+matrixShift(dWin_dx,2,0))/(12*delta)
    d2Win_dy2=(-matrixShift(dWin_dy,0,-2)+8*matrixShift(dWin_dy,0,-1)-8*matrixShift(dWin_dy,0,1)+matrixShift(dWin_dy,0,2))/(12*delta)
    d2Win_dxdy=(-matrixShift(dWin_dy,-2,0)+8*matrixShift(dWin_dy,-1,0)-8*matrixShift(dWin_dy,1,0)+matrixShift(dWin_dy,2,0))/(12*delta)
    
    #In return we change the sign of the simple gradient in order to agree with numpy convention
    return {'Win':Win, 'dWin_dx':-dWin_dx,'dWin_dy':-dWin_dy, 'd2Win_dx2':d2Win_dx2, 'd2Win_dy2':d2Win_dy2,'d2Win_dxdy':d2Win_dxdy}


def applyWinBeamDecon(T_map,window,beam=False,beamFile = None):
    window=initializeDerivativesWindowfuntions(window)
    win =window['Win']
    T_temp=T_map.copy()
    T_temp.data=T_map.data*win
    fT=fftTools.fftFromLiteMap(T_temp)
    if beam:   # deconvolve beam
        ell, f_ell = numpy.transpose(numpy.loadtxt(beamFile))[0:2,:]         
        filt = 1./f_ell
        filt = makeTemplate(ell,filt,fT)
        fT.kMap *= filt
    tMap = T_map.copy() 
    tMap.data = numpy.real(numpy.fft.ifft2(fT.kMap))
    return tMap       


def QUtoPureEB(Q_map,U_map,window,beam=False,beamFile = None):

    modLMap, angLMap = makeEllandAngCoordinate(Q_map)

    window=initializeDerivativesWindowfuntions(window)
    
    win =window['Win']
    dWin_dx=window['dWin_dx']
    dWin_dy=window['dWin_dy']
    d2Win_dx2=window['d2Win_dx2'] 
    d2Win_dy2=window['d2Win_dy2']
    d2Win_dxdy=window['d2Win_dxdy']	

    T_temp=Q_map.copy()
    T_map =Q_map.copy()
    Q_temp=Q_map.copy()
    U_temp=U_map.copy()

    T_temp.data=T_map.data*win
    fT=fftTools.fftFromLiteMap(T_temp)
    
    Q_temp.data=Q_map.data*win
    fQ=fftTools.fftFromLiteMap(Q_temp)
    
    U_temp.data=U_map.data*win
    fU=fftTools.fftFromLiteMap(U_temp)
    
    fE=fT.copy()
    fB=fT.copy()
    
    fE.kMap[:]=fQ.kMap[:]*numpy.cos(2.*angLMap)+fU.kMap[:]*numpy.sin(2.*angLMap)
    fB.kMap[:]=-fQ.kMap[:]*numpy.sin(2.*angLMap)+fU.kMap[:]*numpy.cos(2.*angLMap)
        
    Q_temp.data=Q_map.data*dWin_dx
    QWx=fftTools.fftFromLiteMap(Q_temp)
    
    Q_temp.data=Q_map.data*dWin_dy
    QWy=fftTools.fftFromLiteMap(Q_temp)
    
    U_temp.data=U_map.data*dWin_dx
    UWx=fftTools.fftFromLiteMap(U_temp)
    
    U_temp.data=U_map.data*dWin_dy
    UWy=fftTools.fftFromLiteMap(U_temp)
    
    U_temp.data=2.*Q_map.data*d2Win_dxdy-U_map.data*(d2Win_dx2-d2Win_dy2)
    QU_B=fftTools.fftFromLiteMap(U_temp)
 
    U_temp.data=-Q_map.data*(d2Win_dx2-d2Win_dy2)-2.*U_map.data*d2Win_dxdy
    QU_E=fftTools.fftFromLiteMap(U_temp)
    
    modLMap=modLMap+2

    fB.kMap[:] += QU_B.kMap[:]*(1./modLMap)**2
    fB.kMap[:]-= (2.*1j)/modLMap*(numpy.sin(angLMap)*(QWx.kMap[:]+UWy.kMap[:])+numpy.cos(angLMap)*(QWy.kMap[:]-UWx.kMap[:]))
        
    fE.kMap[:]+= QU_E.kMap[:]*(1./modLMap)**2
    fE.kMap[:]-= (2.*1j)/modLMap*(numpy.sin(angLMap)*(QWy.kMap[:]-UWx.kMap[:])-numpy.cos(angLMap)*(QWx.kMap[:]+UWy.kMap[:]))    

    eMap = Q_map.copy()
    bMap = Q_map.copy()
    fE.kMap[0,0] = 0.
    fB.kMap[0,0] = 0.

    if beam:   # deconvolve beam
        ell, f_ell = numpy.transpose(numpy.loadtxt(beamFile))[0:2,:]
        filt = 1./f_ell
        filt = makeTemplate(ell,filt,fE)
        fE.kMap *= filt
        fB.kMap *= filt

    eMap.data = numpy.real(numpy.fft.ifft2(fE.kMap))
    bMap.data = numpy.real(numpy.fft.ifft2(fB.kMap))
    return eMap, bMap  #fE, fB


def meanCrossSpec(mapList,applySlepianTaper=True,nresForSlepian=3.0):
    count = 0 
    
    for i in xrange(len(mapList)):
        for j in xrange(i):
            
            p2d = fftTools.powerFromLiteMap(mapList[i],mapList[j],\
                                            applySlepianTaper=applySlepianTaper,\
                                            nresForSlepian=nresForSlepian)
            if count == 0:
                p2d0 = p2d.copy()
            else:
                p2d0.powerMap[:] += p2d.powerMap[:]
            count += 1
            
    p2d0.powerMap[:] /= count
    powerM = p2d0.powerMap.copy()
    print 'count=', count
        
    lL,lU,lBin,clBinCrossMean,err,w = p2d0.binInAnnuli(os.environ['LENSRECONSMC_DIR']+os.path.sep+'params/BIN_100_LOG')
            
    return lBin,clBinCrossMean,powerM

def meanAutoSpec(mapList,applySlepianTaper=True,nresForSlepian=3.0):
    count = 0 
    
    for i in xrange(len(mapList)):
            
        p2d = fftTools.powerFromLiteMap(mapList[i],\
                                         applySlepianTaper=applySlepianTaper,\
                                            nresForSlepian=nresForSlepian)
        if count == 0:
            p2d0 = p2d.copy()
        else:
            p2d0.powerMap[:] += p2d.powerMap[:]
        count += 1
            
    p2d0.powerMap[:] /= count
    powerM2 = p2d0.powerMap.copy()
        
        
    lL,lU,lBin,clBinAutoMean,err,w = p2d0.binInAnnuli(os.environ['LENSRECONSMC_DIR']+os.path.sep+'params/BIN_100_LOG')
            
    return lBin,clBinAutoMean,powerM2




def fourierMask(templatePower,\
                lxcut = None, lycut = None, lmin = None, lmax = None):
    
    outTemplate = (templatePower.copy())


    output = zeros(outTemplate.powerMap.shape, dtype = int)
    print output.shape
    output[:] = 1

    if lmin != None:
        wh = where(outTemplate.modLMap <= lmin)
        output[wh] = 0


    if lmax != None:
        wh = where(outTemplate.modLMap >= lmax)
        output[wh] = 0
        
    if lxcut != None:

        wh = where(absolute(outTemplate.lx) < lxcut)
        output[:,wh] = 0
        
    if lycut != None:

        wh = where(absolute(outTemplate.ly) < lycut)
        
        output[wh,:] = 0


    return output

                       
    

    
