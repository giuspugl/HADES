from flipper import *
from flipperPol import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2
import fft as fftFast
from scipy import interpolate
from scipy import *
import matplotlib.pyplot as plt
import scipy
import os
import random
import pickle
import kappaTools
import kappaToolsReal
import kappaToolsBias
import kappaToolsSimpleN0
import polTools
import mapTools
import time

def phaseRandomizeArrays(mapArray1,mapArray2,mapArray3,mapArray4):
    map = mapArray1[0]
    randomMap = map.copy()
    randomMap.data = numpy.random.randn(map.Ny,map.Nx)
    fftR = fftTools.fftFromLiteMap(randomMap)
    randomPhase = numpy.exp(numpy.angle(fftR.kMap)*2.*numpy.pi*1.j)
    for i in xrange(len(mapArray1)):
        fft = fftTools.fftFromLiteMap(mapArray1[i])
        fft.kMap *= randomPhase
        mapArray1[i].data = fft.mapFromFFT()
    for i in xrange(len(mapArray2)):
        fft = fftTools.fftFromLiteMap(mapArray2[i])
        fft.kMap *= randomPhase
        mapArray2[i].data = fft.mapFromFFT()
    for i in xrange(len(mapArray3)):
        fft = fftTools.fftFromLiteMap(mapArray3[i])
        fft.kMap *= randomPhase
        mapArray3[i].data = fft.mapFromFFT()
    for i in xrange(len(mapArray4)):
        fft = fftTools.fftFromLiteMap(mapArray4[i])
        fft.kMap *= randomPhase
        mapArray4[i].data = fft.mapFromFFT()
    return mapArray1,mapArray2,mapArray3,mapArray4


def getKappaFunction(mapArray1,mapArray2,clTheoryArray,noiseArray,polComb,A_L=None,trimAtL=None,randomizePhase=False,trimAtLFinal = None,modeCoupling='kappa',realSpace=True, simpleN0=False):
    templatePower = fftTools.powerFromLiteMap(mapArray1[0])
    templatePower.powerMap *= 0.
    
    l = clTheoryArray[0]
    clTT = clTheoryArray[1]
    clEE = clTheoryArray[2]
    clBB = clTheoryArray[3]
    clTE = clTheoryArray[4]

    # Initialize a 2d power map for the normalization
    crossForNormTT = mapTools.makeTemplate(l,clTT,templatePower)
    crossForNormTE = mapTools.makeTemplate(l,clTE,templatePower)
    crossForNormEE = mapTools.makeTemplate(l,clEE,templatePower)             

    # Read in noise power maps
    noiseForFilter1 = noiseArray[0]
    noiseForFilter2 = noiseArray[1]

    # definition [T,E,B] [TT, EE, BB, TE,l]
    if polComb == 'TT':
        print 'performing TT reconstruction'    
        cl11 = clTT
        cl22 = clTT
        map0 = mapArray1[0]
        map1 = mapArray2[0]
    elif polComb == 'TE': 
        print 'performing TE reconstruction'
        cl11 = clTT 
        cl22 = clEE
        map0 = mapArray1[0]
        map1 = mapArray2[1]
    elif polComb == 'EE': 
        print 'performing EE reconstruction'
        cl11 = clEE
        cl22 = clEE
        map0 = mapArray1[1]
        map1 = mapArray2[1]
    elif polComb == 'EB':
        print 'performing EB reconstruction'
        cl11 = clBB
        cl22 = clEE
        map0 = mapArray1[2]
        map1 = mapArray2[1]

    print polComb, ' ', modeCoupling
    sm0 = kappaToolsReal.superMap(map0,noiseForFilter1,l,cl11,trimAtL=trimAtL,randomizePhase=randomizePhase,trimAtLFinal=trimAtLFinal)
    sm1 = kappaToolsReal.superMap(map1,noiseForFilter2,l,cl22,trimAtL=trimAtL,randomizePhase=randomizePhase,trimAtLFinal=trimAtLFinal)

    print 'beginning recons'
    if realSpace == True:
      ftkappaMap, A_L_new = kappaToolsReal.getKappaReal(sm0,sm1,A_L,crossForNormTE,crossForNormTT,crossForNormEE,pol=polComb,modeCoupling=modeCoupling)
    if realSpace == False:
      ftkappaMap, A_L_new = kappaTools.getKappa(sm0,sm1,crossForNormTE,crossForNormTT,crossForNormEE,pol=polComb,modeCoupling=modeCoupling)
    if simpleN0:
        pass
        print 'getting simple N0'
        #sm0 = kappaToolsReal.superMap(map0,noiseForFilter1,l,cl11,trimAtL=trimAtL,randomizePhase=randomizePhase,trimAtLFinal=trimAtLFinal)
        #sm1 = kappaToolsReal.superMap(map1,noiseForFilter2,l,cl22,trimAtL=trimAtL,randomizePhase=randomizePhase,trimAtLFinal=trimAtLFinal)
        A_L_new = kappaToolsSimpleN0.realspaceALSimpleN0(sm0,sm1,crossForNormTE,crossForNormTT,crossForNormEE,polComb=polComb)
    print 'recons end'
    return ftkappaMap, A_L_new   





def readSimArrays(simLocation,n,patch,iterationNum):
    if n == None:
        mapT = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'_T_'+patch+'.fits')
        print 'reading in file:', simLocation+'preparedSimMap'+iterationNum+'_T_'+patch+'.fits'
        mapE = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'_E_'+patch+'.fits')
        mapB = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'_B_'+patch+'.fits')
    else:
        try:
#          n = n+500
          mapT = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'%03d'%(n)+'_T_'+patch+'.fits')
          print 'reading in file track 1:', simLocation+'preparedSimMap'+iterationNum+'%03d'%(n)+'_T_'+patch+'.fits'
          mapE = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'%03d'%(n)+'_E_'+patch+'.fits')
          mapB = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+iterationNum+'%03d'%(n)+'_B_'+patch+'.fits')
#          print 'WARNING 500 HACK!!!'
#          n = n -500
        except:
          print 'doing exception', iterationNum
          newIter = (int(iterationNum)+n+10)%400#formerly n+10
          mapT = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+'%03d'%(newIter)+'_T_'+patch+'.fits')
          print 'reading in file track 2:', simLocation+'preparedSimMap'+'%03d'%(newIter)+'_T_'+patch+'.fits'
          mapE = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+'%03d'%(newIter)+'_E_'+patch+'.fits')
          mapB = liteMap.liteMapFromFits(simLocation+'preparedSimMap'+'%03d'%(newIter)+'_B_'+patch+'.fits')
    mapT.data -= mapT.data.mean()
    mapE.data -= mapE.data.mean()
    mapB.data -= mapB.data.mean()

    mapArray = [mapT,mapE,mapB]
    return mapArray, mapArray, mapArray, mapArray


def getKappaPower(mapArray1,mapArray2,mapArray3,mapArray4,clTheoryArray,noiseArray1,noiseArray2,polComb1,polComb2,A_L1,A_L2,trimAtL=None,randomizePhase=False,trimAtLFinal = None,modeCoupling='kappa',randomIterations=3,biasType=None,simLocation='../output/',patch='3',meanFieldSub=True,iterationNum='',simLocationN1 = '../output/', realSpace = True):
    kappaMapSave = None


    if biasType == 'namikawa': ### code to calculate namikawa bias ### averaging over random iterations
        kappaPowerMap = 0.
        for i in xrange(randomIterations):
            print 'namikawa initialize, iteration ', i, 'pol combinations', polComb1, polComb2
            mapArray1S,mapArray2S,mapArray3S,mapArray4S = readSimArrays(simLocation,2*i,patch,iterationNum)
            mapArray1SA,mapArray2SA,mapArray3SA,mapArray4SA = readSimArrays(simLocation,2*i+1,patch,iterationNum)
            #iterationNum = randomIterations                                                                                                                 
            kappaMap1SD, A_L = getKappaFunction(mapArray1S,mapArray2,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling,  realSpace = realSpace)
            kappaMap1DS, A_L = getKappaFunction(mapArray1,mapArray2S,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
            kappaMap1SSA, A_L = getKappaFunction(mapArray1S,mapArray2SA,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
            kappaMap2SAS, A_L = getKappaFunction(mapArray3SA,mapArray4S,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
            if ((mapArray1 == mapArray3)&(mapArray2==mapArray4))&(polComb1==polComb2):
                kappaMap2SD = kappaMap1SD.copy()
                kappaMap2DS = kappaMap1DS.copy()
                kappaMap2SSA = kappaMap1SSA.copy()
            else:
                kappaMap2SD, A_L = getKappaFunction(mapArray3S,mapArray4,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
                kappaMap2DS, A_L = getKappaFunction(mapArray3,mapArray4S,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
                kappaMap2SSA, A_L = getKappaFunction(mapArray3S,mapArray4SA,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
            kappaPowerMap += numpy.real(kappaMap1DS*numpy.conjugate(kappaMap2DS)+kappaMap1SD*numpy.conjugate(kappaMap2DS)+kappaMap1DS*numpy.conjugate(kappaMap2SD)+kappaMap1SD*numpy.conjugate(kappaMap2SD)-kappaMap1SSA*numpy.conjugate(kappaMap2SSA)-kappaMap1SSA*numpy.conjugate(kappaMap2SAS))/randomIterations


    elif biasType == 'N1': #### code to calculate N1 bias #### averaging over many sim iterations
        kappaPowerMap = 0.
        kappaMapMeanField = 0.
        nN1 = 1
        for i in xrange(nN1):
            print 'N1 iteration', i, nN1
#            iterationNum = '%03d'%(i)
            print 'namikawa initialize, iteration ', i, 'pol combinations', polComb1, polComb2
            mapArray1S,mapArray2S,mapArray3S,mapArray4S = readSimArrays(simLocation,None,patch,iterationNum)
            mapArray1SA,mapArray2SA,mapArray3SA,mapArray4SA = readSimArrays(simLocation,000,patch,iterationNum)
            mapArray1SP,mapArray2SP,mapArray3SP,mapArray4SP = readSimArrays(simLocationN1,None,patch,iterationNum)#readSimArrays(simLocationN1,000,patch,iterationNum)
            kappaMap1SSA, A_L = getKappaFunction(mapArray1S,mapArray2SA,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
            kappaMap2SAS, A_L = getKappaFunction(mapArray3SA,mapArray4S,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
            kappaMap1SSP, A_L = getKappaFunction(mapArray1S,mapArray2SP,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
            kappaMap2SPS, A_L = getKappaFunction(mapArray3SP,mapArray4S,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
            if ((mapArray1 == mapArray3)&(mapArray2==mapArray4))&(polComb1==polComb2):
                kappaMap2SSA = kappaMap1SSA.copy()
                kappaMap2SSP = kappaMap1SSP.copy()
            else:
                kappaMap2SSA, A_L = getKappaFunction(mapArray3S,mapArray4SA,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
                kappaMap2SSP, A_L = getKappaFunction(mapArray3S,mapArray4SP,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)

            kappaPowerMap += numpy.real(kappaMap1SSP*numpy.conjugate(kappaMap2SSP)+kappaMap1SSP*numpy.conjugate(kappaMap2SPS)-kappaMap1SSA*numpy.conjugate(kappaMap2SSA)-kappaMap1SSA*numpy.conjugate(kappaMap2SAS))/nN1


    elif biasType == 'simpleN0':
        print 'doing simple N0'
        kappaPowerMap = 0.
        kappaMapA, A_L = getKappaFunction(mapArray1,mapArray2,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling,simpleN0=True)
        kappaPowerMap += A_L

    elif randomizePhase: #randomize phase bias code ###
        kappaPowerMap = 0.
        for i in xrange(randomIterations):
            print 'randomizing phase now'
            mapArray1,mapArray2,mapArray3,mapArray4 = phaseRandomizeArrays(mapArray1,mapArray2,mapArray3,mapArray4)
            print 'end phase randomize'
            iterationNum = randomIterations
            kappaMapA, A_L = getKappaFunction(mapArray1,mapArray2,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)
            if (mapArray1 == mapArray3)&(mapArray2==mapArray4):
                kappaMapB = kappaMapA.copy()
            else:
                kappaMapB, A_L = getKappaFunction(mapArray3,mapArray4,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)

            kappaPowerMap += numpy.real(kappaMapA*numpy.conjugate(kappaMapB))/randomIterations


    else: #### core reconstruction code ### for use with data
        print 'reconstructing with pol combs', modeCoupling, polComb1, polComb2
        kappaMapA, A_L = getKappaFunction(mapArray1,mapArray2,clTheoryArray,noiseArray1,polComb1,A_L1,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling, realSpace = realSpace)
        kappaMapABackup = kappaMapA.copy()
        if meanFieldSub:
             meanField1 = pickle.load(open('../output/av_'+modeCoupling+patch+polComb1+'.pkl'))
             kappaMapA -= meanField1
             print 'mean subbed check' 

        if ((mapArray1 == mapArray3)&(mapArray2==mapArray4))&(polComb1==polComb2):
            kappaMapB = kappaMapA.copy()
        else:
            print 'good'
            kappaMapB, A_L = getKappaFunction(mapArray3,mapArray4,clTheoryArray,noiseArray2,polComb2,A_L2,trimAtL=trimAtL,randomizePhase=False,trimAtLFinal = trimAtLFinal,modeCoupling=modeCoupling)

            if meanFieldSub:
                 meanField2 = pickle.load(open('../output/av_'+modeCoupling+patch+polComb2+'.pkl'))
                 kappaMapB -= meanField2
        kappaPowerMap = numpy.real(kappaMapA*numpy.conjugate(kappaMapB))
        kappaMapSave = kappaMapABackup                                                                   

    return kappaPowerMap, kappaMapSave

