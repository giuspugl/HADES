from flipper import *
import pickle
from numpy.fft import fft2,ifft2,fftshift
from mapTools import *
import mapTools
import scipy
from kappaFilters import *

class superMap: ### initialize superMap object. assumes beams have been taken out.
    def __init__(self,map,noisePower,ell,cEllModel,trimAtL=None,randomizePhase=False):
        
        self.map = map
        self.ftMap = fftTools.fftFromLiteMap(map)
        self.power2d = fftTools.powerFromLiteMap(map)
        if trimAtL != None:
            self.ftMapTrim = self.ftMap.trimAtL(trimAtL)
        self.trimAtL = trimAtL        
        self.noisePower = noisePower
        self.ell = ell
        self.cEllModel = cEllModel        
        self.twoDModel = mapTools.makeTemplate(ell,cEllModel,self.ftMap)
        self.filter = None
        self.filteredMap = None  
        
    def makeGeneralFilter(self,mapFilter2D):
        """
        returns low/high pass filtered, beam deconvolved ftMap
        """
        self.filteredMap = None
        self.filter = mapFilter2D
        id1 = numpy.where((self.ftMap.modLMap > self.trimAtL))
        self.filter[id1] = 0.
        self.filteredMap = self.filter*self.ftMap.kMap

    def makeRealSpaceFilteredMap(self):
        self.ftOfFilteredMap = self.ftMap.copy()
        self.ftOfFilteredMap.kMap =  self.filteredMap
        self.realFilteredMap = self.map.copy()
        self.realFilteredMap.data *= 0.
        self.realFilteredMap.data = numpy.fft.ifft2(self.ftOfFilteredMap.kMap)#self.ftOfFilteredMap.mapFromFFT()

def getKappaReal(sm0,sm1,A_L,crossPowerEstimateTE=1.,crossPowerEstimateTT=1.,crossPowerEstimateEE=1.,pol = 'TT'):
    """
    calculates and returns kappa map. outermost wrapper.
    """

    ft = sm0.power2d
    lx = ft.lx
    ly = ft.ly
    lxMap = ft.powerMap.copy()
    lyMap = ft.powerMap.copy()
    for p in xrange(len(ly)):
        lxMap[p,:] = lx[:] 
    for q in xrange(len(lx)):
        lyMap[:,q] = ly[:]  
    
    lxMap /= ft.modLMap
    lyMap /= ft.modLMap

    ####
    lxMap[0,0] = 0.
    lyMap[0,0] = 0.
    ####

    kappaMapLx = sm0.map.copy()
    kappaMapLx.data *= 0.
    kappaMapLy = sm0.map.copy()
    kappaMapLy.data *= 0.

    if pol == 'TT':
        filterListLx0, filterListLx1 = makeFilterListsTTLx(sm0.twoDModel,sm0.noisePower,sm0.ftMap,lxMap,lyMap)
        filterListLy0, filterListLy1 = makeFilterListsTTLy(sm1.twoDModel,sm1.noisePower,sm1.ftMap,lxMap,lyMap)

    if pol == 'EB':
        filterListLx0, filterListLx1 = makeFilterListsEBLx(sm0.twoDModel,sm0.noisePower,sm0.ftMap,lxMap,lyMap)
        filterListLy0, filterListLy1 = makeFilterListsEBLy(sm1.twoDModel,sm1.noisePower,sm1.ftMap,lxMap,lyMap)

    if pol == 'EE':
        filterListLx0, filterListLx1 = makeFilterListsEELx(sm0.twoDModel,sm0.noisePower,sm0.ftMap,lxMap,lyMap)
        filterListLy0, filterListLy1 = makeFilterListsEELy(sm1.twoDModel,sm1.noisePower,sm1.ftMap,lxMap,lyMap)

    length = len(filterListLx0)
    verbose = False#True
    for i in xrange(length):
        filterListLx0[i][0,0]=0.
        sm0.makeGeneralFilter(filterListLx0[i])
        sm0.makeRealSpaceFilteredMap()
        mapLx0Data = sm0.realFilteredMap.data
        
        filterListLx1[i][0,0]=0.
        sm1.makeGeneralFilter(filterListLx1[i])
        sm1.makeRealSpaceFilteredMap()
        mapLx1Data = sm1.realFilteredMap.data

        kappaMapLx.data += mapLx0Data*mapLx1Data
            

        filterListLy0[i][0,0]=0.
        sm0.makeGeneralFilter(filterListLy0[i])
        sm0.makeRealSpaceFilteredMap()
        mapLy0Data = sm0.realFilteredMap.data

        filterListLy1[i][0,0]=0.
        sm1.makeGeneralFilter(filterListLy1[i])
        sm1.makeRealSpaceFilteredMap()
        mapLy1Data = sm1.realFilteredMap.data
        print mapLy0Data, 'mapLy0Data'
        print mapLy1Data, 'mapLy1Data'
        kappaMapLy.data += mapLy0Data*mapLy1Data


        if verbose:####
            p2dLy = fftTools.powerFromLiteMap(sm0.realFilteredMap)
            p2dLy.plot(zoomUptoL = 4000.)
            pylab.savefig('mapLy0Power.png')
            pylab.clf()
            p2dLy = fftTools.powerFromLiteMap(sm1.realFilteredMap)
            p2dLy.plot(zoomUptoL = 4000.)
            pylab.savefig('mapLy1Power.png')
            pylab.clf()

            p2d = fftTools.powerFromLiteMap(kappaMapLy)
            p2d.plot(zoomUptoL = 4000.)
            pylab.savefig('mapProductPower.png')
            pylab.clf()
            ll, ll, lBin, cl, ll, ll = p2d.binInAnnuli('BIN_100_LOG')
            pylab.loglog(lBin,cl)
            pylab.savefig('mapProductPowerBinned.png')
            ############

###        add section filter with lx and al
        ftLx = fftTools.fftFromLiteMap(kappaMapLx)
        ftLx.kMap *= lxMap*ft.modLMap*(0.+1.j)
        ftLx = ftLx.trimAtL(sm0.trimAtL)
        ftLx.kMap *= A_L/ftLx.modLMap**2.*2.

        ftLy = fftTools.fftFromLiteMap(kappaMapLy)
        ftLy.kMap *= lyMap*ft.modLMap*(0.+1.j)

        if verbose:####
            pylab.clf()
            p2d = fftTools.powerFromFFT(ftLy)
            ll, ll, lBin, cl, ll, ll = p2d.binInAnnuli('BIN_100_LOG')
            pylab.loglog(lBin,cl)
            pylab.savefig('trueKappaTimesLy.png')
            pylab.clf()
            ###########

        ftLy = ftLy.trimAtL(sm0.trimAtL)

        if verbose:####
            p2d = p2d.trimAtL(2700.)
            p2d.powerMap = (A_L/ftLy.modLMap**2.*2.)**2.
            ll, ll, lBin, cl, ll, ll = p2d.binInAnnuli('BIN_100_LOG')
            pylab.loglog(lBin,cl)
            pylab.savefig('normalizationFactor.png')
            ###########

        ftLy.kMap *= A_L/ftLy.modLMap**2.*2.

        ftFinalKappa = ftLx.copy()
        ftFinalKappa.kMap += ftLy.kMap

###
 
    factor = numpy.sqrt(sm0.power2d.pixScaleX*sm0.power2d.pixScaleY/sm0.power2d.Nx/sm0.power2d.Ny)
    print factor
    return ftFinalKappa.kMap*factor, A_L


