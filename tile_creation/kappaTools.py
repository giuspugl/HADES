from flipper import *
import pickle
from numpy.fft import fft2,ifft2,fftshift
from mapTools import *
import mapTools
import scipy



class superMap: ### initialize superMap object. assumes beams have been taken out.
    def __init__(self,map,noisePower,ell,cEllModel,trimAtL=None,randomizePhase=False,trimAtLFinal = None):
        
        self.map = map
        self.ftMap = fftTools.fftFromLiteMap(map)
        self.power2d = fftTools.powerFromLiteMap(map)
        if trimAtL != None:
            self.ftMapTrim = self.ftMap.trimAtL(trimAtL)
        self.trimAtL = trimAtL
        self.trimAtLFinal = trimAtLFinal
        self.noisePower = noisePower
        self.ell = ell
        self.cEllModel = cEllModel        
        self.twoDModel = mapTools.makeTemplate(ell,cEllModel,self.ftMap)
        self.filter = None
        self.filteredMap = None  
        
    def makeFilter(self,type,TE=False):
        """
        returns low/high pass filtered, beam deconvolved ftMap
        """
        if type !='lowPass' and type !='highPass':
            raise ValueError, 'type must be lowPass or highPass'
        if type == 'lowPass':
            self.filter = self.twoDModel/(self.twoDModel + self.noisePower)
        else:
            self.filter = (self.twoDModel*0.+1.)/(self.twoDModel + self.noisePower)
        if TE:
            self.filter = self.filter*0. + 1.
            print 'using TE in filter'
        id1 = numpy.where((self.ftMap.modLMap > self.trimAtL))
        self.filter[id1] = 0.
        self.filteredMap = self.filter*self.ftMap.kMap


def getKappa(sm0,sm1,crossPowerEstimateTE=1.,crossPowerEstimateTT=1.,crossPowerEstimateEE=1.,pol = 'TT',modeCoupling = 'kappa'):
    """
    calculates and returns kappa map. outermost wrapper.
    """
        
    if pol == 'TE':
        Ten=True
        print 'using TE'
    else:
        Ten=False
    
    sm1.makeFilter(type='lowPass',TE=Ten)
    sm0.makeFilter(type='highPass',TE=Ten)

    ftex, KappaFTMap, A_L = makeKappaMap(sm0,sm1,crossPowerEstimateTE,crossPowerEstimateTT,crossPowerEstimateEE,pol,modeCoupling) 

    fftFactor = sm0.power2d.pixScaleX*sm0.power2d.pixScaleY*numpy.sqrt(sm0.power2d.pixScaleX*sm0.power2d.pixScaleY/sm0.power2d.Nx/sm0.power2d.Ny) 
    print "FFT factor", fftFactor
    ## value is the relevant FFT factor for lens reconstruction.
    return KappaFTMap*fftFactor, A_L
    

def makeKappaMap(sm0,sm1,crossPowerEstimateTE,crossPowerEstimateTT,crossPowerEstimateEE,pol,modeCoupling):
    """
    calculates kappa map. sets up all required quantities: does fft shifting, calculates shifted maps, passes them to code that actually calculates integrals.
    Finds the normalization and noise bias of the lensing filter.                                                                                                                    
    This is different from normalize() as it does all operations in                                                                                                                   
    the 2-D L-space without resorting to interpolations.                                                                                                                                                    
    Uses NofL2D.                                                                                                                                                                                            
    trim2DAtL trims the 2D calculation at a given L 
    """
    
    if pol == 'TT':
        clMapLarge = numpy.fft.fftshift(crossPowerEstimateTT)
    elif pol == 'TE':
#        crossPowerEstimateTE[numpy.where(sm0.power2d.modLMap>sm0.trimAtL)] *=0.
#        sm0.noisePower *= 0.
#        sm0.noisePower[numpy.where(sm0.power2d.modLMap>sm0.trimAtL)] = 1.e20
        clMapLarge = numpy.fft.fftshift(crossPowerEstimateTE)
    else:
        clMapLarge = numpy.fft.fftshift(crossPowerEstimateEE)

    ft = sm0.power2d.copy()
    trimAtL = sm0.trimAtL
    trimAtLFinal = sm0.trimAtLFinal
    if sm0.trimAtL != None:
        ft = ft.trimAtL(trimAtL)

    # # # # initialize and fft shift all the relevant functions to construct the estimators

    lx = numpy.fft.fftshift(ft.lx)#an array giving the lx values #
    ly = numpy.fft.fftshift(ft.ly)#
    deltaLy = numpy.abs(ly[0]-ly[1])
    deltaLx = numpy.abs(lx[0]-lx[1])
    modLMap = numpy.fft.fftshift(ft.modLMap) #
    thetaMap = numpy.fft.fftshift(ft.thetaMap) #
    cosTheta = numpy.cos(thetaMap*numpy.pi/180.)
    ftb = sm0.power2d.copy()#this is the big, untrimmed version - so that you can use trim shift.
    ftKappa = sm0.ftMap.kMap
    lxb = numpy.fft.fftshift(ftb.lx) #
    lyb = numpy.fft.fftshift(ftb.ly) #
    kappaFTMap = trimShiftKMap2(numpy.fft.fftshift(ftKappa),trimAtL,0,0,lxb,lyb) ##
    lxMap = kappaFTMap.copy()
    lyMap = kappaFTMap.copy()
    print 'a'
    for p in xrange(len(ly)):
        lxMap[p,:] = lx[:] 
    for q in xrange(len(lx)):
        lyMap[:,q] = ly[:]  
    print 'b'
    cl0 = trimShiftKMap2(clMapLarge,trimAtL,0,0,lxb,lyb)

    W = trimShiftKMap2(numpy.fft.fftshift(sm0.filteredMap),trimAtL,0,0,lxb,lyb)##
    w = trimShiftKMap2(numpy.fft.fftshift(sm0.filter),trimAtL,0,0,lxb,lyb)##33
    G = mapFlip2(numpy.fft.fftshift(sm1.filteredMap.copy()),lxb,lyb) #flip low pass filtered map - used to be lowPassBackup = numpy.fft.fftshift(sm1.filteredMap.copy())    
    g = mapFlip2(numpy.fft.fftshift(sm1.filter.copy()),lxb,lyb) # flip low pass filter used to be lpFilterB = numpy.fft.fftshift(sm1.filter.copy())

    print 'using new TE version'
    
    # # # # end of initializations and fftshifts

    normalizationNew = kappaFTMap.copy()
    noisebias = kappaFTMap.copy()
   
    if pol=='TE':
#        cl0[numpy.where(modLMap>trimAtL)] *= 0.
        clTT = trimShiftKMap2(numpy.fft.fftshift(crossPowerEstimateTT),trimAtL,0,0,lxb,lyb)
        clEE = trimShiftKMap2(numpy.fft.fftshift(crossPowerEstimateEE),trimAtL,0,0,lxb,lyb) 
        clNoise = trimShiftKMap2(numpy.fft.fftshift(sm0.noisePower),trimAtL,0,0,lxb,lyb) 
        clNoiseE = trimShiftKMap2(numpy.fft.fftshift(sm1.noisePower),trimAtL,0,0,lxb,lyb) 
        TENarray = [numpy.fft.fftshift(crossPowerEstimateTT),numpy.fft.fftshift(crossPowerEstimateEE),numpy.fft.fftshift(sm0.noisePower),numpy.fft.fftshift(sm1.noisePower),clTT,clEE,clNoise,clNoiseE]
    else:
        TENarray = []

    print "p,q are:",p,q
    count = 0.
        
    for i in xrange(len(ly)):
        a = time.time()
        for j in xrange(len(lx)):
            ## new speedup attempt
            absOfL = lx[j]**2.+ly[i]**2.
            if absOfL < trimAtLFinal**2.:
                kappaFTMap[i,j], normalizationNew[i,j], atL, timeratio, timeratio2 = \
                    kappaIntegral(clMapLarge,cl0,deltaLx,deltaLy,lx[j],ly[i],lxMap,lyMap,lxb,lyb,W,G,w,g,trimAtL,pol,TENarray, modeCoupling)
            else:
                kappaFTMap[i,j] = 0.
                normalizationNew[i,j] = 0.
            #note lx[j] and ly[i] are Lx and Ly, i.e. the pixel coordinates in kspace for the convergence map
    ftransf = ft.copy()
        
    return ftransf, numpy.fft.ifftshift(kappaFTMap), numpy.fft.ifftshift(normalizationNew)

def kappaIntegral(clMapLarge,cl0,deltaLx,deltaLy,Lx,Ly,lx,ly,lxb,lyb,W,G,w,g,trimAtL,pol,TENarray, modeCoupling):
    """
    @brief Computes kappa at a given L. This is the 'heart' of the reconstruction code.
    @param L the multipole to calculate at.
     clMapLarge = cl0 = TENarray - same info 3 times - just data is shifted 
    """
    # W is high pass filtered map, w is just the filter. similar for G and g, but low-pass.
    # Note El = l and L = L    
    # Note also lx is a 2d map of lx. earlier called lxMap but here renamed for convenience. same for ly.
    one = time.time()
    Lsqrd = Lx**2. + Ly**2.    # L^2

    LdotEl = Lx*lx + Ly*ly     # L dot El
    LcurlEl = Lx*ly - Ly*lx     # L dot El

    LdotElPrime = Lx**2. + Ly**2. - (Lx*lx+Ly*ly)  # L dot (L - El) = L^2 - L dot El
    LcurlElPrime = Lx * (Ly - ly) - Ly * (Lx - lx)

    if modeCoupling == 'kappa':
        LstarEl = LdotEl
        LstarElPrime = LdotElPrime

    elif modeCoupling == 'omega':
        LstarEl = LcurlEl
        LstarElPrime = LcurlElPrime
#        print "curl reconstruction"
 
    elif modeCoupling == 'tau':
        LstarEl = 1.
        LstarElPrime = 1.
        print "Warning, modeCoupling = tau is not fully tested yet."
    else:
        print "bad input"

    three = time.time()

    G_shifted =  trimShiftKMap2(G,trimAtL,-Lx,-Ly,lxb,lyb)  # G(L - El) where G(l) is defined in Hu 2001 Eq 5 without the l vector; note this is really G(El - L) of the flipped map 

    four = time.time()

    # # #    W = hpKMap                 # W(El) where W(l) is defined in Hu 2001 Eq 6

    kappaL_integral = (1./(2.*numpy.pi)**2.) * W * G_shifted * LstarElPrime * deltaLx *deltaLy  # Kappa(L) from Hu 2001 Eq 8, where Kappa(L)=(1/2)*L*D(L) 
    two = time.time()
    diff = two - one

    clL = trimShiftKMap2(clMapLarge,trimAtL,-Lx,-Ly,lxb,lyb)     #This is C_El-prime = C(L-l)

    if pol == 'EE':
        fact = numpy.cos(2.*(numpy.arctan2(numpy.real(Ly-ly),numpy.real(Lx-lx))-numpy.arctan2(numpy.real(ly),numpy.real(lx))))
        kappaL_integral *=fact
    if pol == 'EB':
        fact = numpy.sin(2.*(numpy.arctan2(numpy.real(Ly-ly),numpy.real(Lx-lx))-numpy.arctan2(numpy.real(ly),numpy.real(lx))))
        kappaL_integral *=fact
    if pol == 'TE':
        clTT0 = TENarray[4]#3]
        clEE0 = TENarray[5]#4]
        clN0 = TENarray[6]#5]
        clEEN0 = TENarray[7]#5]
        clTTL = trimShiftKMap2(TENarray[0],trimAtL,-Lx,-Ly,lxb,lyb)
        clEEL = trimShiftKMap2(TENarray[1],trimAtL,-Lx,-Ly,lxb,lyb)
        clNL = trimShiftKMap2(TENarray[2],trimAtL,-Lx,-Ly,lxb,lyb)
        clEENL = trimShiftKMap2(TENarray[3],trimAtL,-Lx,-Ly,lxb,lyb)
        fact = numpy.cos(2.*(numpy.arctan2(numpy.real(Ly-ly),numpy.real(Lx-lx))-numpy.arctan2(numpy.real(ly),numpy.real(lx))))
        f = cl0 * LstarEl * fact + clL * LstarElPrime
        fp = cl0 * LstarEl + clL * LstarElPrime * fact
        F = (clEE0+clEEN0) * (clTTL+clNL)*f-cl0*clL*fp
        F /= (clTT0+clN0)*(clEEL+clEENL)*(clEE0+clEEN0)*(clTTL+clNL)-cl0**2.*clL**2.
        kappaL_integral =  1./(2.*numpy.pi)**2.*W*G_shifted*F*deltaLx*deltaLy 

    kappaL_integral = numpy.nan_to_num(kappaL_integral)
    KappaAtL = kappaL_integral.sum()   # Integrates kappaL_integral to get Kappa(L)
        
    diff2 = four - three

    #Calculate N_L and Multiply Kappa(L) by N_L
    # # #    w = hpf   # w = 1/Cl^tot            see Eq 6 of Hu 2001
    # # #    g = lpf   # g = Cl_unlensed/Cl^tot  see Eq 5 of Hu 2001        
    g_shifted = trimShiftKMap2(g,trimAtL,-Lx,-Ly,lxb,lyb)  # Change g(El) to g(L - El)
    if pol == 'TT':
        NLinv_integral = 2./Lsqrd*1./(2.*numpy.pi)**2.*(w*g_shifted)*LstarElPrime*(LstarElPrime*clL+LstarEl*cl0)*deltaLx*deltaLy
    if pol == 'TE':
        NLinv_integral = 2./Lsqrd*1./(2.*numpy.pi)**2.*f*F*(w*g_shifted)*deltaLx*deltaLy
    if pol == 'EE':
        NLinv_integral = 2./Lsqrd*1./(2.*numpy.pi)**2.*(w*g_shifted)*LstarElPrime*(LstarElPrime*clL+LstarEl*cl0)*deltaLx*deltaLy
        NLinv_integral *= fact**2.
    if pol == 'EB':
        NLinv_integral = 2./Lsqrd*1./(2.*numpy.pi)**2.*(w*g_shifted)*LstarElPrime*(LstarElPrime*clL)*deltaLx*deltaLy
        NLinv_integral *= fact**2.

    NLinv_integral = numpy.nan_to_num(NLinv_integral)  

    N_L = 1./NLinv_integral.sum()  # Integrates NLinv_integral to calculate N_L
    
    KappaAtL *= N_L                # Returns kappa(L)
    
    return KappaAtL, N_L*2./4.*Lsqrd, numpy.sqrt(Lsqrd), diff, diff2
    
