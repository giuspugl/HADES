from flipper import *
import numpy as np
from math import pi
from scipy.interpolate import splrep,splev
import pyfits
import fftTools

def ConjugateGradientSolver(my_map,my_map_2,Mask,power_spec,nxside,nyside,eps):
    
    def apply_px_c_inv_px(my_map):
        
        my_map.shape=(nyside,nxside)
        #apply x_proj
        my_new_map=(1-Mask)*my_map
        # change to Fourrier representation
        a_l=np.fft.fft2(my_new_map)
        # apply inverse power spectrum
        a_l=a_l*1/power_spec
        # change back to pixel representation
        my_new_map=np.fft.ifft2(a_l)
        #Remove the imaginary part
        my_new_map=my_new_map.real
        # apply x_proj
        my_new_map=(1-Mask)*my_new_map
        #Array to vector
        my_new_map.shape=(nxside*nyside,)
        my_map.shape=(nxside*nyside,)
        return(my_new_map)
    
    def apply_px_c_inv_py(my_map):
        # apply y_proj
        my_map.shape=(nyside,nxside)
        my_new_map=Mask*my_map
        # change to Fourrier representation
        a_l=np.fft.fft2(my_new_map)
        # apply inverse power spectrum
        a_l=a_l*1/power_spec
        # change back to pixel representation
        my_new_map=np.fft.ifft2(a_l)
        #Remove the imaginary part
        my_new_map=my_new_map.real
        # apply x_proj
        my_new_map=(1-Mask)*my_new_map
        #Array to vector
        my_new_map.shape=(nxside*nyside,)
        return(my_new_map)
    
    b=-apply_px_c_inv_py(my_map-my_map_2)
    
    #Number of iterations
    i_max=2000
    
    #initial value of x
    x=b
    i=0
    
    r=b-apply_px_c_inv_px(x)
    d=r
    
    delta_new=np.inner(r,r)
    delta_o=delta_new
    
    delta_array=np.zeros(shape=(i_max))
    
    while i<i_max and delta_new > eps**2*delta_o:
        #print ""
        #print "number of iterations:", i
        #print ""
        #print "eps**2*delta_o=",eps**2*delta_o
        #print ""
        #print "delta new=",delta_new
        
        q=apply_px_c_inv_px(d)
        alpha=delta_new/(np.inner(d,q))
        x=x+alpha*d
        
        if i/50.<numpy.int(i/50):
            
            r=b-apply_px_c_inv_px(x)
        else:
            r=r-alpha*q
        
        delta_old=delta_new
        delta_new=np.inner(r,r)
        beta=delta_new/delta_old
        d=r+beta*d
        i=i+1
    
    #print "delta_o=", delta_o
    #print "delta_new=", delta_new
    
    x.shape=(nyside,nxside)
    x_old=x
    x=x+my_map_2*(1-Mask)
    complete=my_map*Mask
    rebuild_map=complete+x
    
    return(rebuild_map)



def make2dPowerSpectrum(map,l,cl):
    
    ly = numpy.fft.fftfreq(map.Ny,d = map.pixScaleY)*(2*numpy.pi)
    lx = numpy.fft.fftfreq(map.Nx,d = map.pixScaleX)*(2*numpy.pi)
    modLMap = numpy.zeros([map.Ny,map.Nx])
    iy, ix = numpy.mgrid[0:map.Ny,0:map.Nx]
    modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
    s = splrep(l,cl,k=3)
    ll = numpy.ravel(modLMap)
    kk = splev(ll,s)
    id = numpy.where(ll>l.max())
    kk[id]=cl[-1]
    area = map.Nx*map.Ny*map.pixScaleX*map.pixScaleY
    power_2d = numpy.reshape(kk,[map.Ny,map.Nx])/area * (map.Nx*map.Ny)**2
    return(power_2d)


def fillFromP2d(liteMap,power_2d):
    
    realPart = numpy.sqrt(power_2d)*numpy.random.randn(liteMap.Ny,liteMap.Nx)
    imgPart = numpy.sqrt(power_2d)*numpy.random.randn(liteMap.Ny,liteMap.Nx)
    kMap = realPart+1j*imgPart
    data = numpy.real(numpy.fft.ifft2(kMap))
    data-=numpy.mean(data)
    b = 1
    liteMap.data = data[(b-1)/2*liteMap.Ny:(b+1)/2*liteMap.Ny,(b-1)/2*liteMap.Nx:(b+1)/2*liteMap.Nx]
    
    return liteMap


def Apply1dBeam(map,beam1d):
    ell, f_ell = numpy.transpose(numpy.loadtxt(beam1d['file']))
    t = makeTemplate( map, f_ell, ell, ell.max())
    ft = numpy.fft.fft2(map.data)
    ft*= t.data
    map.data[:]=numpy.real(numpy.fft.ifft2(ft))
    return map


def makeEmptyCEATemplate(raSizeDeg, decSizeDeg,meanRa = 180., meanDec = 0.,\
                         pixScaleXarcmin = 0.5, pixScaleYarcmin=0.5):
    assert(meanDec == 0.,'mean dec other than zero not implemented yet')
    
    
    cdelt1 = -pixScaleXarcmin/60.
    cdelt2 = pixScaleYarcmin/60.
    naxis1 = numpy.int(raSizeDeg/pixScaleXarcmin*60.+0.5)
    naxis2 = numpy.int(decSizeDeg/pixScaleYarcmin*60.+0.5)
    refPix1 = naxis1/2.
    refPix2 = naxis2/2.
    pv2_1 = 1.0
    cardList = pyfits.CardList()
    cardList.append(pyfits.Card('NAXIS', 2))
    cardList.append(pyfits.Card('NAXIS1', naxis1))
    cardList.append(pyfits.Card('NAXIS2', naxis2))
    cardList.append(pyfits.Card('CTYPE1', 'RA---CEA'))
    cardList.append(pyfits.Card('CTYPE2', 'DEC--CEA'))
    cardList.append(pyfits.Card('CRVAL1', meanRa))
    cardList.append(pyfits.Card('CRVAL2', meanDec))
    cardList.append(pyfits.Card('CRPIX1', refPix1+1))
    cardList.append(pyfits.Card('CRPIX2', refPix2+1))
    cardList.append(pyfits.Card('CDELT1', cdelt1))
    cardList.append(pyfits.Card('CDELT2', cdelt2))
    cardList.append(pyfits.Card('CUNIT1', 'DEG'))
    cardList.append(pyfits.Card('CUNIT2', 'DEG'))
    hh = pyfits.Header(cards=cardList)
    wcs = astLib.astWCS.WCS(hh, mode='pyfits')
    data = numpy.zeros([naxis2,naxis1])
    ltMap = liteMap.liteMapFromDataAndWCS(data,wcs)
    
    return ltMap


def makeMask(liteMap,nHoles,holeSize,lenApodMask,show=True):
    
    pixScaleArcmin=liteMap.pixScaleX*60*360/numpy.pi
    holeSizePix=numpy.int(holeSize/pixScaleArcmin)
    
    mask=liteMap.copy()
    mask.data[:]=1
    holeMask=mask.copy()
    
    Nx=mask.Nx
    Ny=mask.Ny
    xList=numpy.random.rand(nHoles)*Nx
    yList=numpy.random.rand(nHoles)*Ny
    
    for k in range(nHoles):
    	#print "number of Holes",k
        holeMask.data[:]=1
        for i in range(Nx):
            for j in range(Ny):
            	rad=(i-numpy.int(xList[k]))**2+(j-numpy.int(yList[k]))**2
            	
            	if rad < holeSizePix**2:
                    holeMask.data[j,i]=0
                for pix in range(lenApodMask):
                	
                    if rad <= (holeSizePix+pix)**2 and rad > (holeSizePix+pix-1)**2:
                        holeMask.data[j,i]=1./2*(1-numpy.cos(-numpy.pi*float(pix)/lenApodMask))
        mask.data[:]*=holeMask.data[:]
    data=mask.data[:]

    if show==True:
    	pylab.matshow(data)
    	pylab.show()


    return mask


def addWhiteNoise(map,rmsArcmin):
    """
        Adds white noise to a given map; returns a new map
        """
    noisyMap = map.copy()
    if rmsArcmin == 0.0:
        pass
    else:
        radToMin = 180/numpy.pi*60
        pixArea = radToMin**2 * map.pixScaleX*map.pixScaleY
        rms = rmsArcmin/numpy.sqrt(pixArea)
        
        noise = numpy.random.normal( scale = rms, size = map.data.shape )
        
        noisyMap.data[:] += noise[:]
    
    return noisyMap

def makeTemplate(m, wl, ell, maxEll, outputFile = None):
    """
        For a given map (m) return a 2D k-space template from a 1D specification wl
        ell = 2pi * i / deltaX
        (m is not overwritten)
        """
    
    ell = numpy.array(ell)
    wl  = numpy.array(wl)
    
    
    fT = fftTools.fftFromLiteMap(m)
    print "max_lx, max_ly", fT.lx.max(), fT.ly.max()
    print "m_dx, m_dy", m.pixScaleX, m.pixScaleY
    print "m_nx, m_ny", m.Nx, m.Ny
    l_f = numpy.floor(fT.modLMap)
    l_c = numpy.ceil(fT.modLMap)
    fT.kMap[:,:] = 0.
    
    for i in xrange(numpy.shape(fT.kMap)[0]):
        for j in xrange(numpy.shape(fT.kMap)[1]):
            if l_f[i,j] > maxEll or l_c[i,j] > maxEll:
                continue
            w_lo = wl[l_f[i,j]]
            w_hi = wl[l_c[i,j]]
            trueL = fT.modLMap[i,j]
            w = (w_hi-w_lo)*(trueL - l_f[i,j]) + w_lo
            fT.kMap[i,j] = w
    
    m = m.copy()
    m.data = abs(fT.kMap)
    if outputFile != None:
        m.writeFits(outputFile, overWrite = True)
    return m


def fillWithGaussianRandomField(self,ell,Cell,bufferFactor = 1):

    ft = fftTools.fftFromLiteMap(self)
    Ny = self.Ny*bufferFactor
    Nx = self.Nx*bufferFactor
    bufferFactor = int(bufferFactor)
    realPart = numpy.zeros([Ny,Nx])
    imgPart  = numpy.zeros([Ny,Nx])
    ly = numpy.fft.fftfreq(Ny,d = self.pixScaleY)*(2*numpy.pi)
    lx = numpy.fft.fftfreq(Nx,d = self.pixScaleX)*(2*numpy.pi)
    modLMap = numpy.zeros([Ny,Nx])
    iy, ix = numpy.mgrid[0:Ny,0:Nx]
    modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
    s = splrep(ell,Cell,k=3)
    ll = numpy.ravel(modLMap)
    kk = splev(ll,s)
    id = numpy.where(ll>ell.max())
    kk[id] = Cell[-1] 

    area = Nx*Ny*self.pixScaleX*self.pixScaleY
    p = numpy.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2
        
    realPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
    imgPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
    kMap = realPart+1j*imgPart
    data = numpy.real(numpy.fft.ifft2(kMap))
        
    b = bufferFactor
    self.data = data[(b-1)/2*self.Ny:(b+1)/2*self.Ny,(b-1)/2*self.Nx:(b+1)/2*self.Nx]
    return(self)



