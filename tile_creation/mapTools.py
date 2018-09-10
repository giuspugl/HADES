from flipper import *
from scipy import * 
import scipy.interpolate
import numpy
import pdb


def makeTemplate(l,Fl,ftMap):
    """                                                                                                                                                    
    Given 1d function Fl of l, creates the 2d version                                                                                                      
    of Fl on 2d k-space defined by ftMap                                                                                                                   
    """
    tck = scipy.interpolate.splrep(l,Fl,k=1)
    template = scipy.interpolate.splev(ftMap.modLMap.ravel(),tck)
    template = numpy.reshape(template,[ftMap.Ny,ftMap.Nx])
    return template


def trimShiftKMap(kMap,elTrim,Lx,Ly,lx,ly):
    """                                                                                                                                                    
    @brief Trims a 2-D powerMap at a certain Lx, Ly, and returns the trimmed power2D object. Note                                                          
    that the pixel scales are adjusted so that the trimmed dimensions correspond                                                                           
    to the same sized map in real-space (i.e. trimming -> poorer resolution real space map).                                                               
    Can be used to shift maps.                                                                                                                             
    @pararm elTrim real >0 ; the l to trim at                                                                                                              
    @return power2D instance                                                                                                                               
    """

    assert(elTrim>0.)
    idx = numpy.where((lx < elTrim+Lx) & (lx > -elTrim+Lx))
    idy = numpy.where((ly < elTrim+Ly) & (ly > -elTrim+Ly))

    trimA = kMap[idy[0],:] # note this used to be kM!!! not kMap                                                                                           
    trimB = trimA[:,idx[0]]

    return trimB


def trimShiftKMap2(kMap,elTrim,Lx,Ly,lx,ly):
    """                                                                                                                                                                        
    @brief Trims a 2-D powerMap at a certain Lx, Ly, and returns the trimmed power2D object. Note                                                                              
    that the pixel scales are adjusted so that the trimmed dimensions correspond                                                                                               
    to the same sized map in real-space (i.e. trimming -> poorer resolution real space map).                                                                                   
    Can be used to shift maps.                                                                                                                                                 
    @pararm elTrim real >0 ; the l to trim at                                                                                                                                  
    @return power2D instance                                                                                                                                                   
    """

    assert(elTrim>0.)
    deltalx = lx[1] - lx[0]
    deltaly = ly[1] - ly[0]
    i1 = (-elTrim+Lx - lx[0])/deltalx
    j1 = (-elTrim+Ly - ly[0])/deltaly
    i2 =  (elTrim+Lx - lx[0])/deltalx
    j2 =  (elTrim+Ly - ly[0])/deltaly
    i1 = ceil(i1)
    j1 = ceil(j1)
    nx = len(lx)-1
    ny = len(ly)-1
    i2 = ceil(nx-i2)
    j2 = ceil(ny-j2)
    trimmedMap=kMap[j1:-j2,i1:-i2]
    return trimmedMap


def mapFlip(kMap,lx,ly):
    kMapNew = kMap.copy()
    for i in ly:
        for j in lx:
            iy0 = numpy.where(ly == i)
            ix0 = numpy.where(lx == j)
            iy = numpy.where(ly == -i)
            ix = numpy.where(lx == -j)
            try:
                kMapNew[iy0[0],ix0[0]] = kMap[iy[0],ix[0]]
            except:
                pass
    return kMapNew


def mapFlip2(kMap,lx,ly):
    kMapNew = kMap.copy()
    kMapNew2 = kMap.copy()
    nx = len(lx)
    ny = len(ly) 

    for i in range(0,ny):
        for j in range(0,nx):
            try:
                kMapNew[i,j] = kMap[ny-1-i,nx-1-j]
            except:
                pass
    if ny%2==0:    # if axis is even, need to roll back 1 for correct mapFlip
        kMapNew2=numpy.roll(kMapNew,1,axis=0)
    if ny%2==0 and nx%2==0:
        kMapNew2=numpy.roll(kMapNew2,1,axis=1)
    if ny%2!=0 and nx%2== 0:
        kMapNew2=numpy.roll(kMapNew,1,axis=1)
    if ny%2!=0 and nx%2!=0:
        kMapNew2 = kMapNew.copy()
    return kMapNew2


def xs_and_ys(nx, ny):
    xvals = range(nx)
    xvals = numpy.tile(xvals, (ny,1))
    yvals = range(ny)
    yvals = numpy.transpose(numpy.tile(yvals, (nx,1)))
    return xvals, yvals


def gausstaper(nx, ny, width, center = None, width_2 = None):
    centervals = (numpy.array([nx/2, ny/2]) if center == None  else center)
    if width_2 == None :
        width_2 = width
    xx, yy = xs_and_ys(nx, ny)
    newarr = numpy.exp( - (xx - centervals[0])**2 / (2. * width_2**2) -  (yy - centervals[1])**2  / (2. * width**2))
    return newarr
                       



                       

