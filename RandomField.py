import numpy as np
from .params import BICEP
a=BICEP()
from flipper import *

def real_fill_from_Cell(liteMap,ell,Cell,bufferFactor = 1,lMin=a.lMin,log=True,returnAll=False,padded_template=None,fourier=False):
        """
        Generates a GRF from an input power spectrum specified as ell, Cell 
        BufferFactor =1 means the map will be periodic boundary function
        BufferFactor > 1 means the map will be genrated on  a patch bufferFactor times 
        larger in each dimension and then cut out so as to have non-periodic bcs.
        returnAll -> if True, also return complete map without cutting down to size (if bufferFactor>1)
        padded_template -> template map if returnAll is used
        
        Fills the data field of the map with the GRF realization.
        
        Yanked from liteMap.fillWithGaussianRandomField -> added log-spline fitting
        """
        import numpy
        
        self=liteMap
        ft = fftTools.fftFromLiteMap(self)
        
        # Discard high ell pixels
        #ft=ft.trimAtL(lCut)
        Ny = self.Ny*bufferFactor
        Nx = self.Nx*bufferFactor
        #self.Nx=Nx
        #self.Ny=Ny
        #self.pixScaleX=ft.pixScaleX
        #self.pixScaleY=ft.pixScaleY
        
        bufferFactor = int(bufferFactor)
        
        
        realPart = numpy.zeros([Ny,Nx])
        imgPart  = numpy.zeros([Ny,Nx])
        
        ly = numpy.fft.fftfreq(Ny,d = self.pixScaleY)*(2*numpy.pi)
        lx = numpy.fft.fftfreq(Nx,d = self.pixScaleX)*(2*numpy.pi)
        #print ly
        modLMap = numpy.zeros([Ny,Nx])
        iy, ix = numpy.mgrid[0:Ny,0:Nx]
        modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
        
	# Fit input Cell to spline
	from scipy.interpolate import UnivariateSpline
	if log:
		spl=UnivariateSpline(ell,np.log(Cell),k=5) # use a quintic spline here
	else:
		spl=UnivariateSpline(ell,Cell,k=5)
	ll=np.ravel(modLMap)
	kk=np.zeros_like(ll)#np.ones_like(ll)*1e-40
	
	# Apply filtering
	idhi=np.where(ll>max(ell))
	idlo=np.where(ll<lMin)
	idgood=np.where((ll<max(ell))&(ll>lMin))
	if log:
		kk[idgood]=np.exp(spl(ll[idgood]))
	else:
		kk[idgood]=spl(ll[idgood])
	kk[idhi]=Cell[-1] #min(kk[idgood]) # set unwanted values to small value
	kk[idlo]=Cell[0]
	
        area = Nx*Ny*self.pixScaleX*self.pixScaleY
        p = numpy.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2        
        
       	# Compute real + imag parts
	realPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	imgPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	
        kMap = realPart+1.0j*imgPart
        
        #data = numpy.real(numpy.fft.ifft2(kMap)) 
        data=numpy.fft.ifft2(kMap)
        
        b = bufferFactor
        self.data = data[(b-1)/2*self.Ny:(b+1)/2*self.Ny,(b-1)/2*self.Nx:(b+1)/2*self.Nx]
        
        if returnAll:
        	selfAll=padded_template.copy()
        	selfAll.data=data[:selfAll.Ny,:selfAll.Nx]
        	return self,selfAll
        else:
        	if fourier:
        		return self,kMap
        	return self

def padded_fill_from_Cell(padded_mask,ell,Cell,lMin=a.lMin):
	""" Function to fill + pad a fourier map with a Gaussian random field implementation of an input Cell_BB spectrum.
	
	Input: 
	padded_mask - mask for generating fourier transforms
	ell - range of ell for Cell
	Cell - spectrum (NB: not normalised to l(l+1)C_ell/2pi)
	lMin - minimum ell pixel to fill
	padding_ratio - ratio of padded-to-unpadded map width
	
	Output: fourier map with GRF from isotropic realisation of Cell spectrum
	"""
	# First compute a realisation of the isotropic Cell spectrum
	from .RandomField import real_fill_from_Cell
	real_map=real_fill_from_Cell(padded_mask.copy(),ell,Cell,bufferFactor=1,lMin=lMin,log=True,fourier=False)
	
   	# Now zero pad this to desired ratio
   	#from .PaddedPower import zero_padding
   	#pad_real=zero_padding(real_map,padding_ratio)
   	
   	real_map.data*=padded_mask.data # multiply by mask
   	windowFactor=np.mean(padded_mask.data**2.) # <W^2> factor 
   	
   	# Now compute fourier spectrum
   	fourier_real=fftTools.fftFromLiteMap(real_map)
   	fourier_real.kMap/=np.sqrt(windowFactor) # account for window factor
   	
   	return fourier_real
	

def fill_from_Cell(powerMap,ell,Cell,lMin=a.lMin,fourier=False,power=True):
	""" Function to fill a power map with a Gaussian random field implementation of an input Cell_BB spectrum.
	
	This is adapted from flipper.LiteMap.FillWithGaussianRandomField
	Input: realMap (for template)
	powerMap (for output)
	ell - rnage of ell for Cell
	Cell - spectrum (NB: not normalised to l(l+1)C_ell/2pi)
	lMin - minimum ell pixel to fill
	fourier - whether to return Fourier space map
	power - whether to return power-space map
	
	Output: PowerMap with GRF
	"""
	from fftTools import fftFromLiteMap
	
	# Map templates
	pow_out=powerMap.copy()
	
	#ft=fftFromLiteMap(real_temp) # for fft frequencies
	Ny=pow_out.Ny
	Nx=pow_out.Nx
	
	#from scipy.interpolate import splrep,splev # for fitting
	
	realPart=np.zeros([Ny,Nx])
	imgPart=np.zeros([Ny,Nx])
	
	# Compute fourier freq. and mod L map
	ly=pow_out.ly#np.fft.fftfreq(Ny,d=real_temp.pixScaleY)*2.*np.pi
	lx=pow_out.lx#np.fft.fftfreq(Nx,d=real_temp.pixScaleX)*2.*np.pi
	
	modLMap=np.zeros([Ny,Nx])
	iy,ix=np.mgrid[0:Ny,0:Nx]
	modLMap[iy,ix]=np.sqrt(ly[iy]**2.+lx[ix]**2.)
	
	# Fit input Cell to spline
	from scipy.interpolate import UnivariateSpline
	spl=UnivariateSpline(ell,np.log(Cell),k=5) # use a quintic spline here
	ll=np.ravel(modLMap)
	kk=np.ones_like(ll)*1e-20
	
	# Apply filtering
	idhi=np.where(ll>max(ell))
	idlo=np.where(ll<lMin)
	idgood=np.where((ll<max(ell))&(ll>lMin))
	kk[idgood]=np.exp(spl(ll[idgood]))
	kk[idhi]=min(kk[idgood]) # set unwanted values to small value
	kk[idlo]=min(kk[idgood])
	
	area = Nx*Ny*pow_out.pixScaleX*pow_out.pixScaleY # map area
	p = np.reshape(kk,[Ny,Nx])/ area * (Nx*Ny)**2.
	
	# Compute real + imag parts
	realPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	imgPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	# NB: 0.5 factor needed to get correct output Cell
	
	if power:
		# Compute power
		pMap=(realPart**2.+imgPart**2.)*area/(Nx*Ny)**2.
	
		if fourier:
			fMap=realPart+1.0j*imgPart
			return pMap,fMap
		else:
			return pMap
	else:
		if fourier:
			fMap=realPart+1.0j*imgPart
			return fMap
	
def fill_from_model(powerMap,model,lMin=a.lMin,lMax=a.lMax,fourier=False,power=True):
	""" Function to fill a power map with a Gaussian random field implementation of an input Cell_BB spectrum.
	
	This is adapted from flipper.LiteMap.FillWithGaussianRandomField
	Input: realMap (for template)
	powerMap (for output)
	model - C_l model function for input ell
	lMin/lMax - Range of ell to fill pixels (must be inside Cell_lens limits)
	fourier - Boolean, whether to return Fourier-space map
	power - Boolean, whether to return power-space map
	
	Output: [PowerMap/fourier map] with GRF from model
	"""
	from fftTools import fftFromLiteMap
	
	# Map templates
	pow_out=powerMap.copy()
	
	Ny=pow_out.Ny
	Nx=pow_out.Nx
	
	realPart=np.zeros([Ny,Nx])
	imgPart=np.zeros([Ny,Nx])
	
	# Compute fourier freq. and mod L map
	ly=pow_out.ly#np.fft.fftfreq(Ny,d=pow_out.pixScaleY)*2.*np.pi
	lx=pow_out.lx#np.fft.fftfreq(Nx,d=pow_out.pixScaleX)*2.*np.pi
	
	modLMap=np.zeros([Ny,Nx])
	iy,ix=np.mgrid[0:Ny,0:Nx]
	modLMap[iy,ix]=np.sqrt(ly[iy]**2.+lx[ix]**2.)
	
	ll=np.ravel(modLMap)
	kk=np.zeros_like(ll)
	
	# Only fill for correct pixels
	id_low=np.where(ll<lMin)
	id_hi=np.where(ll>lMax)
	id_good=np.where((ll>lMin)&(ll<lMax))
	
	kk[id_good]=model(ll[id_good]) # add model value
	kk[id_low]=min(kk[id_good]) # unneeded pixels
	kk[id_hi]=min(kk[id_good]) # (filled to allow for log-plotting)
	
	area = Nx*Ny*pow_out.pixScaleX*pow_out.pixScaleY # map area
	p = np.reshape(kk,[Ny,Nx])/ area * (Nx*Ny)**2.
	
	# Compute real + imag parts
	realPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	imgPart = np.sqrt(p)*np.random.randn(Ny,Nx)*np.sqrt(0.5)
	# NB: 0.5 factor needed to get correct output Cell
	
	if power:
		# Compute power
		pMap=(realPart**2.+imgPart**2.)*area/(Nx*Ny)**2.
	
		if fourier:
			fMap=realPart+1.0j*imgPart
			return pMap,fMap
		else:
			return pMap
	else:
		if fourier:
			fMap=realPart+1.0j*imgPart
			return fMap
	
