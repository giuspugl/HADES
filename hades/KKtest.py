from .params import BICEP
a=BICEP()

import numpy as np

def angle_estimator(map_id,map_size=a.map_size,l_step=a.l_step,lMin=a.lMin,lMax=a.lMax,slope=None,map=None,returnSlope=False):
    """ Use Kamionkowski & Kovetz 2014 test to find polarisation strength and anisotropy angle via fs, fc parameters.
    Inputs: map_id
    map_size = 3,5,10 -> width of map in degrees
    l_step (how large to bin data)
    lMin,lMax= fitting range of l
    slope,map -> best fit slope and map if already estimated (recomputed if None)
    returnSlope = whether to return slope in addition

    NB: This works for arbitrary map 
    
    Outputs:
    p_str,p_ang -> polarisation strength and angle
    """
    from hades import PowerMap
    
    if map==None and slope==None:
        # Create rescaled map and slope
        print 'OLD'
        print 'Creating new map'
        map,slope,_=PowerMap.RescaledPlot(map_id,map_size=map_size,l_min=lMin,l_max=lMax,l_step=l_step,rescale=True,show=False,showFit=False,returnMap=True)
  
    if map!=None and slope==None:
        # Compute slope of map only
        print 'OLD'
        print 'recompute slope'
        slope,_,_=PowerMap.MapSlope(map,lMin,lMax,l_step)

        
    # print('Using map size of %s' %(map_size))


    # Initialise variables
    A_num,A_den=0,0
    Afc_num,Afc_den=0,0
    Afs_num,Afs_den=0,0

    # Construct estimators over all pixels
    for i in range(map.Ny):
        for j in range(map.Nx):
            l=map.modLMap[i,j]
            ang=map.thetaMap[i,j]*np.pi/180
            if (l<lMax) and (l>lMin):
                fiducial = l**(-slope)
                A_num+=map.powerMap[i,j]/fiducial
                A_den+=1 # for A estimator
                Afc_num+=map.powerMap[i,j]/fiducial*np.cos(4*ang)
                Afc_den+=(np.cos(4*ang)**2) # for A*fc estimator
                Afs_num+=map.powerMap[i,j]/fiducial*np.sin(4*ang)
                Afs_den+=(np.sin(4*ang)**2)
    A=A_num/A_den # A estimator
    Afs=Afs_num/Afs_den
    Afc=Afc_num/Afc_den
    fs=Afs / A # fs estimation
    fc=Afc / A

    # Compute polarisation strength + angle
    p_str = np.sqrt(fs**2+fc**2) # Strength
    p_ang=0.25*np.arctan(fs/fc)*180/np.pi # in degrees (-22.5 to 22.5)

    if returnSlope:
        return p_str,p_ang,fs,fc,Afs,Afc,A,slope
    else:
        return p_str,p_ang,A,fs,fc,Afs,Afc

if __name__=='__main__':
    """ Batch process using all available cores to compute the angle and polariasation strengths of a map using the angle_estimator code above. 
    DEPRACATED"""

    # Import modules
    import tqdm
    import sys
    import multiprocessing as mp

    # Default parameters
    nmin=0
    nmax=315
    cores=8

    # Parameters if input from the command line
    if len(sys.argv)>=2:
        nmin = int(sys.argv[1])
    if len(sys.argv)>=3:
        nmax = int(sys.argv[2])
    if len(sys.argv)==4:
        cores = int(sys.argv[3])

    # Start the multiprocessing
    p = mp.Pool(processes=cores)
    file_ids = np.arange(nmin,nmax+1)

    # Multiprocess tasks and display progress with tqdm
    outputs = list(tqdm.tqdm(p.imap(angle_estimator,file_ids),total=len(file_ids)))

    # Reconstruct arrays
    ang = np.zeros(len(outputs))
    pol= np.zeros(len(outputs))
    for l,line in enumerate(outputs):
        pol[l]=line[0]
        ang[l]=line[1]

    # Save outputs
    np.savez('/data/ohep2/angleMaps5Deg.npz',pol=pol,ang=ang)
    
    
def reconstructor(map_size=10):
    """ Plot the angle and power on a spherical grid for all patches
    Input: map_size = 3,5,10
    DEPRACATED 
    """
  
    # Read in data
    data=np.load('/data/ohep2/angleMaps'+str(map_size)+'Deg.npz')
    pol=data['pol']
    ang=data['ang']

    # Also read in coordinates of patches
    if map_size==10:
        coords=np.loadtxt('/data/ohep2/sims/simdata/fvsmapRAsDecs.txt')
  	# Create lists of patch centres
    	ras=np.zeros(len(pol))
    	decs=np.zeros(len(pol))
    	for i in range(len(pol)):
        	ras[i]=coords[i][1]
        	decs[i]=coords[i][2]
    elif map_size==5 or map_size==3:
    	import pickle
    	ras=pickle.load(open('/data/ohep2/sims/'+str(map_size)+'deg/fvsmapRas.pkl','rb'))
    	decs=pickle.load(open('/data/ohep2/sims/'+str(map_size)+'deg/fvsmapDecs.pkl','rb'))
    	ras=ras[:len(pol)]
    	decs=decs[:len(pol)] # remove any extras
    else:
       	return Exception('Invalid map_size')


    # Now plot on grid
    import astropy.coordinates as coords
    import astropy.units as u
    import matplotlib.pyplot as plt

    ra_deg=coords.Angle(ras*u.degree) # convert to correct format
    ra_deg=ra_deg.wrap_at(180*u.degree)
    dec_deg=coords.Angle(decs*u.degree)

    #Plot polarisation strength
    fig=plt.figure()
    fig.add_subplot(111,projection='mollweide')
    plt.scatter(ra_deg.radian,dec_deg.radian,c=pol,marker='o',s=30)
    plt.colorbar()
    plt.title('Polarisation Strength Map')
    plt.savefig('polarisationMap'+str(map_size)+'Deg.png')
    plt.clf()

    #Plot angle
    fig=plt.figure()
    fig.add_subplot(111,projection='mollweide')
    plt.scatter(ra_deg.radian,dec_deg.radian,marker='o',c=ang,s=50)
    plt.colorbar()
    plt.title('Polarisation Angle Map')
    plt.savefig('angleMap'+str(map_size)+'Deg.png')
    plt.clf()

    return None    
   
def zero_estimator(map,map_id,lMin=a.lMin,lMax=a.lMax,FWHM=a.FWHM,noise_power=a.noise_power,\
	slope=a.slope,factor=None,rot=0.,KKmethod=a.KKmethod,\
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors):
	""" Use KK 14 estimators to find polarisation strength and anisotropy angle via fs,fc parameters.
	This uses the noise model in hades.NoisePower.noise_model. 
	A is computed recursively, since S/N ratio depends on it (only weak dependence) as below
    
  	Inputs: map (in power-space)
 	map_size = width of map in degrees
    	lMin,lMax= fitting range of l
    	slope -> fiducial C_l map slope
    	rot -> optional angle for pre-rotation of power-space map in degrees.
    	KKmethod -> Boolean for which S/N ratio to apply (true is Kamionkowski/Kovetz version, false is Sherwin version)
    	delensing_fraction -> efficiency of delensing (0.1-> 90% removed)
    	factor -> expected amplitude factor (to speed convergence)
    	useTensors -> Boolean whether to include r = 0.1 tensor modes from IGWs
    	
    	Outputs:
    	A,fs,fc, Afs, Afc from estimators. NB these are corrected for any map pre-rotation.
    	"""
    	raise Exception('These are not suitable for high-noise')
    	from hades.NoisePower import noise_model
    	
    	if KKmethod:
    		# Compute sky fraction
    		area = (a.map_size*np.pi/180.)**2. # map area in steradians
    		f_sky = area/(4.*np.pi)
    	
    	# Import lensing function (made from CAMB)
    	from hades.NoisePower import lensed_Cl,r_Cl
    	Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction)
    	Cl_r_func=r_Cl()
    	
    	if factor==None:
	    	# First compute the monopole amplitude A for SNR (denoted Afactor)
	    	# This is done recursively since SN depends on A
	    	# (A only weakly depends on the SN factor Afactor, so convergence is quick)
	    	N=0
	    	Afactor=1e-12
	    	while N<10: # max. number of iterations
	    	
	    		if True:
				goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax))
				lMap=map.modLMap.ravel()[goodPix]
				powerMap=map.powerMap.ravel()[goodPix]
				
				noiseClmap=noise_model(lMap,FWHM=FWHM,noise_power=noise_power)
				lensClmap=Cl_lens_func(lMap)
				if useTensors:
					rClmap=Cl_r_func(lMap)
				else:
					rClmap=np.zeros_like(lensClmap)
				fiducialMap=lMap**(-slope)
				SN=(Afactor*fiducialMap)/(Afactor*fiducialMap+noiseClmap+lensClmap+rClmap)
				
				Anum=np.sum(powerMap/fiducialMap*(SN**2.))
				Aden=np.sum(SN**2.)
				lastFactor=Afactor
				Afactor=Anum/Aden
				
	    		else: # depracated
	    			raise Exception('DEPRACATED METHOD in KKtest')
		    		A_num=0.
		    		A_den=0.
		    	    	# Construct estimators over all pixels in range
				for i in range(map.Ny):
					for j in range(map.Nx):
						l=map.modLMap[i,j]
						if l<lMax and l>lMin:
							fiducial=l**(-slope)
							noise_Cl=noise_model(l,FWHM=FWHM,noise_power=noise_power)
							lens_Cl=Cl_lens_func(l)
							
							if KKmethod:
								sigma_l_sq = 2.*((noise_Cl+lens_Cl)**2.)/ (f_sky*(2.*l+1.)) 
		        	                		SN = fiducial/np.sqrt(sigma_l_sq)
		        	                	else:
		        	                		SN=(Afactor*fiducial)/(Afactor*fiducial+noise_Cl+lens_Cl)    
		        	       			# A estimator
		        	        		A_num+=map.powerMap[i,j]/fiducial*(SN**2.)
		        	        		A_den+=SN**2.
		        	lastFactor=Afactor # previous A factor
		        	Afactor=A_num/A_den # new A factor
	        	
	    		if np.abs(Afactor-lastFactor)/Afactor<0.01:
	    			break # approximate convergence reached 
	    		N+=1
	        
	        if N==10:
	        	print 'Map %s did not converge with slope: %.3f Afactor: %.3e, last factor: %.3e' %(map_id,slope,Afactor,lastFactor)
	        finalFactor=Afactor
	else:
		finalFactor=factor # just use from input      
    	
    	# NB: we recompute A so that all best estimators use the same SNR
    	
	# Construct estimators for Afs, Afc over all pixels in range
	if True:
		goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax))
		angMap=(map.thetaMap.ravel()[goodPix]+rot*np.ones_like(map.thetaMap.ravel()[goodPix]))*np.pi/180.
		cosMap=np.cos(4*angMap)
		sinMap=np.sin(4*angMap)
		lMap=map.modLMap.ravel()[goodPix]
		powerMap=map.powerMap.ravel()[goodPix]
		
		noiseClmap=noise_model(lMap,FWHM=FWHM,noise_power=noise_power)
		lensClmap=Cl_lens_func(lMap)
		if useTensors:
			rClmap=Cl_r_func(lMap)
		else:
			rClmap=np.zeros_like(lensClmap)
		fiducialMap=lMap**(-slope)
		SN=(finalFactor*fiducialMap)/(finalFactor*fiducialMap+noiseClmap+lensClmap+rClmap)
		
		Anum=np.sum(powerMap/fiducialMap*(SN**2.))#  biased monopole power
		Aden=np.sum(SN**2.)
		Afcnum=np.sum(powerMap*cosMap/fiducialMap*(SN**2.)) # biased hexadecapole power coeff
		Afcden=np.sum((SN*cosMap)**2.)
		Afsnum=np.sum(powerMap*sinMap/fiducialMap*(SN**2.)) # biased hexadecapole power coeff
		Afsden=np.sum((SN*sinMap)**2.)
		A=Anum/Aden 
		Afc=Afcnum/Afcden 
		Afs=Afsnum/Afsden
		fs=Afs/A # NB: no bias correction so don't use yet
		fc=Afc/A 
	
	if False: # depracated method
		# Initialise other variables
		raise Exception('Depracated KKtest method')
	    	A_num, A_den=0.,0.
	    	Afs_num,Afs_den=0.,0.
	    	Afc_num,Afc_den=0.,0.
	    	
		for i in range(map.Ny):
			for j in range(map.Nx):
				l=map.modLMap[i,j]
				if l<lMax and l>lMin:
					ang=(map.thetaMap[i,j]+rot)*np.pi/180. # in radians with optional rotation
					fiducial=l**(-slope)
					noise_Cl=noise_model(l,FWHM=FWHM,noise_power=noise_power)
					
					lens_Cl=Cl_lens_func(l)
					
					if KKmethod:
						sigma_l_sq = 2.*((noise_Cl+lens_Cl)**2.)/ (f_sky*(2.*l+1.)) 
	        	                	SN = fiducial/np.sqrt(sigma_l_sq)
	        	                else:
	        	                	SN=(finalFactor*fiducial)/(finalFactor*fiducial+noise_Cl+lens_Cl)    
					
	               			# A estimator
	                		A_num+=map.powerMap[i,j]/fiducial*(SN**2.)
	                		A_den+=SN**2.
	                		# Afc estimator
	                		Afc_num+=map.powerMap[i,j]/fiducial*(np.cos(4.*ang)*SN**2.)
	                		Afc_den+=(SN*np.cos(4.*ang))**2.
	                		# Afs estimator
	                		Afs_num+=map.powerMap[i,j]/fiducial*(np.sin(4.*ang)*SN**2.)
	                		Afs_den+=(SN*np.sin(4.*ang))**2.
	                
	    	A=A_num/A_den-A_bias
	    	Afs=Afs_num/Afs_den
	    	Afc=Afc_num/Afc_den
	    	fs=Afs/A
	    	fc=Afc/A
    	
    	# Now correct for rotation:
    	rot_rad=rot*np.pi/180. # in radians
    	fs_corr=fs*np.cos(rot_rad*4.)-fc*np.sin(rot_rad*4.)
    	fc_corr=fs*np.sin(rot_rad*4.)+fc*np.cos(rot_rad*4.)
    	Afs_corr=Afs*np.cos(rot_rad*4.)-Afc*np.sin(rot_rad*4.)
    	Afc_corr=Afs*np.sin(rot_rad*4.)+Afc*np.cos(rot_rad*4.)
    	
    	return A,fs_corr,fc_corr,Afs_corr,Afc_corr
    
def A_estimator(map,map_id,lMin=a.lMin,lMax=a.lMax,FWHM=a.FWHM,noise_power=a.noise_power,\
	slope=a.slope,factor=None,rot=0.,KKmethod=a.KKmethod,\
	delensing_fraction=a.delensing_fraction,useTensors=a.useTensors):
	""" Use KK 14 estimators to find only monopole amplitude A. This is done for noise+lensing sims (i.e. NO FOREGROUNDS)
	We use previous A_estimate for the SNR here for consistency
    
  	Inputs: map (in power-space)
 	map_size = width of map in degrees
    	lMin,lMax= fitting range of l
    	slope -> fiducial C_l map slope
    	rot -> optional angle for pre-rotation of power-space map in degrees.
    	KKmethod -> Boolean for which S/N ratio to apply (true is Kamionkowski/Kovetz version, false is Sherwin version)
    	delensing_fraction -> efficiency of delensing (0.1-> 90% removed)
    	factor -> expected amplitude factor (to speed convergence)
    	useTensors -> whether to include r = 0.1 tensor modes
    	
    	Outputs:
    	A from estimators.
    	"""
    	from hades.NoisePower import noise_model
    	
    	if KKmethod:
    		# Compute sky fraction
    		area = (a.map_size*np.pi/180.)**2. # map area in steradians
    		f_sky = area/(4.*np.pi)
    		return 'Depracated method'
    	
    	# Import lensing function (made from CAMB)
    	from hades.NoisePower import lensed_Cl,r_Cl
    	Cl_lens_func=lensed_Cl(delensing_fraction=delensing_fraction)
    	Cl_r_func=r_Cl()
    	
    	# Compute KK estimated quantities
    	goodPix=np.where((map.modLMap.ravel()>lMin)&(map.modLMap.ravel()<lMax))
	lMap=map.modLMap.ravel()[goodPix]
	powerMap=map.powerMap.ravel()[goodPix]
	
	noiseClmap=noise_model(lMap,FWHM=FWHM,noise_power=noise_power)
	lensClmap=Cl_lens_func(lMap)
	if useTensors:
		rClmap=Cl_r_func(lMap)
	else:
		rClmap=np.zeros_like(lensClmap)
	fiducialMap=lMap**(-slope)
	SN=(factor*fiducialMap)/(factor*fiducialMap+noiseClmap+lensClmap+rClmap)
	
	Anum=np.sum(powerMap/fiducialMap*(SN**2.))
	Aden=np.sum(SN**2.)
	Afactor=Anum/Aden
	
	return Afactor

def noisy_estimator(map,map_size=a.map_size,lMin=a.lMin,lMax=a.lMax,\
FWHM=a.FWHM,noise_power=a.noise_power,slope=a.slope,noNoise=False):
    """ Use Kamionkowski & Kovetz 2014 test to find polarisation strength and anisotropy angle via fs, fc parameters.
    This uses the noise model in hades.NoisePower.noise_model
    
    Inputs: map (in power-space)
    map_size = width of map in degrees
    lMin,lMax= fitting range of l
    slope -> fiducial C_l map slope
    noNoise -> if True, sets S/N ratio to 1 for all pixels
    
    Outputs:
    NEW:    A,fs,fc from estimators
    OLD:    p_str,p_ang,monopole amplitude, fs,fc,Afs,Afc from estimators
 
    """
    # Compute sky fraction
    area = (a.map_size*np.pi/180.)**2. # map area in steradians
    f_sky = area/(4.*np.pi)
    
    # Initialise variables
    A_num,A_den=0.,0.
    Afc_num,Afc_den=0.,0.
    Afs_num,Afs_den=0.,0.
    
    # Import noise model
    from hades.NoisePower import noise_model

    # Construct estimators over all pixels
    #print Afc_num, A_den
    
    for i in range(map.Ny):
        for j in range(map.Nx):
            l=map.modLMap[i,j]
            if (l<lMax) and (l>lMin):
            	ang=(map.thetaMap[i,j]+11.25)*np.pi/180. # in radians
                fiducial = l**(-slope)#  fiducial C_l^f 
                noise_Cl=noise_model(l,FWHM=FWHM,noise_power=noise_power)
                if noNoise:
                	SN = 1. # signal-to-noise ratio
                else:
                	sigma_l_sq = 2.*(noise_Cl**2.)/ (f_sky*(2.*l+1.)) 
                        SN = fiducial/np.sqrt(sigma_l_sq)
                
                # A estimator
                A_num+=map.powerMap[i,j]/fiducial*(SN**2.)
                A_den+=SN**2.
                # Afc estimator
                Afc_num+=map.powerMap[i,j]/fiducial*(np.cos(4.*ang)*SN**2.)
                Afc_den+=(SN*np.cos(4.*ang))**2.
                # Afs estimator
                Afs_num+=map.powerMap[i,j]/fiducial*(np.sin(4.*ang)*SN**2.)
                Afs_den+=(SN*np.sin(4.*ang))**2.
                
    # for stability
    A=A_num/A_den
    Afs=Afs_num/Afs_den
    Afc=Afc_num/Afc_den
    fs=Afs/A
    fc=Afc/A
    
    if False:
       	A=np.exp(np.log(A_num)-np.log(A_den)) # A estimator
    	Afc=np.sign(Afc_num)*np.exp(np.log(np.abs(Afc_num))-np.log(Afc_den))
    	Afs=np.sign(Afs_num)*np.exp(np.log(np.abs(Afs_num))-np.log(Afs_den))
    	fs=np.sign(Afs)*np.exp(np.log(np.abs(Afs))-np.log(A)) # fs estimation
    	fc=np.sign(Afc)*np.exp(np.log(np.abs(Afc))-np.log(A)) # fc estimation
    
    #print Afc_num, Afc_den, SN
    
    if False:
    	# Compute polarisation strength + angle
    	p_str = np.sqrt(fs**2.+fc**2.) # Strength
    	p_ang=0.25*np.arctan(fs/fc)*180./np.pi # in degrees (-22.5 to 22.5)
  
	return p_str,p_ang,A,fs,fc,Afs,Afc

    else:
    	return A,fs,fc,Afs,Afc


def rotation_estimator(map,map_size=a.map_size,lMin=a.lMin,lMax=a.lMax,\
FWHM=a.FWHM,noise_power=a.noise_power,slope=a.slope,noNoise=False):
    """ Use Kamionkowski & Kovetz 2014 test to find polarisation strength and anisotropy angle via fs, fc parameters.
    This uses the noise model in hades.NoisePower.noise_model. Rotate over 22.5 degrees and average to avoid pixellation errors.
    
    Inputs: map (in power-space)
    map_size = width of map in degrees
    lMin,lMax= fitting range of l
    slope -> fiducial C_l map slope
    noNoise -> if True, sets S/N ratio to 1 for all pixels
    
    Outputs:
    fraction and angle from estimators
    
    """
    # Compute sky fraction
    area = (a.map_size*np.pi/180.)**2. # map area in steradians
    f_sky = area/(4.*np.pi)
    
    
    
    # Import noise model
    from hades.NoisePower import noise_model
    
    A_all,frac_all,ang_all=[],[],[] # containing fractions/angles from each theta

    for theta in a.rotation_angles:
    	# Initialise variables
    	A_num,A_den=0.,0.
    	Afc_num,Afc_den=0.,0.
    	Afs_num,Afs_den=0.,0.
    
    	for i in range(map.Ny):
    	    for j in range(map.Nx):
    	        l=map.modLMap[i,j]
    	        if (l<lMax) and (l>lMin):
    	        	ang=(map.thetaMap[i,j]+theta)*np.pi/180. # in radians
    	         	fiducial = l**(-slope)#  fiducial C_l^f 
    	           	noise_Cl=noise_model(l,FWHM=FWHM,noise_power=noise_power)
    	           	if noNoise:
    	            		SN = 1. # signal-to-noise ratio
       		 	else:
                		sigma_l_sq = 2.*(noise_Cl**2.)/ (f_sky*(2.*l+1.)) 
                        	SN = fiducial/np.sqrt(sigma_l_sq)
                
                	# A estimator
                	A_num+=map.powerMap[i,j]/fiducial*(SN**2.)
                	A_den+=SN**2.
                	# Afc estimator
                	Afc_num+=map.powerMap[i,j]/fiducial*(np.cos(4.*ang)*SN**2.)
                	Afc_den+=(SN*np.cos(4.*ang))**2.
                	# Afs estimator
                	Afs_num+=map.powerMap[i,j]/fiducial*(np.sin(4.*ang)*SN**2.)
                	Afs_den+=(SN*np.sin(4.*ang))**2.
                
    	A=A_num/A_den
    	Afs=Afs_num/Afs_den
    	Afc=Afc_num/Afc_den
    	fs=Afs/A
    	fc=Afc/A
    	A_all.append(A)
    	frac_all.append(np.sqrt(fs**2.+fc**2.))
    	ang_all.append(0.25*np.arctan(fs/fc)*180./np.pi-theta)
    
    return np.mean(A_all), np.mean(frac_all), np.mean(ang_all)
