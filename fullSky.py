import numpy as np
from hades.params import BICEP
a=BICEP()

def full_sky_reconstructor(root_dir=a.root_dir,sep=a.sep,map_size=a.map_size,FWHM=a.FWHM,\
	noise_power=a.noise_power,delensing_fraction=a.delensing_fraction,\
	freq=a.freq,maxDec=85.,folder='DebiasedBatchDataFull'):
	"""This reconstructs the full data from the multiprocessed full sky runs. 
	This is optimized for the full sky, due to indexing.
	
	IN: maxDec filters data to only return data below certain Dec.
	Default is 85 degrees
	"""
	
	# First load in coordinates
	import pickle
	map_dir=root_dir+'%sdeg%s/' %(map_size,sep)
	full_ras=pickle.load(open(map_dir+'fvsmapRas.pkl','rb'))
	full_decs=pickle.load(open(map_dir+'fvsmapDecs.pkl','rb'))
	goodMap=pickle.load(open(map_dir+'fvsgoodMap.pkl','rb'))
	ras=[full_ras[i] for i in range(len(full_ras)) if goodMap[i]!=False]
	decs=[full_decs[i] for i in range(len(full_decs)) if goodMap[i]!=False]
	
	index = 0 # for counting
	jndex=0
	import os
	
	A,eps,eps_err,eps_mean,angle,angle_err,H2,H2_mean,H2_err,Afs,Afc=[],[],[],[],[],[],[],[],[],[],[]
	ras_all,decs_all=[],[]
	# Now iterate over files
	while True:
		if index%1000==0:
			print index
		if jndex>5000:
			break
		fileName=root_dir+folder+'/f%s_ms%s_s%s_fw%s_np%s_d%s/%s.npy' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction,index)
    		if not os.path.exists(fileName):
        		print 'no dat at %s' %index
        		#print fileName
        		jndex+=1
        		index+=1
        		#print 'location %s' %fileName
        		continue
        	# Read in data
        	data=np.load(fileName)
        	if data.all()==1:
        		print 'edge data'
        		index+=1
        		continue
        	ras_all.append(ras[index])
        	decs_all.append(decs[index])
		A.append(data[0][0])
		eps.append(data[5][0])
		eps_err.append(data[5][2])
		eps_mean.append(data[5][1]) 
		angle.append(data[6][0])
		angle_err.append(data[6][2])
		H2.append(data[9][0])
		H2_mean.append(data[9][1])
		H2_err.append(data[9][2])
		Afs.append(data[3][0])
		Afc.append(data[4][0])
		index+=1
	ras=ras_all
	decs=decs_all	
	# Remove any extras
	print len(A),len(ras)
	ras=ras[:len(A)]
	decs=decs[:len(A)]
	
	# Filter out bad RAs
	if maxDec!=None:
		ids=np.where(np.abs(np.array(decs))<maxDec)
		ras=np.array(ras)[ids]
		decs=np.array(decs)[ids]
		A=np.array(A)[ids]
		eps=np.array(eps)[ids]
		eps_mean=np.array(eps_mean)[ids]
		eps_err=np.array(eps_err)[ids]
		angle=np.array(angle)[ids]
		angle_err=np.array(angle_err)[ids]
		H2=np.array(H2)[ids]
		H2_mean=np.array(H2_mean)[ids]
		H2_err=np.array(H2_err)[ids]
	# Save output
	np.savez(root_dir+'all_dat%s_f%s_ms%s_s%s_fw%s_np%s_d%s.npz' %(maxDec,freq,map_size,sep,FWHM,noise_power,delensing_fraction),ras=ras,decs=decs,\
		A=A,eps=eps,eps_mean=eps_mean,eps_err=eps_err,\
		angle=angle,angle_err=angle_err,H2=H2,H2_mean=H2_mean,H2_err=H2_err,Afs=Afs,Afc=Afc)
	
	print 'complete'
	
def correlations(NSIDE=16,LMAX=64,maxDec=85.,map_size=a.map_size,sep=a.sep,freq=a.freq,noise_power=a.noise_power,FWHM=a.FWHM,delensing_fraction=a.delensing_fraction):
	"""Compute correlation function between monopole amplitude and debiased epsilon parameter across the whole (masked) sky, using data created using the above full_sky_reconstructor function.
	
	Inputs:
	NSIDE - Healpix NSIDE value (no. HEALPix pixels should exceed no. data points)
	LMAX - maxm ell to compute power spectra for
	maxDec - maxm declination used here (to avoid projection errors)
	
	Outputs: 
	Correlations + power spectra saved in FullSky/Correlations/ subdirectory
	"""
	import healpy as hp
	import numpy as np
	
	# Load in masks
	allMasks=hp.read_map('HFI_Mask_GalPlane-apo2_2048_R2.00.fits',field=[0,1,2,3,4])
	f_sky=[20,40,60,70,80]
	
	# Load in data
	dat = np.load('FullSky/all_dat%s_f%s_ms%s_s%s_fw%s_np%s_d%s.npz' %(maxDec,freq,map_size,sep,FWHM,noise_power,delensing_fraction))
	ra=dat['ras']
	dec=dat['decs']
	eps=dat['eps']
	A=dat['A']
	eps_mean=dat['eps_mean']
	ang=dat['angle']
	ang_err=dat['angle_err']
	eps_err=dat['eps_err']
	
	# Compute fractional sky area removed by removal of top and bottom pixels
	frac_cap = (1.-np.sin(maxDec*np.pi/180.))
	
	print 'Using %d HEALPix pixels for %d data pixels' %(hp.nside2npix(NSIDE),len(A))
	allCorrs=[]
	allErrs=[]
	for maskIndex in range(len(allMasks)):
		print 'Creating map %d' %(maskIndex+1)
		# Compute map mask
		mask=hp.pixelfunc.ud_grade(allMasks[maskIndex],NSIDE)
		
		# Compute effective sky fraction
		skyFrac=f_sky[maskIndex]/100.-frac_cap # since removing top and bottom
		
		eps_debiased=[eps[i]-eps_mean[i] for i in range(len(eps))]
		
		# Compute template maps for projection onto HEALPix grid
		template=np.zeros(hp.nside2npix(NSIDE))
		weights,tempSA,tempSEpsDeb,tempSang=[np.zeros_like(template) for _ in range(4)]
		
		# Now project onto grid
		for i in range(len(ra)):
			pix=hp.ang2pix(NSIDE,ra[i],dec[i],lonlat=True)
			tempSA[pix]=(A[i]+weights[pix]*tempSA[pix])/(weights[pix]+1.) # compute mean if more than one point
        		tempSEpsDeb[pix]=(eps_debiased[i]+weights[pix]*tempSEpsDeb[pix])/(weights[pix]+1.) # compute mean if more than one point
        		tempSang[pix]=(ang[i]+weights[pix]*tempSang[pix])/(weights[pix]+1.)
        		weights[pix]+=1.
        		
        	# Interpolate from nearby pixels for any empty pixels
		if True:
			count=0
			for pix in range(len(template)):
				if tempSA[pix]==0.:
					count+=1
					nearPix=hp.get_all_neighbours(NSIDE,pix)
					nearPix=[nP for nP in nearPix if tempSA[nP]!=0.]
					if len(nearPix)!=0:
						tempSA[pix]=np.mean(tempSA[nearPix])
			            		tempSEpsDeb[pix]=np.mean(tempSEpsDeb[nearPix])
			            		tempSang[pix]=np.mean(tempSang[nearPix])

			print 'zero pixel value - interpolated %s times' %count
		
		# Compute mask for polar regions
		decM=np.zeros_like(mask)
		DECS=[hp.pix2ang(NSIDE,i,lonlat=True)[1] for i in range(hp.nside2npix(NSIDE))]
		ids=np.where(np.abs(DECS)<maxDec)
		decM[ids]=1.
		smoothM=hp.smoothing(decM,fwhm=1.*np.pi/180.) # smooth with 1 degree Gaussian kernel
		mask*=smoothM
		
		maskFactor=np.sum(mask**2.)/len(mask)	
			
		tempSA*=mask
		tempSEpsDeb*=mask
		tempSang*=mask
		
		# Now compute power spectra
		CellA=hp.anafast(tempSA,lmax=LMAX)
		CellEpsDeb=hp.anafast(tempSEpsDeb,lmax=LMAX)
		CellAng=hp.anafast(tempSang,lmax=LMAX)
		CellCross=hp.anafast(tempSA,tempSEpsDeb,lmax=LMAX) # cross spectrum
		
		ells = np.arange(len(CellA))
		factor=ells*(ells+np.ones_like(ells))/(2.*np.pi)/maskFactor#/skyFrac**2. # to correct for incomplete sky
		DellAng=CellAng*factor # in D_l convention
		DellA=CellA*factor
		DellEpsDeb=CellEpsDeb*factor
		
		# Compute correlation
		CorrAEpsDeb=CellCross/np.sqrt(CellA*CellEpsDeb)
		
		# Compute error bars from cosmic variance
		err_bar=np.sqrt(2./((2.*np.array(ells)+1.)*float(skyFrac)))
		
		# Create plots
		outDir='FullSky/PowerSpectra_f%s_ms%s_s%s_fw%s_np%s_d%s/' %(freq,map_size,sep,FWHM,noise_power,delensing_fraction)
		import os
		if not os.path.exists(outDir):
			os.makedirs(outDir)
		
		import matplotlib.pyplot as plt
		from hades.fullSky import binner
		bw=3 # bin width
		plt.figure()
		ELL,DAT,ERR=binner(ells[2:],DellAng[2:],err_bar[2:]*DellAng[2:],BIN=bw)
		plt.errorbar(ELL,DAT,yerr=ERR,fmt='x')
		#plt.errorbar(ells,DellAng,yerr=err_bar*DellAng,fmt='x')
		plt.ylabel(r'$D_l^{\alpha\alpha}$')
		plt.xlabel(r'$l$')
		plt.title('Angle Power Spectrum for GAL%s mask' %f_sky[maskIndex])
		plt.savefig(outDir+'Angle_%s' %f_sky[maskIndex])
		plt.close()
		
		plt.figure()
		ELL,DAT,ERR=binner(ells[2:],DellA[2:],err_bar[2:]*DellA[2:],BIN=bw)
		plt.errorbar(ELL,DAT,yerr=ERR,fmt='x')
		#plt.errorbar(ells,DellA,yerr=err_bar*DellA,fmt='x')
		plt.ylabel(r'$D_l^{AA}$')
		plt.xlabel(r'$l$')
		plt.title('Amplitude Power Spectrum for GAL%s mask' %f_sky[maskIndex])
		plt.savefig(outDir+'Amplitude_%s' %f_sky[maskIndex])
		plt.close()
		
		plt.figure()
		ELL,DAT,ERR=binner(ells[2:],DellEpsDeb[2:],err_bar[2:]*DellEpsDeb[2:],BIN=bw)
		plt.errorbar(ELL,DAT,yerr=ERR,fmt='x')
		#plt.errorbar(ells[2:],DellEpsDeb,yerr=err_bar*DellEpsDeb,fmt='x')
		plt.ylabel(r'$D_l^{\epsilon\epsilon}$')
		plt.xlabel(r'$l$')
		plt.title('Debiased Epsilon Power Spectra for GAL%s mask' %f_sky[maskIndex])
		plt.savefig(outDir+'Epsilon_%s' %f_sky[maskIndex])
		plt.close()
		
		plt.figure()
		ELL,DAT,ERR=binner(ells[2:],CorrAEpsDeb[2:],np.ones_like(ells[2:]),BIN=bw)
		
		plt.errorbar(ELL,DAT,yerr=ERR,fmt='x')
		allCorrs.append(DAT)
		allErrs.append(ERR)
		#plt.scatter(ells[2:],CorrAEpsDeb[2:],marker='x')
		plt.ylabel(r'$C_l^{A\epsilon}/\sqrt{C_l^{AA}C_l^{\epsilon\epsilon}}$')
		plt.xlabel(r'$l$')
		plt.title('Amplitude - Debiased Epsilon Correlation for GAL%s mask' %f_sky[maskIndex])
		plt.savefig(outDir+'Correlation_%s.png' %f_sky[maskIndex],bbox_inches='tight')
		plt.close()
	
	plt.figure()
	for i in range(len(allCorrs)):
		plt.plot(ELL,allCorrs[i],label='GAL%s' %f_sky[i],alpha=0.5)
	plt.xlabel(r'$l$')
	plt.legend()
	plt.ylabel(r'$C_l^{A\epsilon}/\sqrt{C_l^{AA}C_l^{\epsilon\epsilon}}$')
	plt.title('Amplitude - Debiased Epsilon Correlation (10% errors)' %f_sky[maskIndex])
	plt.savefig(outDir+'AllCorrelations.png',bbox_inches='tight')
	plt.close()	
		
	print 'Process complete'
	
# Binning function
def binner(ells,dat,err,BIN=5):
    """This function bins a given set of data and errors.
    Errors are calculated using the unbiased weighted mean errors.
    
    Inputs: ells [xcoords]
    dat [ycoords]
    err [absolute errors]
    BIN [no. points per bin]
    
    Outputs:
    ellCen [central bin x]
    binDat [binned y value]
    binErr [error estimate]
    """
    ellCen=[]
    BinDat=[]
    BinErr=[]
    
    errs,dats,ells_temp=[],[],[]
    for i in range(len(dat)):
    	errs.append(err[i])
    	dats.append(dat[i])
    	ells_temp.append(ells[i])
    	if (i+1)%BIN==0:
    		mu_temp=0.
    		weights_temp=0.
    		for j in range(len(errs)):
    			mu_temp+=dats[j]/errs[j]**2.
    			weights_temp+=1./errs[j]**2.
    		mean=mu_temp/weights_temp
    		
    		sig_temp=0.
    		V2_temp=0.
    		for j in range(len(errs)):
    			sig_temp+=(dats[j]-mean)**2./errs[j]**2.
    			V2_temp+=1./errs[j]**4.
    		err_mean=np.sqrt(sig_temp/(weights_temp-V2_temp/weights_temp))
    		ellCen.append(np.mean(ells_temp))
    		BinDat.append(mean)
    		BinErr.append(err_mean)
    		# reset variables
    		errs,dats,ells_temp=[],[],[]    

    return ellCen,BinDat,BinErr
		
