import healpy as hp
import numpy as np
from ispice import ispice

# load unapodized masks
all_masks=hp.read_map('/data/ohep2/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=[0,1,2,3,4])

FFP8=False
dirName='HexPowSpectra'

if not FFP8:
    root_dir='/data/ohep2/FullSky/'
else:
    root_dir='/data/ohep2/FFP8FullSky/'
outDir=root_dir+dirName+'/'
outDirGood=root_dir+'GoodSpectra/'

maxDec=85. # 85.
sep=1
f_sky=[20,40,60,70,80]
freq=150
map_size=3
FWHM=1.5
noise_power=1.
delensing_fraction=0.1
NSIDE=64

# Load in data
dat = np.load(root_dir+'all_dat%s_f%s_ms%s_s%s_fw%s_np%s_d%s.npz' %(maxDec,freq,map_size,sep,FWHM,noise_power,delensing_fraction))
ra=dat['ras']
dec=dat['decs']
eps=dat['eps']
eps_mean=dat['eps_mean']
H2=dat['H2']
ang=dat['angle']
A=dat['A']
ang_err=dat['angle_err']
eps_err=dat['eps_err']
dat.close()

# Compute fractional sky area removed by removal of top and bottom pixels
frac_cap = (1.-np.sin(maxDec*np.pi/180.))
ClA,ClH2,ClAng,Corrs=[],[],[],[]

print 'Using %d HEALPix pixels for %d data pixels' %(hp.nside2npix(NSIDE),len(A))

# Create template map
template=np.zeros(hp.nside2npix(NSIDE))
weights,tempSA,tempSH2,tempSang=[np.zeros_like(template) for _ in range(4)]

for i in range(len(ra)):
	if H2[i]>=0.:
	        pix=hp.ang2pix(NSIDE,ra[i],dec[i],lonlat=True)
	        tempSA[pix]=(A[i]+weights[pix]*tempSA[pix])/(weights[pix]+1.) # compute mean if more than one point
	        tempSH2[pix]=(H2[i]+weights[pix]*tempSH2[pix])/(weights[pix]+1.) # compute mean if more than one point
	        tempSang[pix]=(ang[i]+weights[pix]*tempSang[pix])/(weights[pix]+1.)
	        weights[pix]+=1.

#Interpolate from nearby pixels for any empty pixels
count=0
for pix in range(len(template)):
    if tempSA[pix]==0.:
        count+=1
        nearPix=hp.get_all_neighbours(NSIDE,pix)
        nearPix=[nP for nP in nearPix if tempSA[nP]!=0.]
        if len(nearPix)!=0:
            tempSA[pix]=np.mean(tempSA[nearPix])
            tempSH2[pix]=np.mean(tempSH2[nearPix])
            tempSang[pix]=np.mean(tempSang[nearPix])

tempSA*=1e12 # in uK^2 units
tempSH2*=1e24 # in uK^2 units

print 'zero pixel value - interpolated %s times' %count

# Save as FITS files
import os
FitsDir=root_dir+'SpiceFITS/'
if not os.path.exists(FitsDir):
    os.makedirs(FitsDir)
hp.write_map(FitsDir+'ang.fits',tempSang)
hp.write_map(FitsDir+'A.fits',tempSA)
hp.write_map(FitsDir+'H2.fits',tempSH2)

def CorByn(Corr,BIN):
    """Binning function (Bin size BIN) for correlations. Error is just standard deviation of inputs"""
    lOut,cOut,cErr=[],[],[]
    templ,tempC=[],[]
    for i in range(len(Corr)):
        templ.append(i)
        tempC.append(Corr[i])
        if (i+1)%BIN==0 or i==len(Corr)-1:
            lOut.append(np.mean(templ))
            modtemp=[tempC[i] for i in range(len(tempC)) if tempC[i]>0.]
            cOut.append(np.mean(modtemp))
            cErr.append(np.std(modtemp))
            templ,tempC=[],[]
    return lOut,cOut,cErr

for mask_index in range(len(all_masks)):
    # Create mask
    mask=hp.pixelfunc.ud_grade(all_masks[mask_index],NSIDE)

    # Now mask polar regions
    DECS=[hp.pix2ang(NSIDE,i,lonlat=True)[1] for i in range(hp.nside2npix(NSIDE))]
    ids=np.where(np.abs(DECS)>maxDec) # in forbidden regions
    mask[ids]=0.

    id2=np.where((mask!=0.)&(mask!=1.))
    mask[id2]=0.
    
    
    mask=hp.smoothing(mask,fwhm=3.*np.pi/180.)
    mask[mask<1e-2]=0. # hard cut off

    hardMask=mask.copy()
    hardMask[hardMask!=0.]=1.
    
    # Save mask files
    hp.write_map(FitsDir+'Mask%s.fits' %f_sky[mask_index],mask)
    hp.write_map(FitsDir+'MaskHard%s.fits' %f_sky[mask_index],hardMask)

    # Now create C_ls using PolSpice
    from ispice import ispice
    from bin_llcl import bin_llcl
    import time

    infiles=['ang','A','H2']
    fullNames=['Angle','Monopole Amplitude','Hexadecapole']
    texNames=[r'$\mathcal{D}_l^{\alpha}$',r'$\mathcal{D}_l^{A}$ $[\mu{}K^2]$' ,r'$\mathcal{D}_l^{\mathcal{H}^2}$ $[\mu{}K^4]$']
    GAL=f_sky[mask_index]
    for i,infile in enumerate(infiles):
        outCl=FitsDir+infile+'%s.cl' %GAL

        # First remove any existing file
        if os.path.exists(outCl):
            os.remove(outCl)
        # Now create file
        if GAL<50:
            sig_apodize=np.sqrt(GAL/50.)*150. # apodization size ~ sky size
        else:
            sig_apodize=180.
            
        ispice(FitsDir+infile+'.fits',outCl,maskfile1=FitsDir+'MaskHard%s.fits' %GAL,\
               weightfile1=FitsDir+'Mask%s.fits' %GAL,apodizesigma=sig_apodize,\
               thetamax=sig_apodize)

        # Now check for file to be created
        start_time=time.time()
        while not os.path.exists(outCl):
            time.sleep(1)

            if time.time()-start_time>10:
                raise Exception('Failed to compute')

      # Bin Cl files
        Cl=hp.read_cl(outCl)
        X,Y,_,E=bin_llcl(Cl[4:],5,flatten=True)
        idx=np.where(Y>0.)
        X=X[idx]
        Y=Y[idx]
        E=E[idx]
        if infile=='A':
            ClA.append([X,Y,E,Cl])
        elif infile=='H2':
            ClH2.append([X,Y,E,Cl])
        else:
            ClAng.append([X,Y,E,Cl])
            
        # E is errors assuming full-sky coverage
        sky_frac = np.sum(mask)/len(mask) # this is sky fraction used
        E/=np.sqrt(sky_frac)

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # Now plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif',size=18)
        plt.errorbar(X,Y,yerr=E,fmt='x')
        plt.xlabel(r'$l$')
        plt.xlim([0,150])
        if infile=='ang':
        	plt.ylim([0,115])
        plt.ylabel(texNames[i])
        #plt.title(r'%s Power Spectrum using GAL%s Mask' %(fullNames[i],GAL),fontsize=14)
        plt.savefig(outDir+infile+'GAL%s.png' %GAL,bbox_inches='tight')
        if GAL==80:
        	plt.savefig(outDirGood+infile+'GAL80.png',bbox_inches='tight')
        plt.close()



    # Now plot cross spectrum
    outCl=FitsDir+'CrossSpectrum%s.cl' %GAL

    # First remove any existing file
    if os.path.exists(outCl):
        os.remove(outCl)
    # Now create file
    ispice(FitsDir+'A.fits',outCl,mapfile2=FitsDir+'H2.fits',apodizesigma=180.,thetamax=180.,\
           weightfile1=FitsDir+'Mask%s.fits'%GAL,\
           weightfile2=FitsDir+'Mask%s.fits'%GAL,\
           maskfile1=FitsDir+'MaskHard%s.fits' %GAL,\
           maskfile2=FitsDir+'MaskHard%s.fits' %GAL,)

    # Now check for file to be created
    start_time=time.time()
    while not os.path.exists(outCl):
        time.sleep(1)

        if time.time()-start_time>30:
            raise Exception('Failed to compute')

    # Now compute correlation
    Cl=hp.read_cl(outCl)
    Corr=Cl/np.sqrt(ClA[-1][3]*ClH2[-1][3])
    X,Y,E=CorByn(Corr[4:],5)
    #idx=np.where(X>0.)
    
    #Xp=X[idx] # avoid negatives
    #Yp=Y[idx]

    Corrs.append([X,Y,E])
    # Now plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)
    plt.errorbar(X,Y,yerr=E,fmt='x')
    plt.xlabel(r'$l$')
    plt.xlim([0,150])
    #plt.yscale('log')
    plt.ylabel(r'Cross Correlation, $\rho_{A\mathcal{H}^2}$')
    #plt.title(r'$A-\mathcal{H}^2$ Correlation Spectrum for GAL%s Mask' %(GAL),fontsize=14)
    plt.savefig(outDir+'CorrelationGAL%s.png' %GAL,bbox_inches='tight')
    if GAL==80:
    	plt.savefig(outDirGood+'Correlation.png',bbox_inches='tight')
    
    plt.close()

plt.figure()
# Now plot combined plots
# Angle:
for i in range(1,len(ClAng)):
    plt.errorbar(ClAng[i][0],ClAng[i][1],yerr=ClAng[i][2],fmt='x',label='GAL%s' %f_sky[i])
    plt.ylabel(texNames[0])
    plt.xlabel(r'$l$')
    plt.legend()
    plt.title('Angle Power Spectra')
    plt.savefig(outDir+'AllAlpha.png',bbox_inches='tight')
plt.close()

plt.figure()    
# Cross correlation:
for i in range(1,len(Corrs)):
    plt.errorbar(Corrs[i][0],Corrs[i][1],yerr=Corrs[i][2],label='GAL%s' %f_sky[i],fmt='x')
    plt.ylabel(r'Cross Correlation')
    plt.xlabel(r'$l$')
    plt.legend()
    plt.title(r'$A-\mathcal{H}^2$ Correlation Spectrum')
    #plt.yscale('log')
    plt.savefig(outDir+'AllCorrelations.png',bbox_inches='tight')
plt.close()

print 'Process Complete'

 
