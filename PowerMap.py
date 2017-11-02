from flipper import *
import flipperPol as fp

def createMap(map_id,warnings=False,plotPNG=True):
    """ Create the B-,E-,T- space 2D power maps from input simulation maps. Simulation maps are those of Vansyngel+16 provided and reduced by Alex van Engelen. This uses the flipperPol hybrid scheme to minimise E-B leakage in B-maps.
    
    Input: map_id - number of map - integer from 00001-00315
    warnings - whether to ignore overwrite warnings
    plotPNG - whether to save PNG files

    Output: T,B,E maps are saved in the ProcMaps/ directory and pngs are in the ProcMaps/png directory 
    """
    import os
    
    # Map directories
    filepath = '/data/ohep2/sims/simdata/' # This houses all the simulation data
    outdir = '/data/ohep2/ProcMaps/'
    pngdir = "/data/ohep2/ProcMaps/png/"
    Q_path = filepath+'fvsmapQ_'+str(map_id).zfill(5)+'.fits'
    U_path = filepath+'fvsmapU_'+str(map_id).zfill(5)+'.fits'
    T_path = filepath+'fvsmapT_'+str(map_id).zfill(5)+'.fits'
    mask_path = filepath+'fvsmapMaskSmoothed_'+str(map_id).zfill(5)+'.fits'
    
    # Create output directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Read in maps
    Tmap = liteMap.liteMapFromFits(T_path)
    Qmap = liteMap.liteMapFromFits(Q_path)
    Umap = liteMap.liteMapFromFits(U_path)
    maskMap = liteMap.liteMapFromFits(mask_path) # this is the smoothed map

    ## Convert T,Q,U maps to T,E,B maps

    # First compute modL and angL maps (common for all files)
    modLMap,angLMap = fp.fftPol.makeEllandAngCoordinate(Tmap)

    # Now compute the FFT (hybrid) maps using the mask map to minimise E->B leakage (no significant B->E since E>>B)
    fftTmap,fftEmap,fftBmap = fp.fftPol.TQUtoPureTEB(Tmap,Qmap,Umap,maskMap,modLMap,angLMap,method='hybrid')

    # Now compute the power maps from this
    pTT,pTE,pET,pTB,PBT,pEE,pEB,pBE,pBB = fp.fftPol.fourierTEBtoPowerTEB(\
        fftTmap,fftEmap,fftBmap,fftTmap,fftEmap,fftBmap)
    # (second argument allows for cross maps

    # Save these files in correct directory

    if not warnings:
        import warnings
        warnings.filterwarnings("ignore") # ignore overwrite warnings

    pTT.writeFits(outdir+'powerT'+str(map_id)+'.fits',overWrite=True)
    pBB.writeFits(outdir+'powerB'+str(map_id)+'.fits',overWrite=True)
    pEE.writeFits(outdir+'powerE'+str(map_id)+'.fits',overWrite=True)

    # Plot and save as pngs
    if plotPNG:
        if not os.path.exists(pngdir):
            os.mkdir(pngdir) # Make directory

        pTT.plot(log=True,show=False,pngFile=pngdir+"powerT"+str(map_id)+".png")
        pEE.plot(log=True,show=False,pngFile=pngdir+"powerE"+str(map_id)+".png")
        pBB.plot(log=True,show=False,pngFile=pngdir+"powerB"+str(map_id)+".png")


    return None

def openPower(map_id,map_type='B'):
    """ Function to read in Power FITS file saved in createMap process.
    Input: map_ids: 0-315 denoting map number
    maptype - E,B or T map to open
    """
    import pyfits

    mapfile = 'ProcMaps/power'+str(map_type)+str(map_id)+'.fits'
    oldFile = 'sims/simdata/fvsmapT_'+str(map_id).zfill(5)+'.fits'
    powMap = fftTools.power2D() # generate template

    hdulist = pyfits.open(mapfile) # read in map
    header = hdulist[0].header

    powMap.powerMap = hdulist[0].data.copy() # add power data

    # Read in old T map - to recover L and ang maps
    oldMap = liteMap.liteMapFromFits(oldFile)
    modL,angL = fp.fftPol.makeEllandAngCoordinate(oldMap)
    easyFT = fftTools.fftFromLiteMap(oldMap)

    # Read in these quantities from simple FT 
    powMap.ix = easyFT.ix
    powMap.iy = easyFT.iy
    powMap.lx = easyFT.lx
    powMap.ly = easyFT.ly
    powMap.pixScaleX = easyFT.pixScaleX
    powMap.pixScaleY = easyFT.pixScaleY
    powMap.Nx = easyFT.Nx
    powMap.Ny = easyFT.Ny
    powMap.modLMap = modL
    powMap.thetaMap = angL

    return powMap

if __name__=='__main__':
     """ Batch process to use all available cores to compute the 2D power spectra
    This uses the makePower routine.

    Inputs are min and max file numbers"""

     import tqdm
     import sys
     import numpy as np
     import multiprocessing as mp

     # Default parameters
     nmin = 0
     nmax = 315
     cores = 7

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

     # Display progress bar with tqdm
     r = list(tqdm.tqdm(p.imap(createMap,file_ids),total=len(file_ids)))


