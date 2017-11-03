from flipper import fftTools

def radialPower(map_id,map_type='B',plot=True,show=False,Return=False,xlog=True,ylog=False):
    """ Function to create radial power spectra for 2D power spectra.

    NB: THIS DOES NOT TAKE INTO ACCOUNT WINDO FUNCTION
    
    Inputs: map_id number (0-315)
    map_type (B,E,T)
    plot -> whether to plot output 1D spectrum
    return -> return plot data
    log -> whether x-axis is in log

    Output: plotted fig (if plot=True)
    binned_l,mean_power,stdev_power,no_pixels_power for each annulus (if Return=True)
    """
    import numpy as np
    from hades.PowerMap import openPower
    
    # Binned Power Directory
    work_dir = '/data/ohep2/Spectra/'
    
    # Load map
    pMap = openPower(map_id,map_type=map_type)

    # Open binning file
    # (this can be created with bin_list function)
    l_dat = np.genfromtxt(work_dir+'binning.dat')

    # Now extract spectra in radial bands:
    l_bin = np.zeros(len(l_dat)) # median l in bin
    mean_pow = np.zeros(len(l_dat)) # mean power in annulus
    std_pow = np.zeros(len(l_dat)) # stdev of power in annulus
    pix_pow = np.zeros(len(l_dat)) # number of pixels sampled
    
    for i,line in enumerate(l_dat):
        l_bin[i] = line[2]
        mean_pow[i],std_pow[i],pix_pow[i]=pMap.meanPowerInAnnulus(line[0],line[1])

    # Plot output
    if plot:
        fftTools.plotBinnedPower(l_bin,mean_pow,minL=min(l_bin),maxL=max(l_bin),\
                                     yrange=None,title=str(map_type)+'-mode Radially Binned Power Spectrum',\
                                     pngFile=work_dir+'radialPower_'+str(map_type)+str(map_id)+'.png',\
                                     show=show,returnPlot=Return,errorBars=std_pow,\
                                     ylog=ylog,xlog=xlog)


        
        #import matplotlib.pyplot as plt
        #plt.errorbar(l_bin,mean_pow,yerr=std_pow,xerr=0.5*(l_bin[1]-l_bin[0]))
        #plt.title(str(map_type)+'-mode Radially Binned Power Spectrum')
        #plt.xlabel('l')
        #plt.ylabel('Mean Power')
        #if log:
        #    plt.xscale('log') # logarithmic scaling
        #if show:
        #    plt.show()
        #plt.savefig(work_dir+'radialPower_'+str(map_type)+str(map_id)+'.png')
        #plt.clf()

    #if Return:
    #    return l_bin,mean_pow,std_pow,pix_pow
    #else:
    #    return None
    
def bin_list(l_min,l_max,l_step):
    """Create the binning.dat file describing the different radial bins used here.
    Inputs: min,max l and bin width.
    Output: Spectra/binning.dat file"""

    binning_dir = '/data/ohep2/Spectra/' # directory of binning file
    
    import os
    import numpy as np

    # Create directory
    if not os.path.exists(binning_dir):
        os.makedirs(binning_dir)

    # All l values    
    l_all = np.arange(l_min,l_max+l_step,l_step)

    l_dat = []
    for i in range(len(l_all)-1):
        l_dat.append([l_all[i],l_all[i+1],0.5*(l_all[i]+l_all[i+1])])

    # Save output
    np.savetxt(binning_dir+'binning.dat',l_dat,fmt='%d',newline='\n')
   
    return None
      
