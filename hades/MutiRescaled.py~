if __name__=='__main__':
    """ Batch process using all available cores to compute the 2D rescaled power spectrum of a given maps using the PowerMap.RescaledPlot routine.
    Input arguments are nmin nmax cores defining range of map_id and number of cores to use.
    Default arguments of RescaledPlot() are used here
    """
    
    # Import files
    import tqdm
    import sys
    import numpy as np
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

     # Display progress bar with tqdm
     r = list(tqdm.tqdm(p.imap(createMap,file_ids),total=len(file_ids)))

     
