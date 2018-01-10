from hades.params import BICEP
import warnings # catch rogue depracation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 	
import numpy as np
import matplotlib
import matplotlib.pylab as pyl
a=BICEP()

def BICEP_border(map_size,sep):
    """ Compute RA/dec coordinates of edge of BICEP region.
    Output is RA,dec in degrees. 
     """
    import pickle
    
    map_dir='BICEP2/'+'%sdeg%s/' %(map_size,sep)
    import os
    if not os.path.exists(map_dir+'fvsmapRas.pkl'):
    	print 'no border'
    	return None
    
    full_ras=pickle.load(open(map_dir+'fvsmapRas.pkl','rb'))
    full_decs=pickle.load(open(map_dir+'fvsmapDecs.pkl','rb'))
    goodMap=pickle.load(open(map_dir+'fvsgoodMap.pkl','rb'))
    ra=[full_ras[i] for i in range(len(full_ras)) if goodMap[i]!=False]
    dec=[full_decs[i] for i in range(len(full_decs)) if goodMap[i]!=False]
    for i in range(len(ra)):
        if ra[i]>180.:
            ra[i]-=360.
    
    DECs=np.unique(dec)
    N=2*len(DECs)
    edge_ra,edge_dec=[np.zeros(N+1) for _ in range(2)]
    for j,D in enumerate(DECs):
        RAs=[ra[i] for i in range(len(ra)) if dec[i]==D]
        edge_ra[j]=min(RAs)
        edge_ra[N-1-j]=max(RAs)
        edge_dec[j]=D
        edge_dec[N-1-j]=D
    edge_ra[N]=edge_ra[0]
    edge_dec[N]=edge_dec[0]
    
    return edge_ra,edge_dec
    
def skyMap(dat,ra,dec,cbar_label=None,cmap='jet',decLims=[-90,-40,5],raLims=[-180,30,10],minMax=None,border=None,outFile=None,show=False):
    """Plot in Albers Equal Area Projection conic grid using the skymapper package.
    Inputs:
    data -> values to plot
    ra,dec -> coordinates
    cbar_label -> label for colorbar
    cmap -> colormap (e.g. 'jet)
    minMax -> [min,max] for colorbar
    decLims/raLims -> give min/max/step for RA and dec lines
    border -> [ra,dec] of border of BICEP region
    outFile -> location to save plot in
    show -> boolean whether to show plot
    
    Output: image saved in outFile directory.
    """
    # load projection and helper functions
    import skymapper as skm
    matplotlib.rcParams.update({'font.size': 22,'text.usetex': True,'font.family': 'serif'})

    # setup figure
    fig = pyl.figure(figsize=(15,15))
    ax = fig.add_subplot(111, aspect='equal')

    # setup map: define AEA map optimal for given RA/Dec
    proj = skm.createConicMap(ax, ra.value, dec.value, proj_class=skm.AlbersEqualAreaProjection)
    # add lines and labels for meridians/parallels (separation 5 deg)
    meridians = np.arange(decLims[0],decLims[1],decLims[2])
    parallels = np.arange(raLims[0],raLims[1],raLims[2])
    skm.setMeridianPatches(ax, proj, meridians, linestyle=':', lw=0.5, zorder=2)
    skm.setParallelPatches(ax, proj, parallels, linestyle=':', lw=0.5, zorder=2)
    skm.setMeridianLabels(ax, proj, meridians, loc="left", fmt=skm.pmDegFormatter)
    skm.setParallelLabels(ax, proj, parallels, loc="top", fmt=skm.degFormatter)

    # convert to map coordinates and plot a marker for each point
    x,y = proj(ra.value, dec.value)
    marker = 's'
    markersize = skm.getMarkerSizeToFill(fig, ax, x, y)
    if minMax==None:
    	vmin,vmax=np.percentile(dat,[0,100])
    else:
    	vmin,vmax=minMax
    sc = ax.scatter(x,y, c=dat, edgecolors='None', marker=marker, s=markersize, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, zorder=1)
    
    # add border of BICEP region
    if border !=None:
    	xB,yB=proj(border[0],border[1]) # read in border coordinates
    	bor = ax.plot(xB,yB,c='k',lw=2,ls='--') # plot border
    
    # add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.0)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(cbar_label,fontsize=20)
 
    # show (and save) ...
    fig.tight_layout()
    if outFile!=None:
        fig.savefig(outFile,bbox_inches='tight')
    if show:
        fig.show()
    else:
        fig.clf()
	fig.clear()

