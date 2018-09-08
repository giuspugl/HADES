


from scipy import *
import scipy

from flipper import *
from flipperPol import *

import pylab
import pdb
import copy

import sys
sys.path.append('/scratch2/r/rbond/engelen/lensRecon/tools/')

import statsTools
import mapTools
# import powerTools
from scipy.interpolate import splrep, splev
import healpy
import pyfits

def allpowersTEB(mapT, mapE, mapB, binFile = None):

    fT = fftTools.fftFromLiteMap(mapT)
    fE = fftTools.fftFromLiteMap(mapE)
    fB = fftTools.fftFromLiteMap(mapB)

    #for now no cross spectra in this routine, can be addded if necessary.

    TT_power,TE_power,ET_power,TB_power,BT_power,EE_power,EB_power,BE_power,BB_power=fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT.copy(),fE.copy(), fB.copy())

    l_Lo,l_Up,lbin,cl_TT,j,k = statsTools.aveBinInAnnuli(TT_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_TE,j,k = statsTools.aveBinInAnnuli(TE_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_TB,j,k = statsTools.aveBinInAnnuli(TB_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_EE,j,k = statsTools.aveBinInAnnuli(EE_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_EB,j,k = statsTools.aveBinInAnnuli(EB_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_BB,j,k = statsTools.aveBinInAnnuli(BB_power, binFile)#, './binningTest')

    #this will be what we leave in after the unblinding......

    print 'got here e'

    
    output = {'lbin':lbin, \
           'cl_TT':cl_TT, \
           'cl_TE':cl_TE, \
           'cl_EE':cl_EE, \
           'cl_EB':cl_EB, \
           'cl_BB':cl_BB}


    return output


def quickPower(inmap, inmap2 = None, window = None, useFlipperRoutine = False, applySlepianTaper = False, nanToNum = True, binFile = None) :

    if binFile == None:
        binFileToUse = 'binningTest'
    else :
        binFileToUse = binFile

    temp = inmap.copy()

    if inmap2 == None:
        temp2 = inmap.copy()
    else:
        temp2 = inmap2.copy()


    if window != None :
        temp.data *= window.data / sqrt(numpy.mean(window.data**2))
        temp2.data *= window.data / sqrt(numpy.mean(window.data**2))
#        print "correcting for window"

    if nanToNum:
        temp.data = numpy.nan_to_num(temp.data)
        temp2.data = numpy.nan_to_num(temp2.data)

    power2D = fftTools.powerFromLiteMap(temp, liteMap2 = temp2, applySlepianTaper = applySlepianTaper)

    if useFlipperRoutine:
        llower,lupper,lbin,clbin_0,clbin_N,binweight = power2D.binInAnnuli(binFileToUse, nearestIntegerBinning=True)
    else:

        llower,lupper,lbin,clbin_0,clbin_N,binweight = statsTools.aveBinInAnnuli(power2D, binFileToUse)

    output = {'lbin':lbin, 'clbin' : clbin_0, 'dlbin': lbin*(lbin+1) * clbin_0 / 2/numpy.pi}

    # return (lbin, clbin_0)
    return output



def binnedPowerFrom2dF(twodf, data = None):

    localF = twodf.copy()
    if data != None:
        localF.data[:] = data

    power2d = fftTools.powerFromFFT(localF)
        
    llower,lupper,lbin,clbin_0,clbin_N,binweight = power2d.binInAnnuli('binningTest', nearestIntegerBinning=True)
    output = {'lbin':lbin, 'clbin' : clbin_0, 'dlbin': lbin*(lbin+1) * clbin_0 / 2/numpy.pi}

    return(output)


def allmaps(beforeroot, afterroot) : 

    T_map = liteMap.liteMapFromFits(beforeroot + 'T' + afterroot)
    Q_map = liteMap.liteMapFromFits(beforeroot + 'Q' + afterroot)
    U_map = liteMap.liteMapFromFits(beforeroot + 'U' + afterroot)

    output =   {'mapT':T_map, \
                    'mapQ':Q_map, \
                    'mapU':U_map}
                

def allpowersFitsFile(beforeroot, afterroot, window = None, beamFile = None, flipQ = False, TOnly = True, method = 'pure', useI = True):


    if TOnly == False:

        T_map = liteMap.liteMapFromFits(beforeroot + ('I' if useI else 'T') + afterroot)
        Q_map = liteMap.liteMapFromFits(beforeroot + 'Q' + afterroot)
        U_map = liteMap.liteMapFromFits(beforeroot + 'U' + afterroot)

        output = allpowers(T_map, Q_map, U_map, window = window, beamFile = beamFile, flipQ = flipQ, method = method)
    else:
        T_map = liteMap.liteMapFromFits(beforeroot + ('I' if useI else 'T') + afterroot)

        output = quickPower(T_map, applySlepianTaper = True)
        

    return output





def allpowers(T_map, Q_map, U_map, T_map2 = None, Q_map2 = None, U_map2 = None , \
                  window = None, beamFile = None, flipQ = False, \
                  method = 'pure', binFile = None) : 


    T_map_local = T_map.copy()
    Q_map_local = Q_map.copy()
    U_map_local = U_map.copy()

    T_map_local.data = numpy.nan_to_num(T_map_local.data)
    Q_map_local.data = numpy.nan_to_num(Q_map_local.data)
    U_map_local.data = numpy.nan_to_num(U_map_local.data)

    if T_map2 != None:
        T_map2_local = T_map2.copy()
        Q_map2_local = Q_map2.copy()
        U_map2_local = U_map2.copy()
    else:
        T_map2_local = T_map.copy()
        Q_map2_local = Q_map.copy()
        U_map2_local = U_map.copy()

    T_map2_local.data = numpy.nan_to_num(T_map2_local.data)
    Q_map2_local.data = numpy.nan_to_num(Q_map2_local.data)
    U_map2_local.data = numpy.nan_to_num(U_map2_local.data)

    print 'got here'

    if flipQ:
        print 'note: flipping sign of Q map'
        Q_map_local.data *= -1.

    if window != None :
        T_map_local.data *= window.data / sqrt(numpy.mean(window.data**2))
        Q_map_local.data *= 1 / sqrt(numpy.mean(window.data**2))
        U_map_local.data *= 1 / sqrt(numpy.mean(window.data**2))

        T_map2_local.data *= window.data / sqrt(numpy.mean(window.data**2))
        Q_map2_local.data *= 1 / sqrt(numpy.mean(window.data**2))
        U_map2_local.data *= 1 / sqrt(numpy.mean(window.data**2))

    else:
        window = T_map.copy()
        window.data[:] = 1.


    modLMap,angLMap=fftPol.makeEllandAngCoordinate(T_map_local,bufferFactor=1)
    print 'got here a'

    fT, fE, fB= fftPol.TQUtoPureTEB(T_map_local,Q_map_local,U_map_local,window,modLMap,angLMap,method=method)
    print 'got here b'

    fT2, fE2, fB2= fftPol.TQUtoPureTEB(T_map2_local,Q_map2_local,U_map2_local,window,modLMap,angLMap,method=method)

    if beamFile != None:    
        beamdata = numpy.loadtxt(beamFile)
    
        ell, f_ell = numpy.transpose(beamdata)[0:2, :]

        t = mapTools.makeTemplate( ell, f_ell, fT)
        fT.kMap /= t
        fE.kMap /= t
        fB.kMap /= t

        fT2.kMap /= t
        fE2.kMap /= t
        fB2.kMap /= t

    # print 'got here'
    # pdb.set_trace()
    print 'got here c'

    TT_power,TE_power,ET_power,TB_power,BT_power,EE_power,EB_power,BE_power,BB_power=fftPol.fourierTEBtoPowerTEB(fT,fE,fB,fT2,fE2,fB2)

    print 'got here d'
    

    l_Lo,l_Up,lbin,cl_TT,j,k = statsTools.aveBinInAnnuli(TT_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_TE,j,k = statsTools.aveBinInAnnuli(TE_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_TB,j,k = statsTools.aveBinInAnnuli(TB_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_EE,j,k = statsTools.aveBinInAnnuli(EE_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_EB,j,k = statsTools.aveBinInAnnuli(EB_power, binFile)#, './binningTest')
    l_Lo,l_Up,lbin,cl_BB,j,k = statsTools.aveBinInAnnuli(BB_power, binFile)#, './binningTest')

    #this will be what we leave in after the unblinding......

    print 'got here e'

    
    output = {'lbin':lbin, \
           'cl_TT':cl_TT, \
           'cl_TE':cl_TE, \
           'cl_EE':cl_EE, \
           'cl_EB':cl_EB, \
           'cl_BB':cl_BB}

    # output = {'lbin':lbin, \
    #               'cl_TT':cl_TT, \
    #               'cl_TE':cl_TE, \
    #               'cl_EE':cl_EE}
                  
    return output





def makeLinearBinFile(deltaEll, lmax, fileName):    

    npts = floor(lmax / deltaEll)

    lefts = 1 + linspace(0, npts * deltaEll , num = npts, endpoint = False)
    rights = lefts + deltaEll - 1

    centers = (lefts + rights) / 2.
    
    savetxt(fileName,transpose([lefts, rights, centers]))



def mapListSum(inlist):

    num = len(inlist)

    output = inlist[0] #placeholder
    output.data[:] = 0
    
    for i in arange(num):
        output.data += inlist[i].data

    return output


def mapListMean(inlist, weights = None):


    num = len(inlist)

    output = inlist[0].copy() #placeholder
    output.data[:] = 0

    weightsTotal = inlist[0].copy() #placeholder
    weightsTotal.data[:] = 0


    for i in arange(num):

        output.data += weights[i].data * inlist[i].data
        weightsTotal.data += weights[i].data

    output.data[:] /= weightsTotal.data

    return output, weightsTotal
        



#this is from 
def onedl(rows):
    return [None] * rows

def twodl(rows, cols):
    a=[]
    for row in xrange(rows): a += [[None]*cols]
    return a


def threedl(rows, cols, depth):
    a=[]
    for row in xrange(rows): 
        a += [[None]*cols]
        for col in xrange(cols): 
            a[row][col] = [None]*depth
    return a


def fourdl(rows, cols, depths1, depths2):
    
    a = [[[[None for x in xrange(depths2)] for x in xrange(depths1)] for x in xrange(cols)] for x in xrange(rows)]

    return a


def statsMultiLabel(arrs, nwanted = None, weights = None, labelArr = None):

    output = {}

    for label in labelArr:

        output[label] = stats(arrs, nwanted, weights, label = label)

    return output

        
        




def stats( arrs, data2subtract = None, nwanted = None,  weights = None, label = None):

    
    
    
    if nwanted == None :
        if label == None:
            nwanted = (arrs[0]).size
        else:
            nwanted = (arrs[0][label]).size

    if weights != None:
        raise Exception, "weighting is not fully implemented yet."



    ncurves = len(arrs)

    if label == None:
        twodarr = numpy.array(arrs)

        if data2subtract != None:
            twodarr -= numpy.array(data2subtract)

    else : #then use some dict element.

        twodarr = numpy.zeros((ncurves, nwanted))

        for i in arange(0,ncurves):
            twodarr[i, :] = (arrs[i])[label]

            if data2subtract != None:
                twodarr[i, :] -= (data2subtract[i])[label]

            
        
    
    average = numpy.average(twodarr, axis = 0, weights = weights)

    #averageArr = outer(numpy.repeat(1, ncurves), average )

    
    
#    average = numpy.average( (twodarr - outer(average)
    stdev = numpy.std(twodarr, axis = 0, ddof = 1) #use the N-1 normalization.
   
    output_cov = numpy.zeros((nwanted, nwanted))

    # for i in xrange(nwanted):
    #     for j in xrange(nwanted):
    #         output_cov[i, j] = numpy.cov(arrs[i], arrs[j])

    output_cov = numpy.cov(twodarr, rowvar = False)
    
    output_corr = output_cov / numpy.outer(stdev, stdev)
    

    # if only_one_output:
    output = {'mean':average, 'stdev' : stdev, 'cov' :  output_cov, 'corr':output_corr}

    
    

    return output




def covar(arrs1, arrs2, nwanted1 = None, nwanted2 = None):
 

    if nwanted1 == None:
        nwanted1 = arrs1[0].size


    if nwanted2 == None:
        nwanted2 = arrs2[0].size

    twodarr1 = numpy.array(arrs1)
    twodarr2 = numpy.array(arrs2)

 
    stdev1 =  numpy.std(twodarr1, axis = 0, ddof = 1)
    stdev2 =  numpy.std(twodarr2, axis = 0, ddof = 1)

    output_cov = numpy.zeros((nwanted1, nwanted2))

    # fopr i in xrange(nwanted1):
    #     for j in xrange(nwanted2):
    #         output_cov[i, j] = numpy.cov(arrs1[i], arrs2[j])
    output_cov = numpy.cov(twodarr1,twodarr2, rowvar = False)


    print output_cov.shape

#    output_corr = output_cov / numpy.outer(stdev1, stdev2)
    output_corr = numpy.corrcoef(twodarr1, twodarr2, rowvar = False)
    
    output = {'cov': output_cov, 'corr'  : output_corr}
    return output


# stolen from
# https://github.com/marcelhaas/python/blob/master/value_locate.py
def value_locate(refx, x):
    """
    VALUE_LOCATE locates the positions of given values within a
    reference array. The reference array need not be regularly
    spaced. This is useful for various searching, sorting and
    interpolation algorithms.

    The reference array should be a monotonically increasing or
    decreasing list of values which partition the real numbers. A
    reference array of NBINS numbers partitions the real number line
    into NBINS+1 regions, like so:


    REF: X[0] X[1] X[2] X[3] X[NBINS-1]
    <----------|-------------|------|---|----...---|--------------->
    INDICES: -1 0 1 2 3 NBINS-1


    VALUE_LOCATE returns which partition each of the VALUES falls
    into, according to the figure above. For example, a value between
    X[1] and X[2] would return a value of 1. Values below X[0] return
    -1, and above X[NBINS-1] return NBINS-1. Thus, besides the value
    of -1, the returned INDICES refer to the nearest reference value
    to the left of the requested value.
    Example:
    >>> refx = [2, 4, 6, 8, 10]
    >>> x = [-1, 1, 2, 3, 5, 5, 5, 8, 12, 30]
    >>> print value_locate(refx, x)
    array([-1, -1, 0, 0, 1, 1, 1, 3, 4, 4])
    This implementation is likely no the most efficient one, as there is
    a loop over all x, which will in practice be long. As long as x is
    shorter than 1e6 or so elements, it should still be fast (~sec).
    """

    print "TODO: check if refx is monotonically increasing."

    refx = numpy.array(refx)
    x = numpy.array(x)
    loc = numpy.zeros(len(x), dtype='int')

    for i in xrange(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else: loc[i] = ind[-1]

    return loc


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

#    pdb.set_trace() 
    newarr = numpy.exp( - (xx - centervals[0])**2 / (2. * width_2**2) -  (yy - centervals[1])**2  / (2. * width**2))

    return newarr
                       



def rotateHealpixGtoC(inmap, workdir = '../data/', mapname = 'myData', actuallyDoCtoG = False):

    if (healpy.get_nside(inmap) != 2048):
        print "uh oh, haven't implemented other map sizes yet"
        stop
        
    if actuallyDoCtoG:
        inLet = 'C'
        outLet = 'G'
    else:
        inLet = 'G'
        outLet = 'C'
    

#write out
    healpy.fitsfunc.write_map(workdir + mapname + '_map.fits', inmap)

    
#set up anafast params
    paramsfilecontents = """
simul_type =            1
infile = """ + workdir + mapname + '_map.fits' + """
infile2 = ''
nlmax =         4096
maskfile = ''
theta_cut_deg =   0.000000000000000E+000
regression =            0
plmfile = ''
outfile = !""" + workdir + mapname  + """_cls.fits
outfile_alms = !""" + workdir + mapname  + """_alms.fits
won =            0
iter_order =            0
    """
    paramfile = open(workdir + mapname + '_anafastParam.txt', 'w')
    paramfile.write(paramsfilecontents)
    paramfile.close()

    os.system('anafast --single ' + workdir + mapname + '_anafastParam.txt')


    alteralmParamsContents = """    
infile_alms = """ + workdir + mapname +  """_alms.fits
fwhm_arcmin_in =   0.000000000000000E+000
# beam_file_in
fwhm_arcmin_out =   0.000000000000000E+000
beam_file_out = ''
coord_in = """ + inLet + """
epoch_in =    2000.00000000000
coord_out = """ + outLet + """
epoch_out =    2000.00000000000
nsmax_out =         2048
nlmax_out =         4096
outfile_alms = !"""  + workdir + mapname + """_rot_alms.fits
""" 
    paramfile = open(workdir + mapname + '_alteralmParam.txt', 'w')
    paramfile.write(alteralmParamsContents)
    paramfile.close()

    os.system('alteralm --single ' + workdir + mapname + '_alteralmParam.txt')

    
    synfastParamsContents = """
simul_type =            1
nsmax = 2048
nlmax =         4096
infile = ''
iseed =           -1
fwhm_arcmin = 0
beam_file = ''
almsfile = """ + workdir + mapname + """_rot_alms.fits
apply_windows =  F
plmfile = ''
outfile = !""" + workdir + mapname + """_map_rotated.fits
outfile_alms = '' """

    paramfile = open(workdir + mapname + '_synfastParam.txt', 'w')
    paramfile.write(synfastParamsContents)
    paramfile.close()



    os.system('synfast --single ' + workdir + mapname + '_synfastParam.txt')


    output = healpy.read_map(workdir + mapname + '_map_rotated.fits')
    
    return output


def rotateHealpixGtoCExp(inmap, workdir = '../data/', mapname = 'myData',nside = 2048, isFileName = False):

#this writing from python was crashing for nside = 8192, so make option for just a file copy    
    if not isFileName:
#write out
        healpy.fitsfunc.write_map(workdir + mapname + '_map.fits', inmap)
    else:
        os.system('cp ' + inmap + ' ' + workdir + mapname + '_map.fits')
    
#set up anafast params
    paramsfilecontents = """
simul_type =            1
infile = """ + workdir + mapname + '_map.fits' + """
infile2 = ''
nlmax =   """ + str(2 * nside) + """
maskfile = ''
theta_cut_deg =   0.000000000000000E+000
regression =            0
plmfile = ''
outfile = !""" + workdir + mapname  + """_cls.fits
outfile_alms = !""" + workdir + mapname  + """_alms.fits
won =            0
iter_order =            0
    """
    paramfile = open(workdir + mapname + '_anafastParam.txt', 'w')
    paramfile.write(paramsfilecontents)
    paramfile.close()

    os.system('anafast --single ' + workdir + mapname + '_anafastParam.txt')


    alteralmParamsContents = """    
infile_alms = """ + workdir + mapname +  """_alms.fits
fwhm_arcmin_in =   0.000000000000000E+000
# beam_file_in
fwhm_arcmin_out =   0.000000000000000E+000
beam_file_out = ''
coord_in = G
epoch_in =    2000.00000000000
coord_out = C
epoch_out =    2000.00000000000
nsmax_out =         """ + str(nside) + """
nlmax_out =         """ + str(2 * nside) + """
outfile_alms = !""" + workdir + mapname + """_rot_alms.fits
"""
    paramfile = open(workdir + mapname + '_alteralmParam.txt', 'w')
    paramfile.write(alteralmParamsContents)
    paramfile.close()

    os.system('alteralm --single ' + workdir + mapname + '_alteralmParam.txt')

    synfastParamsContents = """
simul_type =            1
nsmax = """ + str(nside) + """
nlmax =     """ +     str(2 * nside) + """
infile = ''
iseed =           -1
fwhm_arcmin = 0
beam_file = ''
almsfile = """ + workdir + mapname + """_rot_alms.fits
apply_windows =  F
plmfile = ''
outfile = !""" + workdir + mapname + """_map_rotated.fits
outfile_alms = '' """

    paramfile = open(workdir + mapname + '_synfastParam.txt', 'w')
    paramfile.write(synfastParamsContents)
    paramfile.close()



    os.system('synfast --single ' + workdir + mapname + '_synfastParam.txt')


    output = healpy.read_map(workdir + mapname + '_map_rotated.fits')
    
    return output


def allCrossPowers( inArr, windows):

    nPatches = len(inArr[0])
    nMaps = len(inArr)
    print nPatches, nMaps
    assert len(windows) == nPatches, 'bad input'
        
    output = aveTools.threedl(nMaps, nMaps, nPatches)

    for i in arange(nMaps):
        for j in arange(nMaps):
            for k in arange(nPatches):
                output[i][j][k] = aveTools.quickPower(inArr[i][k], inArr[j][k], window = windows[k])


    return output
    



def plotAllCrossPowers(inClArr, names, patchNames = ['deep1', 'deep5', 'deep6' ], ylimval = None, xlimval = None, figNum = 30, bandpowersToAdd = None, bandpowerColor = None, showFig = True, saveFig = False, saveFigName = None):

    nTreatments = len(names)
    nPatches = len(inClArr[0][0])


    
    figure(figNum)
    clf()



    for k  in arange(nPatches):
        
        subplot(1, nPatches, k + 1) 
        lbin = inClArr[0][0][0]['lbin']

        for i in arange(nTreatments):
            for j in arange(i,nTreatments):

                semilogy(lbin, inClArr[i][j][k]['clbin'], label = names[i] + ' x ' + names[j], linestyle = ('-' if i == j else '--'))

        if bandpowersToAdd != None:
            print 'adding errors'
            errorbar(bandpowersToAdd[0], bandpowersToAdd[1], bandpowersToAdd[2], capsize = 0, color = 'b', marker = 'o', linestyle = 'None')
        else:
            print 'adding nothing'

        title(patchNames[k])
        if xlimval != None:
            xlim(xlimval)
        if ylimval != None:
            ylim(ylimval)

    legend(prop={'size':9})

    if showFig:
        show()
    if saveFig == True:

        if saveFigName != None:
            savefig(saveFigName)
        else:
            print "didn't save anything, no name provided."




def forecastSignalToNoise(clSignal, autoPower1,  areaSqdeg, autoPower2 = None, clCross = None):

    if autoPower2 == None:
        autoPower2 = copy.deepcopy(autoPower1)

    if not array_equal( autoPower1[0], autoPower2[0]) :
        print 'ells do not agree, this is not yet implemented'
        
    fsky = areaSqdeg * (pi / 180)**2 / (4 * pi)

    dls = autoPower1[0][1:] - autoPower1[0][0:-1]

    #repeat last element of dls so that we have same number of objects.
    dls = append(dls, dls[-1])
    
    variance = 1./( (autoPower1[0] + 1) * dls *  fsky) * (autoPower1[1] * autoPower2[1] + \
                                             (clCross[1]**2 if clCross != None else 0.))

    stdev = sqrt(variance)

    f = scipy.interpolate.interp1d(clSignal[0], clSignal[1])
    clSignal_onells = f(autoPower1[0])
    
    sovern = sqrt(numpy.sum(clSignal_onells**2 / variance))
    
    return (sovern, stdev)
    

def makeCircMask(inmap, xcoords, ycoords, radius, smoothfactor = .08):

    mymask = numpy.ones(inmap.data.shape)
    nSources = len(xcoords)

    if nSources != len(ycoords):
        stop
    xx, yy  =     xs_and_ys(inmap.data.shape[1], inmap.data.shape[0])
    print xx.shape, yy.shape
    print 'number of sources is ' , nSources    

    for i   in range(nSources):


        locs = numpy.array(inmap.skyToPix( xcoords[i], ycoords[i]))

    
        # mymask[locs[1], locs[0]] = 0 # 

        dist = numpy.sqrt((xx - locs[0])**2 + (yy - locs[1])**2)

        
        wh = numpy.where (dist < radius)
        print 'loc is ', locs, 'length is ', wh[1].shape, wh[0].shape#, wh

        mymask[wh] = 0


    if smoothfactor != 0.0:
        pixScaleArcmin = 180. * 60. / numpy.pi * numpy.mean((inmap.pixScaleX, inmap.pixScaleY))

        output = scipy.ndimage.filters.gaussian_filter(mymask, radius * smoothfactor / pixScaleArcmin )
    else:
        output = mymask


    return output

def joinTwoMasks(mask1, mask2):

    if mask1 == None:
        return mask2

    if mask2 == None: 
        return mask1

    output = mask1.copy()
    output.data = mask1 * mask2

    return output




def deltaTOverTcmbToJyPerSr(freqGHz,T0 = 2.726):
    """
    @brief the function name is self-eplanatory
    @return the converstion factor
    stolen from Flipper but couldn't import due to underscore -- van engelen
    """
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freqGHz*1.e9
    x = h*nu/(kB*T0)
    cNu = 2*(kB*T0)**3/(h**2*c**2)*x**4/(4*(numpy.sinh(x/2.))**2)
    cNu *= 1e23
    return cNu




def deltaTOverTcmbToY(freqGHz, T0 = 2.726):
    """
    stolen from Flipper but couldn't import due to underscore -- van engelen
    """
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freqGHz*1.e9
    x = h*nu/(kB*T0)
    f_nu = x*(numpy.exp(x)+1)/(numpy.exp(x)-1) - 4
    return 1./f_nu 


def randomizePhase(inMap, inMap2 = None, inMap3 = None, seed = None):
    #stolen from /home/r/rbond/bsherwin/dis2/latestLensRecons/lensRecon/maps/dataMaps/actpolDeep/processDust.py
    #just changed "dustMap" -> "inMap"
    if seed != None:
        numpy.random.seed(seed)

    outputMap = inMap.copy()
    randomMap = inMap.copy()

    randomMap.data = numpy.random.randn(inMap.Ny,inMap.Nx)

    
    fftQ = fftTools.fftFromLiteMap(inMap)
    fftR = fftTools.fftFromLiteMap(randomMap)
    #fftU = fftTools.fftFromLiteMap(mapU)
    #    randomPhase = numpy.exp(numpy.random.rand(fftQ.Ny,fftQ.Nx)*2.*numpy.pi*1.j)

    # randomPhases = numpy.exp(numpy.angle(fftR.kMap)*2.*numpy.pi*1.j)
    randomPhases = fftR / numpy.absolute(fftR)
    
    fftQ.kMap *= randomPhases
    #fftU.kMap *= randomPhase
    outputMap.data = fftQ.mapFromFFT()


    if inMap2 != None:  #then we have three maps, i.e., T, Q, U
        if inMap3 == None:
            stop

        fft2 = fftTools.fftFromLiteMap(inMap2)
        fft3 = fftTools.fftFromLiteMap(inMap3)



        fft2.kMap *= randomPhases
        fft3.kMap *= randomPhases

        outputMap2 = inMap2.copy()
        outputMap3 = inMap3.copy()

        outputMap2.data = fft2.mapFromFFT()
        outputMap3.data = fft3.mapFromFFT()


        return outputMap, outputMap2, outputMap3

    else:
        return outputMap

    
    


    #mapU.data = fftQ.mapFromFFT()

    
def dustFromCommander(nuGHz, A_d, beta_d, nu0GHz = 545, T0 = 23):
    '''
    input map of A_d and beta_d, output map in RJ temperature units.  Follows table 4 of 1502.01588

    T0 is assumed a constant..
    '''
    h_planck = 6.62606957e-34
    kboltzmann = 1.3806488e-23

    

    gamma = h_planck  / (kboltzmann * T0)

    

    output = A_d * (nuGHz / nu0GHz)**(1 + beta_d) \
        * (exp(gamma * (nu0GHz * 1e9)) - 1) / (exp(gamma * (nuGHz * 1e9)) - 1)

    return output


def addMultiLiteMaps(liteMapList1, liteMapList2):

    output = []

    for i, thisMap in enumerate(liteMapList1):

        output += [thisMap.copy()]

        output[i].data += liteMapList2[i].data

        
    return output





def writeMultiLiteMaps(liteMapList, prefix, postfix):

    if len(liteMapList) == 2:
        
        liteMapList[0].writeFits(prefix + 'Q' + postfix, overWrite = True)
        liteMapList[1].writeFits(prefix + 'U' + postfix, overWrite = True)

    if len(liteMapList) == 3:
        liteMapList[0].writeFits(prefix + 'T' + postfix, overWrite = True)
        liteMapList[1].writeFits(prefix + 'Q' + postfix, overWrite = True)
        liteMapList[2].writeFits(prefix + 'U' + postfix, overWrite = True)

    return


def readMultiLiteMaps(prefix, postfix, doTwo = False):

    if doTwo:
        liteMapList = [None] * 2
        
        liteMapList[0] = liteMap.liteMapFromFits(prefix + 'Q' + postfix, overWrite = True)
        liteMapList[1] = liteMap.liteMapFromFits(prefix + 'U' + postfix, overWrite = True)

    else:
        liteMapList = [None] * 3

        liteMapList[0] = liteMap.liteMapFromFits(prefix + 'T' + postfix, overWrite = True)
        
        liteMapList[1] = liteMap.liteMapFromFits(prefix + 'Q' + postfix, overWrite = True)
        liteMapList[2] = liteMap.liteMapFromFits(prefix + 'U' + postfix, overWrite = True)

    return liteMapList


def kmapFromList( kFilterFromList, modLMap , nearest  = True ) :
#this was mostly stolen from fftTools.mapFromFFT()

    if nearest:
        func = scipy.interpolate.interp1d(kFilterFromList[0], kFilterFromList[1], kind = 'nearest', \
                                      fill_value = 0., bounds_error = False)

        kk = func(modLMap.flatten())
        
    else:  #do cubic spline.
        kFilter = output.kMap.copy()*0.
        l = kFilterFromList[0]
        Fl = kFilterFromList[1] 
        FlSpline = splrep(l,Fl,k=3)

        ll = numpy.ravel(modLMap)

        kk = (splev(ll,FlSpline))
    
    # output.kMap = numpy.reshape(kk,[fftTemplate.Ny,fftTemplate.Nx])
    return  numpy.reshape(kk,[modLMap.shape[1],modLMap.shape[0]])
        

def filterTquMapsFrom2d(inTqu, inFilters2d):

    output = []

    for i, thisMap in enumerate(inTqu):
        output += [filterFrom2d(thisMap, inFilters2d[i])]


        
    return output


def filterFrom2d(inmap, filter2d):

    output = inmap.copy()

    infft = fftTools.fftFromLiteMap(inmap)

    infft.kMap *= filter2d

    output.data = infft.mapFromFFT()

    return output


    # return output



def onedDict(names1):

    output = {}
    for a in names1:
        output[a] = {}
    return output

def twodDict(names1, names2):

    output = {}
    for a in names1:
        output[a] = {}
        
        for b in names2:
            output[a][b] = {}
            

    return output


def threedDict(names1, names2, names3):

    output = {}
    for a in names1:
        output[a] = {}
        
        for b in names2:
            output[a][b] = {}
            
            for c in names3:
                output[a][b][c] = {}

    return output

def fourdDict(names1, names2, names3, names4):

    output = {}
    for a in names1:
        output[a] = {}
        
        for b in names2:
            output[a][b] = {}
            
            for c in names3:
                output[a][b][c] = {}

                for d in names4:
                    output[a][b][c][d] = {}

    return output

def fivedDict(names1, names2, names3, names4, names5):

    output = {}
    for a in names1:
        output[a] = {}
        
        for b in names2:
            output[a][b] = {}
            
            for c in names3:
                output[a][b][c] = {}

                for d in names4:
                    output[a][b][c][d] = {}

                    for e in names5:
                        output[a][b][c][d][e] = {}

    return output


import matplotlib.cm as cm

#from http://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
import numpy as np
def threeColorScales(number_of_lines, start = 0.2, stop = 1.0):

    cm_subsection = np.linspace(start, stop, number_of_lines)

    colorsBlue = [ cm.Blues(x) for x in cm_subsection ]
    colorsRed = [ cm.Reds(x) for x in cm_subsection ]
    colorsGreen = [ cm.Greens(x) for x in cm_subsection ]

    allColors = [colorsBlue, colorsRed,colorsGreen]        
    return allColors


def checkInMask(raList, decList, goodMapArray, maskfilename):
    
    nPoints = len(raList)
    output = onedl(nPoints)
    
    mask, maskHeaderMask = healpy.fitsfunc.read_map(maskfilename,   h = True)

    hdulist = pyfits.open(maskfilename)
    nside = hdulist[1].header['NSIDE']
    
    for i in arange(nPoints):

        pix = healpy.pixelfunc.ang2pix(nside,
                                       numpy.pi / 180 * (90 - decList[i]) ,
                                       numpy.pi / 180 * raList[i])         
        if mask[pix] == 1:
            output[i] = True
        else:
            output[i] = False


    return output

    



def dustUnitFactors(mapUnitsStr):
    TCMB = 2.726e6
    output = dict()
    if mapUnitsStr == 'MJySr353':
        output['unitToUK150Fac'] =   1. / deltaTOverTcmbToJyPerSr(150) * TCMB * 1e6  / (353. / 150.)**(2 + 1.5)
        output['unitToMJy353Sr'] = 1.

    elif mapUnitsStr == 'KCMB353':
        output['unitToUK150Fac'] = 1e6 / (353. / 150.)**(2 + 1.5) * deltaTOverTcmbToJyPerSr(353) / deltaTOverTcmbToJyPerSr(150.)
        output['unitToMJy353Sr'] = deltaTOverTcmbToJyPerSr(353.) / TCMB
    elif mapUnitsStr == 'KCMB143':
        output['unitToUK150Fac'] = 1e6
        output['unitToMJy353Sr'] =  1. * deltaTOverTcmbToJyPerSr(150) / TCMB  * (353. / 150.)**(2 + 1.5)
    elif mapUnitsStr == 'uKCMB143':
        output['unitToUK150Fac'] = 1.
        output['unitToMJy353Sr'] =  1. * deltaTOverTcmbToJyPerSr(150) / TCMB  * (353. / 150.)**(2 + 1.5) / 1e6

    else:
        raise ValueError("mapUnits not recognized.")

    return output






