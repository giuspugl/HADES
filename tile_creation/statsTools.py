

from flipper import *
from flipperPol import *
from scipy import *
from scipy import linalg

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

def fivedl(rows, cols, depths1, depths2, depths3):
    
    a = [[[[[None for x in xrange(depths3)] for x in xrange(depths2)] for x in xrange(depths1)] for x in xrange(cols)] for x in xrange(rows)]

    return a




def combineFieldBandpowers(dataSpecs, stats, goodPatches):

    nMCs = len(stats)
    nPolCombs = len(stats[0])
    nPatches = len(stats[0][0])

    outputVals = twodl(nMCs, nPolCombs)
    outputStdev = twodl(nMCs, nPolCombs)

    for mc in arange(nMCs):
        for pc in arange(nPolCombs):

            vals = array([dataSpecs[mc][pc][i] for i in goodPatches])
            stdevs = array([stats[mc][pc][i]['stdev'] for i in goodPatches])

            outputVals[mc][ pc], outputInvVar = average(vals, axis = 0, weights = 1/stdevs**2, returned = True)
            outputStdev[mc][ pc] = 1/sqrt(outputInvVar)
        
    return (outputVals, outputStdev)



    
def getChi2(model, data, covariance):
    
    # dataMinModel = dataCrossSpec[mc][pc][patchNum] - model    # Cl - Cl_theory
    dataMinModel = data - model    # Cl - Cl_theory

    invcov = linalg.inv(covariance)                    # inv covariance matrix 
    firstdot = numpy.dot(invcov, numpy.transpose(dataMinModel))                  # inv cov mat * tr(Cl)

    output = numpy.dot(dataMinModel, firstdot)          # Cl-Cl_th * inv cov mat * tr(Cl-Cl_th) 


    return output



def getChi2sSeveral(mymodels, data, covariance, is2d = False):
    if is2d == False:
        n = len(mymodels)

        output = numpy.zeros(n)

        for i in arange(n):

            output[i] = getChi2(mymodels[i], data, covariance)





    else:
        n1 = len(mymodels)
        n2 = len(mymodels[0])
        
        output = numpy.zeros((n1, n2))
        print data.shape, covariance.shape

        for i in numpy.arange(n1):
            for j in numpy.arange(n2):
                output[i, j] = getChi2(mymodels[i][j], data, covariance)

    return output
        
def aveBinInAnnuli(p2d, binfile = 'binningTest'):
    
    (lower, upper, center) =  fftTools.readBinningFile(binfile)
    bins = concatenate(([lower[0]], upper))

    #note, include the right hand endpoints by adding a half to bins.  This should be avoidable if your version of digitize has the "right" keyword as an option; mine does not.
    digitized = numpy.digitize(ndarray.flatten(p2d.modLMap), bins + .5)

    data = ndarray.flatten(p2d.powerMap)
    #this is the one for loop.

    bin_means = numpy.zeros(len(center))
    bin_stds = numpy.zeros(len(center))
    bincount = numpy.zeros(len(center))

    for i in range(1, len(bins)):
        thisd = data[digitized == i]
        bin_means[i-1] = thisd.mean()
        bin_stds[i-1] = thisd.std()
        #we use half because only one half-plane is independent, if the field is real
        bincount[i-1] = len(thisd)/2
        
    # bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    # bin_stds = [data[digitized == i].std() for i in range(1, len(bins))]
        
    return lower, upper, center, bin_means, bin_stds, bincount


def stats( arrs, nwanted = None,  weights = None):


    if nwanted == None :
        nwanted = (arrs[0]).size

    if weights != None:
        raise Exception, "weighting is not fully implemented yet."

    ncurves = len(arrs)
 
    twodarr = numpy.array(arrs)

    
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






def showThreeMaps(mapList, minVal = None, maxVal = None, figNum = None, \
                      patchNames = ['deep1', 'deep5', 'deep6'], title = '', \
                      useData = False, plotLoc = '../data/', plotName = None, rescaleFacs = [1, 1, 1], doFigs = True, figSize = (8, 15), cmap = None, interpolation = 'none'): #whether to use the ".data" element
    print 'mm', maxVal, minVal
    if plotName == None:
        plotName = title + '.pdf'
        
        #the default will just open up a figure numbered 1.
    pylab.figure(figNum, figsize=figSize)
    fig = pylab.gcf()
    if title != '':
        fig.canvas.set_window_title(title)
    pylab.clf()

    nPatches = len(patchNames)
    
#set global min and max for all 3 maps.
    if minVal == None and maxVal == None:

        mins = numpy.zeros(nPatches)
        maxes = numpy.zeros(nPatches)

        for i, patchName in enumerate(patchNames):
            
            if mapList[i] != None:
                if useData:
                    mins[i] = mapList[i].data.min()
                    maxes[i] = mapList[i].data.max()
                else:
                    mins[i] = mapList[i].min()
                    maxes[i] = mapList[i].max()
            else:
                mins[i] = nan
                maxes[i] = nan
                
        minVal = numpy.nanmin(mins)
        maxVal = numpy.nanmax(maxes)

        print 'mm2', maxVal, minVal, maxes, mins
        print 'mm3', maxVal, minVal




    for i, patchName in enumerate(patchNames):

        pylab.subplot(len(patchNames), 1, i + 1)
        if mapList[i] != None:
            if useData:
                toShow = mapList[i].data
            else:
                toShow = mapList[i]


            pylab.imshow(toShow / rescaleFacs[i], vmin = minVal, vmax = maxVal, cmap = cmap, interpolation = interpolation)

            pylab.title(title + ', ' + patchName)

            pylab.colorbar()

    #pylab.tight_layout()
    if doFigs: 
        pylab.show()
    # else:
    #     pylab.savefig('../plot/' + )

    return fig


def rescaleWeights(weightMap):
    #rescale a weight map so that it has value of roughly 1, inverse variance weighting by itself -- assuming the pixels have units like 1/sigmasquared.

    one_over_sigmanaughtsqr = numpy.sum(weightMap.data**2) / numpy.sum(weightMap.data)

    sigmanaught = 1/sqrt(one_over_sigmanaughtsqr)
    
    dimlesswt = weightMap.data * sigmanaught**2
    return dimlesswt

def effectivearea( weightMap ):
    one_over_sigmanaughtsqr = numpy.sum(weightMap.data**2) / numpy.sum(weightMap.data)

    sigmanaught = 1/sqrt(one_over_sigmanaughtsqr)
    
    dimlesswt = weightMap.data * sigmanaught**2

    return    weightMap.area * numpy.mean(dimlesswt)


def quickPower(inmap, inmap2 = None, window = None, useFlipperRoutine = False) :

    temp = inmap.copy()

    if inmap2 == None:
        temp2 = inmap.copy()
    else:
        temp2 = inmap2.copy()


    if window != None :
        temp.data *= window.data / sqrt(numpy.mean(window.data**2))
        temp2.data *= window.data / sqrt(numpy.mean(window.data**2))
#        print "correcting for window"

    power2D = fftTools.powerFromLiteMap(temp, liteMap2 = temp2)

    if useFlipperRoutine:
        llower,lupper,lbin,clbin_0,clbin_N,binweight = power2D.binInAnnuli('binningTest', nearestIntegerBinning=True)
    else:
        llower,lupper,lbin,clbin_0,clbin_N,binweight = aveBinInAnnuli(power2D, 'binningTest')

    output = {'lbin':lbin, 'clbin' : clbin_0, 'dlbin': lbin*(lbin+1) * clbin_0 / 2/numpy.pi}

    # return (lbin, clbin_0)
    return output




def plusminuscheck(val1, val2, formatcode = '4.2f'):
    str1 = '%4.2f' %val1
    str2 = '%4.2f' %val2
    if str1 == str2:
        return('$  \pm ' + str1 + '  $')
    else:
        return('$^{+' + str1 + '}_{-' + str2 + ' }$')


def plusminuscheckNodollar(val1, val2, formatcode = '4.2f'):
    str1 = '%4.2f' %val1
    str2 = '%4.2f' %val2
    if str1 == str2:
        return('  \pm ' + str1 + '  ')
    else:
        return('^{+' + str1 + '}_{-' + str2 + ' }')


def findNearestIndex(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
