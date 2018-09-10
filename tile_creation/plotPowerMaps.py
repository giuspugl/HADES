


from flipper import *
from flipperPol import *
import healpy
import aveTools
import pickle
p = flipperDict.flipperDict()
p.read_from_file(sys.argv[1])
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import sys

tqus = ['T', 'Q', 'U']
TCMB = 2.726e6

goodMap = pickle.load( open(p['workDir'] + p['basename'] + 'goodMap.pkl', 'r'))
mapRas = pickle.load( open(p['workDir'] + p['basename'] + 'mapRas.pkl', 'r'))
mapDecs = pickle.load( open(p['workDir'] + p['basename'] + 'mapDecs.pkl', 'r'))

#write out this info.
np.savetxt(p['workDir'] + p['basename'] + 'mapRAsDecs.txt', \
           np.transpose([np.arange(len(goodMap)), \
                         mapRas, \
                         mapDecs]) , \
           fmt = ('%05d', '%f', '%f'), header = "patch number; galactic 'ra'; galactic 'dec'")

stop


nMaps = len(goodMap)
# nGoodMaps = np.sum(goodMap)

powers = pickle.load(open(p['workDir'] + p['basename'] + 'Powers.pkl', 'r'))



nGoodMaps = sum(goodMap)

polCombsToPlot = ['cl_TT',  'cl_EE', 'cl_BB']
polCombsPretty = ['$C_l^{TT}$', '$C_l^{EE}$', '$C_l^{BB}$']
uniqueAbsDecs  = np.unique(np.fabs(mapDecs).round(decimals = 4))
nUniqueAbsDecs = len(uniqueAbsDecs)

allColors = aveTools.threeColorScales(nUniqueAbsDecs)



plt.figure('dustPower ' + p['leadAuthorTag'], figsize = (15,15))
plt.clf()

# gs = gridspec.GridSpec(1, 4, width_ratios=[30,1, 1, 1])


# axes = [None] * 4
# for p in range(2):
#     axes[p] = plt.subplot(gs[p])

nPolCombs = len(polCombsToPlot)
linestyles = ['solid', 'dashed', 'dotted']


unitFactors = aveTools.dustUnitFactors(p['mapUnits'])

# prefacKappa = unitToUK150Fac**4
prefacDustPower = unitFactors['unitToMJy353Sr']**2

nMaps = 256
print 'kludge, nMaps set to %i' % nMaps


for pc, polComb in enumerate(polCombsToPlot):
    plt.subplot(1,3,pc + 1)
    for  mapnum in range(nMaps):

        if goodMap[mapnum]:



            # plt.sca(axes[0])
            # subplot(nPolCombs, 1, pc + 1)
            # semilogy(powers[mapnum]['lbin'], powers[mapnum][polComb], color = colors[pc])
            thisUniqueDec  = np.fabs(mapDecs[mapnum]).round(decimals = 4)
            colorIndex = np.where(uniqueAbsDecs == thisUniqueDec)
            try:
                plt.semilogy(powers[mapnum]['lbin'], prefacDustPower * powers[mapnum][polComb], \
                             color = allColors[pc][colorIndex[0][0]] )
            except:
                pass
            # plt.sca(ax1)

            
    plt.ylabel("$\widehat C_{\ell}  (MJy^2_\mathrm{353 GHz}$ / Sr)"  , \
               fontsize = 18)
    plt.xlabel('$\ell$' ,fontsize = 18)
    plt.xlim([0,4000])


    plt.title(polCombsPretty[pc])

    plt.ylim([1e-15,1e-1])
plt.tight_layout()

    # text(.1 + .1 * pc, 0 , polComb,  transform = gca().transAxes, \
            #      color = allColors[pc][len(uniqueAbsDecs) - 1], fontsize = 18)


# for pc, polComb in enumerate(polCombsToPlot):
#     plt.sca(axes[pc + 1])
#     for dn, decVal in  enumerate(uniqueAbsDecs):
#         plt.axhline(decVal, color = allColors[pc][dn])            


            
plt.savefig('../plot/dustPower%s.pdf' % p['leadAuthorTag'])            
plt.savefig('../plot/dustPower%s.png'% p['leadAuthorTag'])            
plt.show()



firstTimeMean = True

for  mapnum in range(nMaps):

    if goodMap[mapnum]:
        if firstTimeMean:
            meanRatio = np.zeros(len(powers[mapnum]['cl_BB']))
            firstTimeMean = False
        meanRatio += powers[mapnum]['cl_EE'] / powers[mapnum]['cl_BB']
meanRatio /= nGoodMaps
        
            

figure('ratioPower ' + p['leadAuthorTag'], figsize = (7,7))
clf()
for  mapnum in range(nMaps):

    if goodMap[mapnum]:

    # subplot(nPolCombs, 1, pc + 1)
        semilogy(powers[mapnum]['lbin'],  powers[mapnum]['cl_EE'] / powers[mapnum]['cl_BB'] , color = '.7')

semilogy(powers[mapnum]['lbin'],  meanRatio , color = 'k')
        
title('$C_\ell^{EE} / C_\ell^{BB}$')
xlim([0,4000])
ylim([1e-1,1e1])
axhline(1, linestyle = 'dotted')        
axhline(2, linestyle = 'dotted')        

show()
plt.savefig('../plot/ratioPower.pdf')
plt.savefig('../plot/ratioPower.png')


            

figure('ratioPowerEEoverTT '+ p['leadAuthorTag'], figsize = (7,7))
clf()
for  mapnum in range(nMaps):

    if goodMap[mapnum]:

    # subplot(nPolCombs, 1, pc + 1)
        semilogy(powers[mapnum]['lbin'],  powers[mapnum]['cl_EE'] / powers[mapnum]['cl_TT'] , color = '.7')

# semilogy(powers[mapnum]['lbin'],  meanRatio , color = 'k')
        
title('$C_\ell^{EE} / C_\ell^{TT}$')
xlim([0,4000])
ylim([1e-2,1e2])
axhline(1, linestyle = 'dotted')        
axhline(2, linestyle = 'dotted')        

show()
plt.savefig('../plot/ratioPowerEEoverTT.pdf')
plt.savefig('../plot/ratioPowerEEoverTT.png')


stop
figure('ratioPower', figsize = (7,7))
clf()
for  mapnum in range(nMaps):
# for  mapnum in range(nMaps):

    if goodMap[mapnum]:

    # subplot(nPolCombs, 1, pc + 1)
        semilogy(powers[mapnum]['lbin'],  powers[mapnum]['cl_EE'] / powers[mapnum]['cl_BB'] , color = '.7')

semilogy(powers[mapnum]['lbin'],  meanRatio , color = 'k')
        
title('$C_\ell^{EE} / C_\ell^{BB}$')
xlim([0,4000])
ylim([1e-1,1e1])
axhline(1, linestyle = 'dotted')        
axhline(2, linestyle = 'dotted')        

show()



figure('decs', figsize = (7,7))

for  mapnum in range(nMaps):
    if goodMap[mapnum]:
        lbin = powers[mapnum]['lbin']
        break

    
myLval = 2000

myInd = where( (lbin < myLval)  *   (lbin > myLval - 60))[0][0]





for pc, polComb in enumerate(polCombsToPlot):
    for  mapnum in range(nMaps):



        if goodMap[mapnum]:
 
            # subplot(nPolCombs, 1, pc + 1)
            print mapDecs[mapnum], powers[mapnum][polComb][myInd]

            scatter( [mapDecs[mapnum]], [powers[mapnum][polComb][myInd]]  , color = allColors[pc][6])
            


# semilogy(powers[mapnum]['lbin'],  meanRatio , color = 'k')
        
title('$C_\ell^{EE} / C_\ell^{BB}$')
xlim([0,4000])
ylim([1e-1,1e1])
axhline(1, linestyle = 'dotted')        
axhline(2, linestyle = 'dotted')        

show()


