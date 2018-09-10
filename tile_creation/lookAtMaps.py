



import healpy





clarkMap = '/home/r/rbond/bsherwin/dis2/HIbased_templates/COM_CompMap_Dust-GNILC-F353_2048_R2.00_Equ_DR2_allsky_theta_RHT_hp_w75_s15_t70_p0.1const.fits'

clarkMaprotated = '../data/dust_map_gal.fits'
##############################################################################################################################

ffp8map = '../ffp8/COM_SimMap_thermaldust-ffp8-nobpm-143_2048_R2.00_full.fits'

ffp8MapRotated = '../ffp8/ffp8_map_gal.fits'






##############################################################################################################################
mapNameArr = [clarkMap, clarkMaprotated, ffp8map]


for i, mapName in enumerate(mapNameArr):

    healpixMapT,healpixMapQ,healpixMapU, mapHeader = healpy.fitsfunc.read_map(mapName, field = (0,1,2),\
                                                                              h = True)
    figure(i , figsize = ())
    clf()
    # subplot(311)
    healpy.mollview(healpixMapT, fig = i, sub = 311, title = mapName + ' T')
    # subplot(312)
    healpy.mollview(healpixMapQ, fig = i, sub = 312, title = mapName + ' Q')
    # subplot(313)
    healpy.mollview(healpixMapU, fig = i, sub = 313, title = mapName + ' U')

    
    
    show()
