

############### begin EB filters  ########

def makeEBxxx(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx**3 * twoDModel / (twoDModel+noisePower)
    return filter

def makeEBxxy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx**2 * ly * twoDModel / (twoDModel+noisePower)
    return filter

def makeEBxyy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx * ly**2 * twoDModel / (twoDModel+noisePower)
    return filter

def makeEByyy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * ly**3 * twoDModel / (twoDModel+noisePower)
    return filter

def makeEBxx(twoDModel,noisePower,ft, lx, ly):
    filter = lx**2 / (twoDModel+noisePower)
    return filter

def makeEBxy(twoDModel,noisePower,ft, lx, ly):
    filter = lx * ly / (twoDModel+noisePower)
    return filter

def makeEByy(twoDModel,noisePower,ft, lx, ly):
    filter = ly**2 / (twoDModel+noisePower)
    return filter

def makeFilterListsEBLx(twoDModel, noisePower, ft, lx, ly):
    listX0 = [2.*makeEBxxx(twoDModel, noisePower, ft, lx, ly), -2.*makeEBxyy(twoDModel, noisePower, ft, lx, ly), 2.*makeEBxxy(twoDModel, noisePower, ft, lx, ly), -2.*makeEBxxy(twoDModel, noisePower, ft, lx, ly)]
    listX1 = [makeEBxy(twoDModel, noisePower, ft, lx, ly), makeEBxy(twoDModel, noisePower, ft, lx, ly), makeEByy(twoDModel, noisePower, ft, lx, ly), makeEBxx(twoDModel, noisePower, ft, lx, ly)]
    return listX0, listX1

def makeFilterListsEBLy(twoDModel,noisePower,ft,lx,ly):
    listY0 = [2.*makeEBxxy(twoDModel, noisePower, ft, lx, ly), -2.*makeEByyy(twoDModel, noisePower, ft, lx, ly), 2.*makeEBxyy(twoDModel, noisePower, ft, lx, ly),-2.*makeEBxyy(twoDModel, noisePower, ft, lx, ly)]
    listY1 = [makeEBxy(twoDModel, noisePower, ft, lx, ly), makeEBxy(twoDModel, noisePower, ft, lx, ly), makeEByy(twoDModel, noisePower, ft, lx, ly), makeEBxx(twoDModel, noisePower, ft, lx, ly)]
    return listY0, listY1

####### End EB filters ##########


####### begin EE filters #############

def makeEExxx(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx**3 * twoDModel / (twoDModel+noisePower)
    return filter

def makeEExxy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx**2 * ly * twoDModel / (twoDModel+noisePower)
    return filter

def makeEExyy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx * ly**2 * twoDModel / (twoDModel+noisePower)
    return filter

def makeEExx(twoDModel,noisePower,ft, lx, ly):
    filter = lx**2 / (twoDModel+noisePower)
    return filter

def makeEExy(twoDModel,noisePower,ft, lx, ly):
    filter = lx * ly / (twoDModel+noisePower)
    return filter

def makeEEyy(twoDModel,noisePower,ft, lx, ly):
    filter = ly**2  / (twoDModel+noisePower)
    return filter

def makeEEx(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * lx / (twoDModel+noisePower)
    return filter

def makeEEy(twoDModel,noisePower,ft, lx, ly):
    l = ft.modLMap
    filter = l * ly / (twoDModel+noisePower)
    return filter

def makeEE(twoDModel,noisePower,ft, lx, ly):
    filter = 1. / (twoDModel+noisePower)
    return filter

def makeFilterListsEELx(twoDModel,noisePower,ft,lx,ly):
    listX0 = [2.*makeEExxx(twoDModel,noisePower,ft,lx,ly),4.*makeEExxy(twoDModel,noisePower,ft,lx,ly),\
                  makeEExyy(twoDModel,noisePower,ft,lx,ly),-makeEEx(twoDModel,noisePower,ft,lx,ly)]
    listX1 = [makeEExx(twoDModel,noisePower,ft,lx,ly),makeEExy(twoDModel,noisePower,ft,lx,ly),\
                  makeEEyy(twoDModel,noisePower,ft,lx,ly),makeEE(twoDModel,noisePower,ft,lx,ly)]
    return listX0, listX1

def makeFilterListsEELy(twoDModel,noisePower,ft,lx,ly):
    listY0 = [2.*makeEExxy(twoDModel,noisePower,ft,lx,ly),4.*makeEExyy(twoDModel,noisePower,ft,lx,ly),\
                  makeEEyyy(twoDModel,noisePower,ft,lx,ly),-makeEEy(twoDModel,noisePower,ft,lx,ly)]
    listY1 = [makeEExx(twoDModel,noisePower,ft,lx,ly),makeEExy(twoDModel,noisePower,ft,lx,ly),\
                  makeEEyy(twoDModel,noisePower,ft,lx,ly),makeEE(twoDModel,noisePower,ft,lx,ly)]
    return listY0, listY1

########  end EE filters ############


####### begin TT filters #########

def makeTTF1x(twoDModel,noisePower,ft,lx,ly):
    l = ft.modLMap
   # lx[ft.modLMap<500.] = 0.
    filter = twoDModel/(twoDModel+noisePower)*lx*l*(0.+1.j)
    return filter

def makeTTF1y(twoDModel,noisePower,ft,lx,ly):
    l = ft.modLMap
   # lx[ft.modLMap<500.] = 0.
    filter = twoDModel/(twoDModel+noisePower)*ly*l*(0.+1.j)
    return filter

def makeTTF2(twoDModel,noisePower,ft,lx,ly):
    filter = 1./(twoDModel+noisePower)
    return filter

def makeFilterListsTTLx(twoDModel,noisePower,ft,lx,ly):
    listX0 = [makeTTF1x(twoDModel,noisePower,ft,lx,ly)]
    listX1 = [makeTTF2(twoDModel,noisePower,ft,lx,ly)]
    return listX0, listX1

def makeFilterListsTTLy(twoDModel,noisePower,ft,lx,ly):
    listY0 = [makeTTF1y(twoDModel,noisePower,ft,lx,ly)]
    listY1 = [makeTTF2(twoDModel,noisePower,ft,lx,ly)]
    return listY0, listY1

###### end TT filters #########

