import scipy
import numpy as np
import cv2
import PreProcess


import FindEdges_Hessian
import FindExtremePoints_Harris

class FindExtreme(object):
    def getPatextremes(self, ims, pa):
        """
        find local extremas on pattern image
        """

        # instantiate funtional class
        hs = FindExtremePoints_Harris.FindCornerPoints_Harris()
        hess = FindEdges_Hessian.FindEdges_Hessian()
        #        cont = contrast.contrast()

        coordinates = []
        temp = {}

        H = [0, 1, 2, 3]
        W = [0, 1, 2, 3]

        for i in range(4):
            H[i] = len(ims[i][0])
            W[i] = len(ims[i][0][0])

        localArea = [0, 1, 2]

        # get the unstable and low contrast pixel
        hs_points = hs.Corner(pa)
        hess_points = hess.EdgeDetect(pa)
        #        low_contrast = cont.lowcontrast(pa)

        # compute the pixels which are not situable for pixel matching
        bad_points = list(set(hs_points) | set(hess_points))
        bad = dict.fromkeys(bad_points, 0)

        for m in range(4):
            for n in range(1, 3):
                for i in range(16, H[m] - 16):
                    for j in range(16, W[m] - 16):
                        if bad.__contains__((i * 2 ** m, j * 2 ** m)) == False:

                            # compare local pixel with its 26 neighbour
                            currentPixel = ims[m][n][i][j]
                            localArea[0] = ims[m][n - 1][i - 1:i + 2, j - 1:j + 2]
                            localArea[1] = ims[m][n][i - 1:i + 2, j - 1:j + 2]
                            localArea[2] = ims[m][n + 1][i - 1:i + 2, j - 1:j + 2]

                            Area = np.array(localArea)

                            maxLocal = np.array(Area).max()
                            minLocal = np.array(Area).min()

                            if (currentPixel == maxLocal) or (currentPixel == minLocal):
                                if temp.__contains__((i * 2 ** m, j * 2 ** m)) == False:
                                    coordinates.append([int(i * 2 ** m), int(j * 2 ** m)])
                                    temp[(i * 2 ** m, j * 2 ** m)] = [i * 2 ** m, j * 2 ** m]
        return coordinates


#test
img_1 = cv2.imread('E:\_TempPhoto\p004.jpg')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_1 = np.dot(img_1, 255 / img_1.max())  # 归一化

CreateDOG = PreProcess.PreProcess()
DOG = CreateDOG.CreateDOG(img_1)
FE = FindExtreme()
coords = FE.getPatextremes(DOG, img_1)
print(coords)