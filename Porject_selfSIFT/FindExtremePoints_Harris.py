import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import filters

#Harris
class FindCornerPoints_Harris(object):
    def __init__(self):
        self.threshold = 0.0001
        self.eps = 0.000001#避免计算时分母为0

    def Corner(self, img):
        """
        takes an image array as input and return the pixels
        which are corners of the image
        """
        HarrisM = self.ComputeHarrisResponse(img)
        filtered_coords = self.GetHarrisPoints(HarrisM, 3)
        return filtered_coords

    def ComputeHarrisResponse(self, img, sigma = 1.5):
        imx = np.zeros(img.shape)#对图像求导，来源参考https://www.cnblogs.com/king-lps/p/6374916.html
        filters.gaussian_filter(img, (sigma,sigma), (0,1), imx)
        imy = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma,sigma), (1,0), imy)

        Wxx = filters.gaussian_filter(imx*imx,sigma)
        Wxy = filters.gaussian_filter(imx*imy,sigma)
        Wyy = filters.gaussian_filter(imy*imy,sigma)
        #计算Harris矩阵的行列式和迹

        Wdet = Wxx*Wyy - Wxy**2 #行列式的值
        Wtr = Wxx + Wyy+self.eps #迹

        return Wdet/Wtr #角点响应函数


    def GetHarrisPoints(self, HarrisM, min_dist):#通过计算H矩阵来去除角点
        Corner_threshold = self.threshold * HarrisM.max()
        Harrism_t = HarrisM > Corner_threshold
        coords = np.array(Harrism_t.nonzero()).T #摘取其中的非零点作为候选
        candidate_values = [HarrisM[c[0], c[1]] for c in coords]#对应的值

        index = np.argsort(candidate_values)
        allowed_locations = np.zeros(HarrisM.shape)
        allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

        # select the best points taking min_distance into account
        filtered_coords = []
        for i in index:
            if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
                filtered_coords.append(tuple(coords[i]))

                allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
                (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

        return tuple(filtered_coords)
#Test
img1 = cv2.imread('E:\_TempPhoto\p004.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.subplot(121)
plt.imshow(img1)
img2 = cv2.imread('E:\_TempPhoto\p005.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
FEP = FindCornerPoints_Harris()

