from scipy.ndimage import filters
import scipy
import numpy as np
import scipy.ndimage
import math
import itertools

class FindEdges_Hessian(object):
    def __init__(self):
        self.eps = 0.000001#避免计算时分母为0
        self.EdgeThreshold = 4.1

    def EdgeDetect(self, img, sigma = 3):
        '''
        :param img:输入一组图片
        :return: 输出边缘像素坐标
        '''

        imx = np.zeros(img.shape)#对图像求导，来源参考https://www.cnblogs.com/king-lps/p/6374916.html
        filters.gaussian_filter(img, (sigma,sigma), (0,1), imx)
        imy = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma,sigma), (1,0), imy)

        Wxx = filters.gaussian_filter(imx*imx,sigma)
        Wxy = filters.gaussian_filter(imx*imy,sigma)
        Wyy = filters.gaussian_filter(imy*imy,sigma)

        Wdet = Wxx * Wyy - Wxy ** 2
        Wtr = Wxx + Wyy

        coord = []

        Hess = Wtr**2/(Wdet + self.eps)
        Harrism_t = np.where(Hess > self.EdgeThreshold)

        for i in range(len(Harrism_t)):
            coord.append((Harrism_t[0][i], Harrism_t[1][i]))

        return tuple(coord)

