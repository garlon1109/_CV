import numpy as np
from scipy.ndimage import filters
import math

class GenerateDescriptor(object):
    def __init__(self):
        self.eps = 0.00001 #保证计算结果不为无穷

    def CreatDes(self, features, img):
        '''
        :param features: 特征点，即极值点坐标
        :param imgs: 图
        :return:
        '''
        desDict = {}
        img = img.astype(np.float64)

        for i in range(len(features)):
            desDict[(features[i][0], features[i][1])] = self.allocate(features[i][0], features[i][1], img)

        return desDict

    def direction(self,i,j,imarr):
        '''
        :param i:
        :param j:通过坐标计算每个像素的方向
        :param img:
        :return:
        '''
        mij = math.sqrt((imarr[i + 1, j] - imarr[i - 1, j]) ** 2
                        + (imarr[i, j + 1] - imarr[i, j - 1]) ** 2)
        theta = math.atan((imarr[i, j + 1] - imarr[i, j - 1])
                          / (imarr[i + 1, j] - imarr[i - 1, j] + self.eps))

        return mij, theta

    def HistoAssign(self, i, j, img):
        '''
        :param i:
        :param j:直方图分匹配，每个直方图20度，（lowe建议10度，这里先用20度算算看）
        :param img:
        :return:
        '''
        P = math.pi
        localDir = [0] * 18

        for b in range(i - 8, i):
            for c in range(j - 8, j):
                m, t = self.direction(b, c, img)
                if t >= P * -9 / 18 and t <= P * -8 / 18:
                    localDir[0] += m
                if t > P * -8 / 18 and t <= P * -7 / 18:
                    localDir[1] += m
                if t > P * -7 / 18 and t <= P * -6 / 18:
                    localDir[2] += m
                if t > P * -6 / 18 and t <= P * -5 / 18:
                    localDir[3] += m
                if t > P * -5 / 18 and t <= P * -4 / 18:
                    localDir[4] += m
                if t > P * -4 / 18 and t <= P * -3 / 18:
                    localDir[5] += m
                if t > P * -3 / 18 and t <= P * -2 / 18:
                    localDir[6] += m
                if t > P * -2 / 18 and t <= P * -1 / 18:
                    localDir[7] += m
                if t > P * -1 / 18 and t <= 0:
                    localDir[8] += m
                if t > 0 and t <= P * 1 / 18:
                    localDir[9] += m
                if t > P * 1 / 18 and t <= P * 2 / 18:
                    localDir[10] += m
                if t > P * 2 / 18 and t <= P * 3 / 18:
                    localDir[11] += m
                if t > P * 3 / 18 and t <= P * 4 / 18:
                    localDir[12] += m
                if t > P * 4 / 18 and t <= P * 5 / 18:
                    localDir[13] += m
                if t > P * 5 / 18 and t <= P * 6 / 18:
                    localDir[14] += m
                if t > P * 6 / 18 and t <= P * 7 / 18:
                    localDir[15] += m
                if t > P * 7 / 18 and t <= P * 8 / 18:
                    localDir[16] += m
                if t > P * 8 / 18 and t <= P * 9 / 18:
                    localDir[17] += m

        return localDir

    def allocate(self, i, j, img):
        '''
        SIFT描述子是关键点邻域高斯图像梯度统计结果的一种表示。
        通过对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量，这个向量是该区域图像信息的一种抽象，具有唯一性。
    Lowe建议描述子使用在关键点尺度空间内4*4的窗口中计算的8个方向的梯度信息，共4*4*8=128维向量表征。
        '''
        vec = [0] * 16
        vec[0] = self.HistoAssign(i - 8, j - 8, img)
        vec[1] = self.HistoAssign(i - 8, j, img)
        vec[2] = self.HistoAssign(i - 8, j + 8, img)
        vec[3] = self.HistoAssign(i - 8, j + 16, img)

        vec[4] = self.HistoAssign(i, j - 8, img)
        vec[5] = self.HistoAssign(i, j, img)
        vec[6] = self.HistoAssign(i, j + 8, img)
        vec[7] = self.HistoAssign(i, j + 16, img)

        vec[8] = self.HistoAssign(i + 8, j - 8, img)
        vec[9] = self.HistoAssign(i + 8, j, img)
        vec[10] = self.HistoAssign(i + 8, j + 8, img)
        vec[11] = self.HistoAssign(i + 8, j + 16, img)

        vec[12] = self.HistoAssign(i + 16, j - 8, img)
        vec[13] = self.HistoAssign(i + 16, j, img)
        vec[14] = self.HistoAssign(i + 16, j + 8, img)
        vec[15] = self.HistoAssign(i + 16, j + 16, img)

        return [val for subl in vec for val in subl]