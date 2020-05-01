import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter




class PreProcess(object):
    def __init__(self):
        self.sigma = 1.6

    def CreateDOG(self, img):#create Difference of Gauss
        res = [0,1,2,3] #共生成4个octave
        res[0] = self.diff(self.GaussBlur(self.sigma,img))
        for i in range(1,4):
            base = self.sampling(res[i-1][2])#由于倒数第二张尺度与上一层尺度类似，因此取倒数第二张
            res[i] = self.diff(self.GaussBlur(self.sigma, base))
        return res

    def diff(self, images):
        '''
        接收一个octave下的5个不同图片，并输出其差分
        :param images:
        :return:
        '''

        diffArray = [0,1,2,3]

            # compute the difference bewteen two adjacent images in the same ovtave
        for i in range(1,5):
            diffArray[i-1] = images[i]-images[i-1]

        return np.array(diffArray)


    def GaussBlur(self, k, img):
        """
        use gaussina blur to generate five images in different sigma value
        input: a k as constant, and an image in array form
        return: a list contains five images in image form which are blurred
        """
        SIG = self.sigma
        sig = [SIG, k * SIG, k * k * SIG, k * k * k * SIG, k * k * k * k * SIG]
        gsArray = [0, 1, 2, 3, 4]
        scaleImages = [0, 1, 2, 3, 4]

        for i in range(5):
            gsArray[i] = gaussian_filter(img, sig[i])

        return gsArray


    def normalize(self,img):
        """
        normalize the pixel intensity
        """
        img = img/(img.max()/255.0)
        return img

    def sampling(self, img): #resize 插值法。。有点慢
        imgInfo = img.shape
        height = imgInfo[0]
        width = imgInfo[1]

        dstHeight = int(height / 2)
        dstwidth = int(width / 2)

        dst = np.zeros((dstHeight, dstwidth), np.uint8)
        for i in range(dstHeight):
            for j in range(dstwidth):
                dsti = i * 2
                dstj = j * 2

                dst[i][j] = img[int(dsti)][int(dstj)]

        return dst

#Test
'''
img = cv2.imread('E:\_TempPhoto\p004.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sigma = 1.5

CreatDog = PreProcess()
Dog = CreatDog.CreateDOG(img)
print(Dog[2][1])
'''