import cv2
import matplotlib.pyplot as plt
import numpy as np

#读取图像
img1 = cv2.imread('E:\_TempPhoto\aaa.jpeg')
img2 = cv2.imread('E:\_TempPhoto\p005.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
matchpoint = 50#想要匹配的点的数目
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2,k=2) #利用k近邻算法找到最近的两个数据点，该值大于阈值（此处为0.75），则认为是good匹配点
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

#Homography Matrix && RANSAC
if len(good) > 4:#通过RANSAC找到其中最优点来求解单应性矩阵需要的4个像素点，从而得到8个参数
    ptsA = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,5.0)
    imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#拼接与显示
allImg = np.concatenate((img1,img2,imgOut),axis=1)
cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
cv2.imshow('Result',allImg)
cv2.waitKey(0)
