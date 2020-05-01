import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd

def gen_sample_data():
    N = 400
    centers = 3
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    return data, y

def InitCenter(k, x_trian):
    '''
    :param k:聚类中心
    :param x_trian:训练集
    :return:k个中心
    '''
    m = x_trian.shape[0]
    n = x_trian.shape[1]
    Center = np.zeros([k,n])
    np.random.seed(15)
    for i in range(k):
        x = np.random.randint(m)
        Center[i] = np.array(x_trian[x])
    return Center

def GetDistance(x_train, Center):
    '''
    :param x_train:训练集
    :param Center:中心点
    :return:
    '''
    m = x_train.shape[0]
    n = x_train.shape[1]
    k = Center.shape[0]

    Distance = []
    for j in range(k):
        for i in range(m):
            x = np.array(x_train[i])
            a = x - Center[j]
            Dist = np.sqrt(np.sum(np.square(a)))#欧式距离
            Distance.append(Dist)#共生成m个与k个中心的距离
    Dist_array = np.array(Distance).reshape(k,m)
    return Dist_array

def GetNewCenter(x_train, Dist_array):
    m = x_train.shape[0]
    n = x_train.shape[1]
    k = Dist_array.shape[0]

    NewCenter = []
    train = [0] * k
    cls = np.argmin(Dist_array, axis = 0)#最小值的索引,用来筛选出每个点距离哪个中心点更近
    axisx, axisy = [], []
    for i in range(k):
        train[i] = np.array(x_train[cls == i])
        print(train[i].shape)
        xx, yy = train[i][:,0], train[i][:,1]
        axisx.append(xx)
        axisy.append(yy)
        meanC = np.mean(train[i], axis = 0)
        NewCenter.append(meanC)
    NewCen = np.array(NewCenter).reshape(k, n)
    newcen = np.nan_to_num(NewCen)
    return newcen, axisx, axisy

def KMcluster(x_train,k,threshold):
    m = x_train.shape[0]
    n = x_train.shape[1]

    global axis_x, axis_y
    center = InitCenter(k, x_train)
    initcenter = center
    centerChanged = True
    t=0
    while centerChanged:
        Dis_array = GetDistance(x_train, center)
        center ,axis_x,axis_y= GetNewCenter(x_train, Dis_array)
        err = np.linalg.norm(initcenter[-k:] - center)
        t+=1
        #plt.ion()
        print('err of Iteration '+str(t),'is',err)
        plt.figure(1)
        p1,p2,p3,p4 = plt.scatter(axis_x[0], axis_y[0], c='r'),plt.scatter(axis_x[1], axis_y[1], c='g'),plt.scatter(axis_x[2], axis_y[2], c='b'),plt.scatter(axis_x[3], axis_y[3], c='y')
        plt.legend(handles=[p1, p2, p3, p4], labels=['0', '1', '2','4'], loc='best')
        if err < threshold:
            centerChanged = False
        else:
            initcenter = np.concatenate((initcenter, center), axis=0)
    plt.show()
    return center, axis_x, axis_y, initcenter

