import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
from matplotlib import animation, rc
from sklearn import datasets



class TrainRegression:
    def __init__(self, w_init, b_init, algo = 'GD', inference = 'LogicRegression'):
        self.w = w_init
        self.b = b_init
        self.w_h = [] #记录各个初始w，b值下的训练情况
        self.w_b = []
        self.e_h = []#记录误差函数
        self.algo = algo #算法
        self.inference = inference

    def Inference(self, x, w = None, b = None): #目标函数
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        if self.inference == 'LinearRegression':
            return w*x + b
        elif self.inference == 'LogicRegression':
            return 1./(1.+np.exp(-(w*x+b)))

    def LossFunction(self, X, Y, w = None, b = None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        avg_loss = 0
        for x,y in zip(X, Y):
            avg_loss += 0.5 * (self.Inference(x, w, b) - y)**2
            avg_loss /= X.shape[0]
        return avg_loss


    def grad(self,x,y, w = None, b = None):#单点梯度
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        if self.inference == 'LinearRegression':
            w = (self.Inference(x, w, b) - y) * x
            b = self.Inference(x, w, b) - y

        elif self.inference == 'LogicRegression':
            w = (self.Inference(x, w, b) - y) * self.Inference(x, w, b) * (1 - self.Inference(x, w, b))*x
            b = (self.Inference(x, w, b) - y) * self.Inference(x, w, b) * (1 - self.Inference(x, w, b))
        return w, b


    def fit(self, X, Y, Iteration = 100, mini_batch_size = 100, alpha = 0.01,gamma = 0.9):
        '''

        :param Iteration: 迭代次数
        :param mini_batch_size: 最小样本数
        :param alpha: 学习率
        :return:
        '''
        self.w_h = []
        self.b_h = []
        self.e_h = []
        self.X = X
        self.Y = Y
        if self.algo == 'GD':
            for i in range(Iteration):
                dw, db = 0, 0
                for x, y in zip(X,Y):
                    dw += self.grad(x,y)[0]
                    db += self.grad(x,y)[1]

                self.w -= alpha * dw / X.shape[0]
                self.b -= alpha * db / X.shape[0]
                self.append_log()

        elif self.algo == 'MiniBatch':#使用变量points_seen记录已看到的点数，当为MiniBatch_Size的倍数时更新w和d，当MiniBatch_Size为1时为随机梯度下降算法
            for i in range(Iteration):
                dw, db = 0, 0
                points_seen = 0
                for x,y in zip(X,Y):
                    dw += self.grad(x,y)[0]
                    db += self.grad(x,y)[1]
                    points_seen += 1
                if points_seen % mini_batch_size == 0:
                    self.w -= alpha * dw / mini_batch_size
                    self.b -= alpha * db / mini_batch_size
                self.append_log()

        elif self.algo == 'Momentum':#根据之前的学习率进行学习率的调整
            v_w, v_b = 0, 0
            for i in range(Iteration):
                dw, db = 0
                v_w = gamma * v_w
                v_b = gamma * v_b
                for x, y in zip(X,Y):
                    dw += self.grad(x,y,self.w - v_w, self.b-v_b)[0]
                    db += self.grad(x,y,self.w - v_w, self.b-v_b)[1]
                v_w = gamma * v_w + alpha*dw
                v_b = gamma * v_b + alpha*db
                self.w = self.w - v_w
                self.b = self.b - v_b
                self.append_log()

    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
        self.e_h.append(self.LossFunction(self.X, self.Y))



#data1 for linearregression
bias = 10.
X, Y, coef = make_regression(n_features=1, noise=9, bias=bias, coef=True)
X = X.reshape(-1)

#learning algorithum parameter
algo = 'GD'
w_init = -2
b_init = -2
Iteration = 1000
mini_batch_size = 6
alpha = 1

GD = TrainRegression(w_init, b_init, algo)
GD.fit(X,Y, Iteration = Iteration,mini_batch_size= mini_batch_size, alpha = alpha)
plt.plot(GD.e_h,'r')
plt.plot(GD.w_h,'b')
plt.plot(GD.b_h,'g')
plt.legend(('error','weight','bias'))#图例
plt.title("Variation of lossfunction & parameters ")#图名
plt.xlabel("Iteration")
plt.show()


#plot
plt.ion()
plt.title('linear regression ')
plt.text(-2, 120, "Red line is real linear equation \n,w={0} \n,b={1}".format(GD.w,GD.b), color="b")
plt.scatter(X, Y, c='r', alpha=0.5,marker='x')
plt.plot(X, GD.w * X + GD.b,label = 'linear regresion')
plt.legend(loc = 'lower right')
plt.pause(10)



