from __future__ import print_function
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

folder_list = ["./data/I/", "./data/II/"]

train_name = "train.txt"
test_name = "test.txt"
data1_root_file = "./data/I/label.txt"
data2_root_file = "./data/II/label.txt"


# # 为了获取生成bounding box的坐标
# def gen_bbox(label_txt):
#     with open(label_txt, "r") as f:
#         lines = f.readlines()
#         # print(lines)
#     # 因为readlines（）可以把每一行当做一个元素储存起来，所以for line in lines就会遍历每一行
#     bbox_list = []
#     for line in lines:
#         line = line.strip()  # 因为readlines它包括换行符，所以strip()是将换行符给去了
#         # print(line)
#         components = line.split()
#         # print(components)
#         bbox = [components[1], components[2], components[3], components[4]]
#         # print(bbox)
#         bbox_list.append(bbox)
# [['229.0', '38.0', '289.0', '99.0'], ['4.0', '54.0', '109.0', '158.0'], ['44.0', '9.0', '153.0', '118.0'], ['218.0', '82.0', '328.0', '192.0'],.....] list形式
# print(bbox_list)
# print(len(bbox_list))  # 1953
# return bbox_list

# def gen_bbox(label_txt):
#     data = pd.read_csv(data1_root_file, delimiter=" ")
#     print(data)
#     x = data.iloc[0, :]
#     print(x)
#
#     return x

def get_bbox(label_txt):
    # 不用想一次性将两种数据类型都一下子能分离开来，只要将它们统一成字符串的形式分割开来，然后想要数字是在将其转化过来即可
    data = np.loadtxt(label_txt, dtype=str)
    # print(data)
    bbox = data[:, 1:5].astype("float")
    # print(bbox)
    return bbox


# gen_bbox(data1_root_file)


# def gen_landmark(label_txt):
#     with open(label_txt, "r") as f:
#         lines = f.readlines()
#         print(lines)
#         # 因为readlines（）可以把每一行当做一个元素储存起来，所以for line in lines就会遍历每一行
#     for line in lines:
#         line = line.strip()  # 因为readlines它包括换行符，所以strip()是将换行符给去了
#         # print(line)
#         components = line.split()
#         # print(components)
#         x_landmark_list = list(components[5::2])
#         # print(len(x_landmark_list))
#         # print(x_landmark_list)
#         y_landmark_list = list(components[6::2])
#         # print(y_landmark_list)
#         landmark_list = list(zip(x_landmark_list, y_landmark_list))
#         # print(x_landmark_list)
#         print(landmark_list)
#
#     return landmark_list

def get_landmark(label_txt):
    # 不用想一次性将两种数据类型都一下子能分离开来，只要将它们统一成字符串的形式分割开来，然后想要数字是在将其转化过来即可
    data = np.loadtxt(label_txt, dtype=str)
    # print(data)
    # 对对象切片，s是一个字符串，可以通过类似数组索引的方式获取字符串中的字符，同时也可以用s[a🅱️c]的形式对s在a和b之间，
    # 以c为间隔取值，c的值还可以为负，负值则意味着反向取值, 所以这里5::;3意味着从5开始去值一直取完，其间隔值为2
    # method 1： 这个容易理解
    # num_data = len(data[:])
    # for i in range(num_data):
    #     x_landmark = data[i,5::2].astype("float")
    #     # print(x_landmark)
    #     y_landmark = data[i,6::2].astype("float")
    #     landmark = np.array(list(zip(x_landmark, y_landmark)))
    # method 2：这个运行更好
    # 将其变换成（1953,1）的数组形式，再讲其合并，这样的到的数组将会是两两对应数组的元素匹配的数组
    num_landmark_per_group = 21  # 关键点有21个,其中两两配对。
    x_landmark = data[:, 5::2].astype("float").reshape(-1, 1)
    y_landmark = data[:, 6::2].astype("float").reshape(-1, 1)
    landmark = np.hstack((x_landmark, y_landmark))  # （33453,2）
    landmark = landmark.reshape(-1, num_landmark_per_group, 2)  # （1593,21,2）

    # print(landmark)

    return landmark


# gen_landmark(data1_root_file)


def expend_roi(x1, y1, x2, y2, img_w, img_h, ratio=0.25):
    width = x2 - x1 + 1  # +1是因为图像生成的边框宽度也要占一个像素
    height = y2 - y1 + 1  # 这也很上面同理
    # width = torch.IntTensor(width)  # 必须转化成Tensorr才能做torch运算操作
    # height = torch.IntTensor(height)
    # width = torch.abs(width)
    # height = torch.abs(height)

    # 这里是需要扩大后的宽度，即x1扩大,相当于把宽度加长了0.5倍
    padding_width = np.array(width * ratio).astype("int")
    # 这里是需要扩大后的高度， 即y1扩大，相当于把高度加长了0.5倍，这样扩大后的面积因为扩大前的面积的2.25倍
    padding_height = np.array(height * ratio).astype("int")
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    # 超过图像的界限时即等于他的界限
    roi_x1 = np.where(roi_x1 < 0, 0, roi_x1)
    roi_y1 = np.where(roi_y1 < 0, 0, roi_y1)
    # roi_x2[roi_x2 > img_w] = img_w
    # roi_y2[roi_y2 > img_h] = img_h
    # 1. np.where(condition, x, y)
    # 满足条件(condition)，输出x，不满足输出y。
    roi_x2 = np.where(roi_x2>img_w, img_w, roi_x2)
    roi_y2 = np.where(roi_y2>img_h, img_h, roi_y2)
    return roi_x1, roi_y1, roi_x2, roi_y2, roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


# print(expend_roi(229.0, 38.0, 289.0, 99.0, 557, 665))


def get_img_name(label_txt):
    data = np.loadtxt(label_txt, dtype=str)
    img_name = data[:, 0]
    # print(img_name)
    return img_name


# get_img_name(data1_root_file)


# def get_label_values(bboxs,landmarks):
#     data = np.column_stack((bboxs,landmarks))
#     label_values = data[:, 1:]
#     # print(label_values)
#
#     return label_values


# get_label_values(data1_root_file)


def data_split(img_name, label_values):
    '''
    格式：

    X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.3, random_state=0)

    train_data：被划分的样本特征集

    train_target：被划分的样本标签

    test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量

    random_state：是随机数的种子。
    '''
    # train_test_split将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
    # x_train,x_test是子集，y为标签
    x_train, x_test, y_train, y_test = train_test_split(img_name, label_values, test_size=0.3)

    # np.column_stack()可以按列连接两个维数不同的数组
    trainset = np.column_stack((x_train, y_train))
    testset = np.column_stack((x_test, y_test))

    return trainset, testset


# trainset, testset = data_split(get_img_name(data1_root_file), get_label_values(data1_root_file))


def get_txt_file(trainset, testset, folder_dir):
    train_txt = np.savetxt(folder_dir + train_name, trainset, delimiter=" ", fmt="%s")
    test_txt = np.savetxt(folder_dir + test_name, testset, delimiter=" ", fmt="%s")

    return train_txt, test_txt


# gen_txt_file(folder_list[0])




def get_iou(bbox1, bbox2):
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    bbox1_area = (bbox1[2] - bbox1[1]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[1]) * (bbox2[3] - bbox2[1])

    left_up = np.maximum(bbox1[:2], bbox2[:2])  # 这个是因为坐标系的原点是基于左上角的，画一下图就明白了
    right_bottom = np.maximum(bbox2[2:], bbox2[2:])

    # 这里是为了既可以保证有无交接时的高和宽都有值，相当与判断了有无交叉，无交接是宽和高都为负数，即判断其值为0
    inter_section = np.maximum(left_up - right_bottom + 1, 0.0)
    # inter_section[0]为宽+边框的宽度1，inter_section[1]为高+边框的高度1。即交集的面积
    inter_area = inter_section[0] * inter_section[1]
    union_area = bbox1_area + bbox2_area - inter_area  # 并集的面积
    # eps是取非负的最小值。当计算的IOU为0或为负（但从代码上来看不太可能为负），使用np.finfo(np.float32).eps来替换， 当做固定套路
    iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return iou


def channel_norm(img):
    img = img.astype('float32')
    m_mean = np.mean(img)
    m_std = np.std(img)

    print('mean: ', m_mean)
    print('std: ', m_std)

    return (img - m_mean) / m_std

def show_landmarks(folder_dir, img_name, landmarks, expand_bbox_info):
    """Show image with landmarks"""
    imgs = os.path.join(folder_dir, img_name)
    image = Image.open(imgs)
    # img.thumbnail((480, 480))

    # 使用PIL裁切图片使用PIL需要引用Image，使用Image的open(file)方法可以返回打开的图片，使用crop((x0,y0,x1,y1))方法可以对图片做裁切。
    #
    # 区域由一个4元组定义，表示为坐标是 (left, upper, right, lower)，Python Imaging Library 使用左上角为 (0, 0)的坐标系统
    #
    # box(100,100,200,200)就表示在原始图像中以左上角为坐标原点，截取一个100*100（像素为单位）的图像，为方便理解，如下为示意图box（b1,a1,b2,a2）
    image = image.crop((expand_bbox_info[0, 0], expand_bbox_info[0, 1], expand_bbox_info[0, 2], expand_bbox_info[0, 3]))
    plt.imshow(image)
    plt.scatter(landmarks[0, 0::2], landmarks[0, 1::2], s=10, marker='.', c='r')
    plt.pause(10)  # pause a bit so that plots are updated
    plt.show()




def main():
    # 得到数据集I的Bbox的数据
    I_bbox = get_bbox(data1_root_file)
    # 得到数据集II的Bbox的数据
    II_bbox = get_bbox(data2_root_file)
    # I的数据集上图片的名字
    I_img_names = get_img_name(data1_root_file)
    # II的数据集上图片的名字
    II_img_names = get_img_name(data2_root_file)

    I_img_w_list, I_img_h_list = [], []

    II_img_w_list, II_img_h_list = [], []



    for idx in range(len(I_img_names)):
        # 读取I中的图片
        I_img = Image.open(folder_list[0] + I_img_names[idx])

        # 获取I中的图片宽和高的尺寸
        I_img_w, I_img_h = I_img.size
        I_img_w_list.append(I_img_w)
        I_img_h_list.append(I_img_h)

    I_img_w = np.array(I_img_w_list)
    I_img_h = np.array(I_img_h_list)

    for idx in range(len(II_img_names)):
        # 读取II中的图片
        II_img = Image.open(folder_list[1] + II_img_names[idx])

        # 获取II中的图片宽和高的尺寸
        II_img_w, II_img_h = II_img.size
        II_img_w_list.append(II_img_w)
        II_img_h_list.append(II_img_h)

    II_img_w = np.array(II_img_w_list)
    II_img_h = np.array(II_img_h_list)

    # 获取扩大后的bbox的左上角坐标
    I_expand_bbox_info = expend_roi(I_bbox[:, 0], I_bbox[:, 1], I_bbox[:, 2], I_bbox[:, 3], I_img_w, I_img_h)
    I_expand_bbox_info = np.array(I_expand_bbox_info).transpose()
    I_roi_x1 = I_expand_bbox_info[:, 0]
    I_roi_y1 = I_expand_bbox_info[:, 1]

    II_expand_bbox_info = expend_roi(II_bbox[:, 0], II_bbox[:, 1], II_bbox[:, 2], II_bbox[:, 3], II_img_w, II_img_h)
    II_expand_bbox_info = np.array(II_expand_bbox_info).transpose()
    II_roi_x1 = II_expand_bbox_info[:, 0]
    II_roi_y1 = II_expand_bbox_info[:, 1]

    # 获取bbox的两个点的坐标信息
    I_expand_bbox = I_expand_bbox_info[:, :4]
    II_expand_bbox = II_expand_bbox_info[:, :4]

    # 获取关键点坐标
    I_landmarks = get_landmark(data1_root_file)
    II_landmarks = get_landmark(data2_root_file)

    # 以左上角坐标为原点，获取新的关键点的坐标,效果是想将真正有用的部分,是人脸,以及人脸关键点。所以,希望对人脸进行截取
    # 将其扩张为与Landmarks点集shape然后再相减，用debug来调试
    I_top_left_set= np.array([I_roi_x1, I_roi_y1]).transpose()
    I_top_left_set = np.repeat(I_top_left_set, 21, axis=0).reshape((-1, 21, 2))
    I_landmarks -= I_top_left_set
    I_landmarks = I_landmarks.reshape((len(I_landmarks),-1))

    II_top_left_set = np.array([II_roi_x1, II_roi_y1]).transpose()
    II_top_left_set = np.repeat(II_top_left_set, 21, axis=0).reshape((-1, 21, 2))
    II_landmarks -= II_top_left_set
    II_landmarks = II_landmarks.reshape((len(II_landmarks), -1))

    # show_landmarks(folder_list[0], I_img_names[0], I_landmarks, I_expand_bbox_info)  # 查看人脸关键点

    # I的数据集上图片后面的相关数据，即标签
    I_label_values = np.concatenate((I_expand_bbox, I_landmarks), axis= 1)
    # II的数据集上图片后面的相关数据，即标签
    II_label_values = np.concatenate((II_expand_bbox, II_landmarks), axis= 1)
    # 将I数据集划分为trainset和testset
    I_trainset, I_testset = data_split(I_img_names, I_label_values)
    # 将II数据集划分为trainset和testset
    II_trainset, II_testset = data_split(II_img_names, II_label_values)

    # 生成I数据集的train.txt, 生成test.txt
    get_txt_file(I_trainset, I_testset, folder_list[0])
    # 生成II数据集的train.txt, 生成test.txt
    get_txt_file(II_trainset, II_testset, folder_list[1])

    # I_roi_x1, I_roi_y1 = I_expand_bbox_info[0, 0], I_expand_bbox_info[0, 1]
    # I_roi_x2, I_roi_y2 = I_expand_bbox_info[0 ,2], I_expand_bbox_info[0, 3]


if __name__ == "__main__":
    main()
