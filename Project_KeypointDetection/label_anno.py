import pandas as pd
import os
import cv2
import numpy as np


class Data_annotation:
    def __init__(self, root):
        self.root = root

    def get_faces_landmarks(self):

        FOLDER = ['I', 'II']
        # 将其变为字典形式，便于下面pandas.DataFrame的格式分类
        DATA_INFO = {'path': [], 'face_bbox': [], 'face_landmarks': []}
        tmp_lines = []

        for f in FOLDER:
            DATA_DIR = os.path.join(self.root, f)
            FILE_PATH = os.path.join(DATA_DIR,'label.txt')
            with open(FILE_PATH) as fd:
                lines = fd.readlines()
            # extend 只接受list，并且把list中的元素从列表取出，添加到同一个列表中，参考请按
            # https://blog.csdn.net/u012736685/article/details/88799588?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522158677465119724843308864%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=158677465119724843308864&biz_id=0&utm_source=distribute.pc_search_result.none-task-blog-soetl_so_first_rank_v2_rank_v25-1
            tmp_lines = tmp_lines.extend(list(map(lambda x: os.path.join(DATA_DIR,x),lines)))

        for file in tmp_lines:
            file = file.strip().split()


        # # I数据集上的图片路径的拼接
        # DATA1_DIR = self.root + FOLDER[0] + '/'
        # FILE1_PATH = DATA1_DIR + 'label.txt'
        # FILE1_PATH = pd.read_csv(FILE1_PATH, header=None, sep=' ')
        # # II数据集上的图片路径的拼接
        # DATA2_DIR = self.root + FOLDER[1] + '/'
        # FILE2_PATH = DATA2_DIR + 'label.txt'
        # FILE2_PATH = pd.read_csv(FILE2_PATH, header=None, sep=' ')
        # # I 和 II 汇总得到的图片路径凭接
        # FILE_PATH = pd.concat([FILE1_PATH,FILE2_PATH],sort=False)
        #
        # for i in range(FILE_PATH.shape[0]):
        #     try:
        #         #  读取图像，解决imread不能读取中文路径的问题
        #         cv2.imdecode(np.fromfile(FILE_PATH.iloc[i, 0], dtype=np.uint8), cv2.IMREAD_COLOR)
        #     except OSError:
        #         pass
        # print(FILE_PATH.columns)
        # def add_col_name(file):
        #     if col_name==0:
        #         col_name = 'path'
        #     if col_name == range(1,5)
        #
        # # DATA_INFO['path']=FILE_PATH.iloc[:, 0]
        # # DATA_INFO['face_bbox']=FILE_PATH.iloc[:, 1:5]
        # # DATA_INFO['face_landmarks']=FILE_PATH.iloc[:, 5:]
        # # ANNOTATION = pd.DataFrame(DATA_INFO)
        # # ANNOTATION.to_csv('face_keypoints_annotation.csv')


        print('face_keypoints_annotation file is saved.')






            # file = os.path.join(DATA_DIR, img_names)


        # 这里的try命令是为了能够测试是否可以运行以下命令, except命令则是如果出错即跟在except命令后的OSError，出现这种类型的错误，他就会执行以下命令
        # and else命令则是为了如果可以执行try以后的命令的话，这就可以执行else以后的命令
        # try:
        #     cv2.imdecode(np.fromfile(file[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        # except OSError:
        #     pass
        #
        # else:
        #     #
        #     DATA_INFO['path'] = file[:, 0]
        #     DATA_INFO['face_bbox'] = file[:, 1:5]
        #     DATA_INFO['face_landmarks'] = file[:, 5:]

        # 生成DataFrame形式，并将文件转换并储存为csv格式
        ANNOTATION = pd.DataFrame(DATA_INFO)
        ANNOTATION.to_csv('face_landmark_annotations')
        print('face_landmark_annotations file is saved')

    def get_vaild_data(self, DATA_INFO, data_info_anno, expand_ratio=0.25):
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
            roi_x2 = np.where(roi_x2 > img_w, img_w, roi_x2)
            roi_y2 = np.where(roi_y2 > img_h, img_h, roi_y2)
            return [roi_x1, roi_y1, roi_x2, roi_y2]

        def get_landmark(file):
            # 不用想一次性将两种数据类型都一下子能分离开来，只要将它们统一成字符串的形式分割开来，然后想要数字是在将其转化过来即可
            data = np.loadtxt(file, dtype=str)
            # print(data)
            # 对对象切片，s是一个字符串，可以通过类似数组索引的方式获取字符串中的字符，同时也可以用s[a🅱️c]的形式对s在a和b之间，
            # 以c为间隔取值，c的值还可以为负，负值则意味着反向取值, 所以这里5::;3意味着从5开始去值一直取完，其间隔值为2
            x_landmark = data[:, 5::2].astype("float").reshape(-1, 1)
            y_landmark = data[:, 6::2].astype("float").reshape(-1, 1)
            landmark = np.hstack((x_landmark, y_landmark))  # （33453,2）
            landmark = landmark.reshape(-1, num_landmark_per_group, 2)  # （1593,21,2）

            # print(landmark)

            return landmark

    # 生成训练和测试集
    def get_train_test_set(self):
        FILE = 'face_landmark_annotations.csv'
        DATA_INFO = pd.read_csv(FILE)
        # 将其变为字典形式，便于下面pandas.DataFrame的格式分类
        data_info_anno = {'path': [], 'bbox': [], 'landmarks': []}
        # 这个与上面的expand_ratio值可一尝试换一下，这是另一个选择
        expand_ratio = 0.2


if __name__ == "__main__":
    ROOTS = "./data/"
    data_anno = Data_annotation(ROOTS)
    data_anno.get_faces_landmarks()
    data_anno