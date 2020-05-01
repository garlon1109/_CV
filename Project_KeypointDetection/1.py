# import os
#
# folder_list = ["I","II"]
#
# # for folder_name in folder_list:
# #     print(folder_name)
# #     folder = os.path.join('data', folder_name)
# #     print(folder)
# #     label_file = os.path.join(folder, 'label.txt')
# #     print(label_file)
#
# # class Number:
# #
# #     def __init__(self, num):
# #         self.num = num
# #
# #     def __str__(self):
# #         return str(self.num)
# #
# #     # 对象出现在'+'的左边时自动触发
# #     def __add__(self, other):
# #         print('__add__')
# #         return self.num + other
#
#
#
# def remove_invalid_image(lines):
#     images = []
#     for line in lines:
#         name = line.split()[0]
#         if os.path.isfile(name):
#             images.append(line)
#     return images
#
# def load_metadata():
#     tmp_lines = []
#     # folder_list = ['I', 'II'], 所以 folder_name 第一次值为I, 第二次值为II。
#     for folder_name in folder_list:
#         # folder 就会变为 "data\I", "data\II"
#         folder = os.path.join('data', folder_name)
#         metadata_file = os.path.join(folder, 'label')
#         with open(metadata_file) as f:
#             lines = f.readlines()
#             print(lines)
#         # __add__，表示‘左加’，若对象在+右边则会报错TypeError
#         tmp_lines.extend(list(map((folder + '/').__add__, lines)))
#     # print(tmp_lines)
#     res_lines = remove_invalid_image(tmp_lines)
#     # print(res_lines)
#     return res_lines
#
# load_metadata()
# import numpy as np
# from PIL import Image
#
# # # print(np.array([4, 5, 6]) > np.array([7, 8, 9]))
# a = np.array([4, 5, 6, -1])
# b = np.array([3, 6, 1, 3])
# # c = np.zeros(len(a))
# # from PIL import Image
# #
# # # a[a < np.array([3, 6, 1, 3])]
# #
# # # for idx, m in enumerate(a<b):
# # #     if a<b
# #
# # a = np.where(a<b,a,b)
# # print(a)
# #
# m = []
#
# for i in a:
#     m.append()

# def get_img_size():
#     I_img_w_list, I_img_h_list = [], []
#     roi_x1, roi_y1 = [], []
#
#     for idx in range(len(I_img_names)):
#         # 读取图片
#         I_img = Image.open(folder_list[0] + I_img_names[idx])
#         # 获取图片宽和高的尺寸
#         I_img_w, I_img_h = I_img.size
#         I_img_w_list.append(I_img_w)
#         I_img_h_list.append(I_img_h)
#     I_img_w = np.array(I_img_w_list)
#     I_img_h = np.array(I_img_h_list)

import numpy as np
import pandas as pd
