"""
# @file name        : img_primary_filter.py
# @created  by      : Mr.Lee
# @modified by      : RangeKing
# @creation date    : 2021-05-13
# @modification date: 2021-11-17
# @brief            : 初级图片筛选
"""
import re
import os
import cv2
import shutil
import numpy as np

# 判断图片是彩色还是黑白
def is_gray_img(img):    
    b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
    r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])

    b_2_g = cv2.compareHist(b_hist, g_hist, cv2.HISTCMP_CORREL)
    g_2_r = cv2.compareHist(g_hist, r_hist, cv2.HISTCMP_CORREL)
    r_2_b = cv2.compareHist(r_hist, b_hist, cv2.HISTCMP_CORREL)

    CORREL_ABS = np.abs(b_2_g - g_2_r) + \
                 np.abs(g_2_r - r_2_b) + \
                 np.abs(r_2_b - b_2_g)

    if CORREL_ABS < 0.003:  # 黑白彩色图像的直方图很相似
        return True
    else:
        return False

# 检测输入图像是否需要
def check_img(img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)

    # 文件信息 file info
    file_size = os.path.getsize(img_path)
    img_height, img_width = img.shape[:2]
    if file_size < 10 * 1024 or img_width < 256 or img_height < 256:
        return False

    # 图片梯度 image basic feature
    img_dy = img[:img_height-1] - img[1:]
    img_dx = img[:, :img_width-1] - img[:, 1:]
    img_gradient = np.mean(np.abs(img_dx)) + np.mean(np.abs(img_dy))
    print(img_path, "img_gradient =", img_gradient)
    if img_gradient < 50:
        return False

    # 图片是否为黑白图片
    if is_gray_img(img):
        print(img_path, " gray")
        return False

    return True


if __name__ == '__main__':
    root_dir_list = ["../Image-Downloader-master/download_images/dog",\
                     "../Image-Downloader-master/download_images/cat"]
    file_suffix = "jpeg|jpg|png"
    for root_dir in root_dir_list:
        remove_dir = os.path.join(root_dir,"remove")
        if not os.path.exists(remove_dir):
            os.makedirs(remove_dir)
        for img_name in os.listdir(root_dir):
            # 对处理文件的类型进行过滤
            if re.search(file_suffix, img_name) is None:
                continue
            img_path = os.path.join(root_dir,img_name)
            if not check_img(img_path):
                output_path = os.path.join(remove_dir,img_name)
                shutil.move(img_path, output_path)
