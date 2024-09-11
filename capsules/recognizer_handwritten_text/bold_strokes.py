import logging

import cv2
import numpy as np
import os

def bold_strokes(gray_image, black_groud, bold):
    if black_groud:
        ret, binary = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, binary = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)

    if bold:
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)  # 1:迭代次数，也就是执行几次膨胀操作
    return binary

def cv_show(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''水平投影'''
def getHProjection(image, image_pad):
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    #h_ = [0] * h
    first_y = end_y = -1
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0:
                first_y = y - image_pad if y>=image_pad else 0
                break
        if first_y != -1:
            break
    if first_y != -1:
        for y in range(h-1, -1, -1):
            for x in range(w):
                if image[y, x] == 0:
                    end_y = y + image_pad if y + image_pad < h else h-1
                    break
            if end_y != -1:
                break
    else:
        first_y = 0
        end_y = h-1

    #cv_show(hProjection)
    return first_y, end_y


def getVProjection(image, image_pad):
    (h, w) = image.shape
    first_x = end_x = -1
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 0:
                first_x = x-image_pad if x>=image_pad else 0
                break
        if first_x != -1:
            break

    if first_x != -1:
        for x in range(w-1, -1, -1):
            for y in range(h):
                if image[y, x] == 0:
                    end_x = x+image_pad if x+image_pad < w else w-1
                    break
            if end_x != -1:
                break
    else:
        first_x = 0
        end_x = w-1
    return first_x, end_x


def get_handwriten_text_area_image(gray_image, image_pad, black_groud):
    # 图像灰度化
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将图片二值化 & bold strokes
    #img = bold_strokes(img)
    # ret, binary = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    if black_groud:
        ret, binary = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, binary = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    #cv_show(binary)

    (h, w) = binary.shape
    #print(f"h={h},w={w}")

    # 水平投影
    y1, y2 = getHProjection(binary, image_pad)

    # 对行图像进行垂直投影
    x1, x2 = getVProjection(binary[y1:y2, 0:w], image_pad)
    #img = gray_image[y1:y2, x1:x2]
    #cv_show(img)

    #return img, [x1, y1, x2, y2]
    return [np.array([[x1, y1], [x2,y1], [x2,y2], [x1,y2]]).astype(dtype="float32")]

def check_black_backgroud(gray_image):
    # 二值化图像
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # 统计黑白像素数量
    black_pixels = np.sum(gray_image == 0)
    white_pixels = np.sum(gray_image == 255)

    # 判断背景颜色
    if black_pixels > white_pixels:
        return True  # 黑底
    else:
        return False  # 白底

def get_hw_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    black_groud = check_black_backgroud(gray)

    #logging.warning(f"gray h={gray.shape[0]} w={gray.shape[1]} pad={options['strokes_pad']}")
    box = get_handwriten_text_area_image(gray, 10, black_groud)
    return box

if __name__ == "__main__":
    ipath = "/mnt/hgfs/share/images/handwrite/large"

    if os.path.isdir(ipath):
        images = os.listdir(ipath)
        for image in images:
            file = os.path.join(ipath, image)
            if not os.path.isfile(file):
                continue

            # 读入原始图像
            origineImage = cv2.imread(file)
            img = get_handwriten_text_area_image(origineImage, 20, False)
            #cv_show(img)
            list_path = os.path.split(file)
            file_name = os.path.splitext(list_path[1])
            folder = os.path.join(list_path[0],'out')
            if not os.path.exists(folder):
                os.makedirs(folder)

            save_path = os.path.join(folder, file_name[0] + '_1' + file_name[1])

            cv2.imwrite(save_path, img)

    # 图像灰度化
    # image = cv2.imread('test.jpg',0)

