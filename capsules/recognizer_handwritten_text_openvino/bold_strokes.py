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
        binary = cv2.erode(binary, kernel, iterations=1)  # 1: the number of iterations, i.e. how many times the dilation is applied
    return binary

def cv_show(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''Horizontal projection'''
def getHProjection(image, image_pad):
    (h, w) = image.shape
    # An array as long as the image height
    #h_ = [0] * h
    first_y = end_y = -1
    # Loop over the rows and count the white pixels in each
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
    # Loop over the columns and count the white pixels in each
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
    # Convert the image to grayscale
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarize the image & bold strokes
    #img = bold_strokes(img)
    # ret, binary = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    if black_groud:
        ret, binary = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, binary = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    #cv_show(binary)

    (h, w) = binary.shape
    #print(f"h={h},w={w}")

    # Horizontal projection
    y1, y2 = getHProjection(binary, image_pad)
    '''
    start = 0
    H_Start = []
    H_End = []
    # Get the vertical split positions from the horizontal projection
    for i in range(len(H)):
        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0

    img = binary
    # Split the rows; after that, split the columns and save the split positions
    for i in range(len(H_Start)):
        # Get the row image
        if H_Start[i] >= image_pad:
            H_Start[i] -= image_pad
        if H_End[i] < h-image_pad:
            H_End[i] += image_pad
        #cropImg = img[H_Start[i]:H_End[i], 0:w]
    '''
        # Project the row image vertically
    x1, x2 = getVProjection(binary[y1:y2, 0:w], image_pad)
    img = gray_image[y1:y2, x1:x2]
    #cv_show(img)

    return img, [x1, y1, x2, y2]

if __name__ == "__main__":
    ipath = "/mnt/hgfs/share/images/handwrite/large"

    if os.path.isdir(ipath):
        images = os.listdir(ipath)
        for image in images:
            file = os.path.join(ipath, image)
            if not os.path.isfile(file):
                continue

            # Read the original image
            origineImage = cv2.imread(file)
            img = get_handwriten_text_area_image(origineImage, 20)
            #cv_show(img)
            list_path = os.path.split(file)
            file_name = os.path.splitext(list_path[1])
            folder = os.path.join(list_path[0],'out')
            if not os.path.exists(folder):
                os.makedirs(folder)

            save_path = os.path.join(folder, file_name[0] + '_1' + file_name[1])

            cv2.imwrite(save_path, img)

    # Convert the image to grayscale
    # image = cv2.imread('test.jpg',0)

