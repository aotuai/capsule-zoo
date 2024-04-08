import copy
import json
#import logging

import cv2
import math
import os
import numpy as np
from enum import Enum
import time


class Line_Mode(Enum):
    horizonal = 1
    vertical = 2
    max_angle = 0
    '''
    min_hline_space = 8
    min_vline_space = 9
    hline_gap = 10
    vline_gap = 10
    min_rect_width = 537
    min_rect_high  = 322
    hw_block_col = 1
    hw_block_row = 3
    '''

take_x1 = lambda arg1: arg1[0]
take_x2 = lambda arg1: arg1[2]
take_y1 = lambda arg1: arg1[1]
take_y2 = lambda arg1: arg1[3]
# take_xy = lambda arg1: arg1[0],arg1[1]
# take_yx = lambda arg1: arg1[1],arg1[0]
get_hline_length = lambda arg1: take_x2(arg1) - take_x1(arg1)
get_vline_length = lambda arg1: take_y2(arg1) - take_y1(arg1)
hline_in_vlines = lambda arg1, arg2, arg3: take_x1(arg1) >= arg2 and take_x2(arg1) <= arg3

class detect_lines():
    def __init__(self, frame, options):
        self.frame = frame
        self.split_mode = options["split_mode"]
        self.min_line_spacing = options["min_line_spacing"]
        self.min_vertical_spacing = options["min_vertical_spacing"]
        self.lines_pad = options["lines_pad"]
        self.vertical_pad = options["vertical_pad"]
        self.min_rect_width = options["min_rect_width"]
        self.min_rect_height = options["min_rect_height"]
        #self.ic_block = options["ic_block"],
        #self.face_block = options["face_block"]
        #self.signature_block = options["signature_block"]
        #self.ic_back_block = options["ic_back_block"]
        #self.reception_block = options["reception_block"]

        self.frame_h, self.frame_w = frame.shape[0:2]

    @staticmethod
    def calc_angle(p1, p2):
        if p2[0] == p1[0]:
            return 90
        x = (p2[1] - p1[1]) / (p2[0] - p1[0])
        if x >= 0:
            angle = np.rad2deg(np.arctan(x))
        else:
            angle = 180 + np.rad2deg(np.arctan(x))
        return angle

    @staticmethod
    def check_angle(angle):
        max_angle = Line_Mode.max_angle.value
        angle = abs(angle)
        if (angle >= 0 and angle <= 0 + max_angle) or (angle > 360 - max_angle and angle <= 360) \
                or (angle >= 180 - max_angle and angle <= 180 + max_angle):
            return Line_Mode.horizonal.value
        if (angle >= 90 - max_angle and angle <= 90 + max_angle) or (
                angle > 270 - max_angle and angle <= 270 + max_angle):
            return Line_Mode.vertical.value
        return 0

    @staticmethod
    def check_minLength(p1, p2, min):
        a = abs(p2[1] - p1[1])
        b = abs(p2[0] - p1[0])
        if max(a, b) >= min:
            return True
        return False

    @staticmethod
    def check_maxLength(p1, p2, maxl):
        a = abs(p2[1] - p1[1])
        b = abs(p2[0] - p1[0])
        if max(a, b) >= maxl:
            return True
        return False
    @staticmethod
    def take_xy(list):
        return list[0], list[1]
    @staticmethod
    def take_yx(list):
        return list[1], list[0]


    def sort_list(self, lists, h_flag):
        if h_flag == Line_Mode.horizonal.value:
            for list in lists:
                if list[0] > list[2]:
                    list[0], list[2] = list[2], list[0]
            lists.sort(key=lambda x: self.take_xy(x))
            self.aligned_line(lists, 0)
            self.aligned_line(lists, 2)
            lists.sort(key=lambda x: self.take_xy(x))

        elif h_flag == Line_Mode.vertical.value:
            for list in lists:
                if list[1] > list[3]:
                    list[1], list[3] = list[3], list[1]
            lists.sort(key=lambda x: self.take_yx(x))
            self.aligned_line(lists, 1)
            self.aligned_line(lists, 3)
            lists.sort(key=lambda x: self.take_yx(x))

        return lists

    def multi_line(self, line1, line2, h_flag):
        if h_flag == Line_Mode.horizonal.value:
            if abs(line2[0] - line1[0]) <= self.lines_pad and abs(
                    line2[2] - line1[2]) <= self.lines_pad and \
                    abs(line2[1] - line1[1] - 1) <= self.min_line_spacing:
                return True
            return False
        else:
            if abs(line2[1] - line1[1]) <= self.vertical_pad and abs(
                    line2[3] - line1[3]) <= self.vertical_pad and \
                    abs(line2[0] - line1[0] - 1) <= self.min_vertical_spacing:
                return True
            return False

    @staticmethod
    def marge_one_line(lists, marge, h_flag):
        sum = len(marge)
        if sum <= 1:
            return
        line1 = lists[marge[0]]
        line2 = lists[marge[sum - 1]]
        if h_flag == Line_Mode.horizonal.value:
            line1[1] = line1[3] = int((line2[1] + line1[1]) / 2)
        else:
            line1[0] = line1[2] = int((line2[0] + line1[0]) / 2)

        for i in range(sum - 1):
            del lists[marge[1]]

    def remove_multiline(self, lists, h_flag):
        lines = len(lists)
        marge = []
        top = i = 0
        while top + 1 < lines:
            j = i + 1
            if j < lines:
                if self.multi_line(lists[i], lists[j], h_flag):
                    if i not in marge:
                        marge.append(i)
                    marge.append(j)
                    i += 1
                else:
                    if len(marge) >= 2:
                        self.marge_one_line(lists, marge, h_flag)
                        lines = len(lists)
                    top += 1
                    i = top
                    marge.clear()
            else:
                if len(marge) >= 2:
                    self.marge_one_line(lists, marge, h_flag)
                break
    def get_image_mlen(self, h_flag):
        if h_flag == Line_Mode.horizonal.value:
            x = self.frame_w
        else:
            x = self.frame_h
        return int(x * 2 / 3), int(x / 6)

    def lsd_detect_line(self, image, h_flag):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #lsd = cv2.createLineSegmentDetector(0)
        #lines = lsd.detect(gray)

        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(gray)
        #h, w = image.shape[:2]
        # 创建一个新图像显示线条
        # line_image = np.copy(image) * 0  # 创建一个与原图像大小相同的空白图像

        # 画出线条
        line_list = []

        if lines is not None:
            for line in lines:
                line[0].tolist()
                x1 = int(round(line[0][0]))
                y1 = int(round(line[0][1]))
                x2 = int(round(line[0][2]))
                y2 = int(round(line[0][3]))

                angle = self.calc_angle((x1, y1), (x2, y2))
                angle_ok = self.check_angle(angle)
                maxl, minl = self.get_image_mlen(h_flag)
                if angle_ok == h_flag:
                    if self.check_minLength((x1, y1), (x2, y2), minl):
                        line_list.append([x1, y1, x2, y2])

            self.sort_list(line_list, h_flag)
            # add_hv(line_list, hv, h_flag)
            self.remove_multiline(line_list, h_flag)

        return line_list

    '''
    def get_hline_space(line1, line2):
        return line2[1] - line1[1]

    def check_rectamgle_hline(h_lines):
        h_total = len(h_lines)
        h1 = 0
        v1 = 0
        rect_hline = []
        while True:
            h2 = h1 + 1
            v2 = v1 + 1
            line1 = h_lines[h1]
            line2 = h_lines[h2]
            if get_hline_length(line1) == get_hline_length(line2) and line1[0] == line2[0]:
                if get_hline_space(line1, line2) >= Line_Mode.line_space.value:
                    rect = [line1[0], line1[1], line2[2], line2[3]]
                    # rect_hline.append([i,j])
    '''

    def get_main_outline(self, hlines, vlines, hmax, vmax):
        h_outline = []
        v_outline = []
        h_conut = v_count = 0
        for i in hlines:
            len1 = get_hline_length(i)
            if len1 > hmax:
                h_outline.append(i)
                h_conut += 1
        for i in vlines:
            len1 = get_vline_length(i)
            if len1 > vmax:
                v_outline.append(i)
                v_count += 1

        rects = []
        if v_count >= 1 and h_conut == 0:
            mode = 0
            v_outline.sort(key=lambda x: self.take_xy(x))
            idx = len(v_outline) - 1
            i = 0

            line = copy.deepcopy(hlines[0])
            hlines.insert(0, line)
            hlines[0][1] = hlines[0][3] = 0

            line = copy.deepcopy(hlines[-1])
            hlines.append(line)
            hlines[-1][1] = hlines[-1][3] = self.frame_h -1
            #logging.warning(hlines)

            while i < idx:
                line1 = v_outline[i]
                line2 = v_outline[i + 1]
                if take_x1(line2) - take_x1(line1) >= self.min_rect_width:
                    h_len = len(hlines) - 1
                    j = 0
                    while j < h_len:
                        hl1 = hlines[j]
                        hl2 = hlines[j + 1]
                        if hline_in_vlines(hl1, take_x1(line1), take_x1(line2)):
                            if hline_in_vlines(hl2, take_x1(line1), take_x1(line2)):
                                if take_y1(hl2) - take_y1(hl1) >= self.min_rect_height:
                                    rects.append([take_x1(line1), take_y1(hl1), take_x2(line2), take_y2(hl2)])
                            else:
                                if self.frame_h-take_y1(hl1) >= self.min_rect_height:
                                    rects.append([take_x1(line1), take_y1(hl1), take_x2(line2), self.frame_h-1])
                        else:
                            if hline_in_vlines(hl2, take_x1(line1), take_x1(line2)):
                                if take_y1(hl2) >= self.min_rect_height:
                                    rects.append([take_x1(line1), 0, take_x2(line2), take_y2(hl2)])
                        j += 1
                i += 1
        return rects

    def aligned_line(self, lines, idx):
        if idx > 3:
            return
        i = j = 0
        num = len(lines)
        while j+1 < num:
            j = i + 1
            l1 = lines[i]
            l2 = lines[j]
            if abs(l2[idx]-l1[idx]) <= self.lines_pad:
                k = j
                while k+1 < num:
                    k += 1
                    l3 = lines[k]
                    if abs(l3[idx] - l1[idx]) > self.lines_pad:
                        break
                for ll in range(i,k-1):
                    lines[ll][idx] = lines[k-1][idx]
                i = k
            else:
                i += 1


    def get_rectangles(self, h_lines, v_lines):
        hmax, hmin = self.get_image_mlen(Line_Mode.horizonal.value)
        vmax, vmin = self.get_image_mlen(Line_Mode.vertical.value)
        outlines = self.get_main_outline(h_lines, v_lines, hmax, vmax)
        return outlines

    def detect_lines(self):
        # 转换为灰度图
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        # 转为二值图
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 膨胀算法的色块大小
        h, w = binary.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)

        # 白底黑字，膨胀白色横向色块，抹去文字和竖线，保留横线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        hors = ~cv2.dilate(binary, kernel, iterations=1)  # 迭代两次，尽量抹去文本横线，变反为黑底白线
        h_lines = self.lsd_detect_line(hors, Line_Mode.horizonal.value)

        # 白底黑字，膨胀白色竖向色块，抹去文字和横线，保留竖线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        verts = ~cv2.dilate(binary, kernel, iterations=1)  # 迭代两次，尽量抹去文本竖线，变反为黑底白线
        v_lines = self.lsd_detect_line(verts, Line_Mode.vertical.value)

        return h_lines, v_lines


if __name__ == "__main__":

    image_path = '/mnt/hgfs/share/images/split'
    start_time = time.time()
    combo_image_all = None


    for filename in os.listdir(image_path):
        file = os.path.join(image_path, filename)
        if not os.path.isfile(file):
            continue
        image = cv2.imread(file)
        options = {
            "split_mode": 0,
            "min_line_spacing": 12,
            "min_vertical_spacing": 12,
            "lines_pad": 10,
            "vertical_pad": 10,
            "min_rect_width": 537,
            "min_rect_height": 300,
            "hw_block_col": 1,
            "hw_block_row": 3
        }
        print(f"get lines from {file}")
        det_lines = detect_lines(image, options)

        # combo_image, hors, verts = detect_lines(file)
        hors, verts = det_lines.detect_lines()

        line_image = np.copy(image) * 0  # 创建一个与原图像大小相同的空白图像
        for line in hors:
            cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)
        for line in verts:
            cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)

        rect = det_lines.get_rectangles(hors, verts)

        list_path = os.path.split(file)
        file_name = os.path.splitext(list_path[1])
        folder = list_path[0] + '/out2'
        if not os.path.exists(folder):
            os.makedirs(folder)

        save_path1 = os.path.join(folder, file_name[0] + '_1.json')
        save_path2 = os.path.join(folder, file_name[0] + '_2.json')
        with open(save_path1, 'w') as f:
            json.dump(hors, f)
        with open(save_path2, 'w') as f:
            json.dump(verts, f)
        save_path3 = os.path.join(folder, file_name[0] + '_3' + file_name[1])
        # save_path4 = os.path.join(folder, file_name[0]+'_4'+file_name[1])
        # save_path5 = os.path.join(folder, file_name[0]+'_5'+file_name[1])
        # cv2.imwrite(save_path1, gray)
        # cv2.imwrite(save_path2, binary)
        # cv2.imwrite(save_path1, hors)
        # cv2.imwrite(save_path2, verts)
        cv2.imwrite(save_path3, line_image)

        #print(f"get lines from {file}")

    # save_path = os.path.join(image_path+'/out2', 'all.png')
    # cv2.imwrite(save_path, combo_image_all)
    end_time = time.time()
    end_time = end_time - start_time

    print(f"total time: {end_time}")
