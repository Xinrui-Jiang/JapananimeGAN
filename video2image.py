import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Get the Key Frame from a video
Date:2020-3-26
Author:Zhutian Lin
Github:https://github.com/MiaoMiaoKuangFei
'''


def exponential_smoothing(alpha, s):
    '''
    Primary exponential smoothing
    :param alpha:  Smoothing factor,num
    :param s:      List of data,list
    :return:       List of data after smoothing,list
    '''
    s_temp = [s[0]]
    print(s_temp)
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i - 1] + (1 - alpha) * s_temp[i - 1])
    return s_temp


def precess_image(image):
    '''
    Graying and GaussianBlur
    :param image: The image matrix,np.array
    :return: The processed image matrix,np.array
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # Add Gaussian filter
    return gray_image


def abs_diff(pre_image, curr_image):
    '''
    Calculate absolute difference between pre_image and curr_image
    :param pre_image:The image in past frame,np.array
    :param curr_image:The image in current frame,np.array
    :return:
    '''
    gray_pre_image = precess_image(pre_image)
    gray_curr_image = precess_image(curr_image)
    diff = cv2.absdiff(gray_pre_image, gray_curr_image)
    res, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnt_diff = np.sum(np.sum(diff))
    return cnt_diff


class KeyFrameGetter:
    '''
    Get the key frame
    '''

    def __init__(self, video_path, img_path, window=25):
        '''
        Define the param in model.
        :param video_path: The path points to the movie data,str
        :param img_path: The path we save the image,str
        :param window: The comparing domain which decide how wide the peek serve.
        '''
        self.window = window  # number of keyframes
        self.video_path = video_path  # video path
        self.img_path = img_path  # image storage path
        self.diff = []
        self.idx = []

    def load_diff_between_frm(self, smooth=False, alpha=0.07):
        '''
        Calculate and get the model param
        :param smooth: Decide if you want to smooth the difference.
        :param alpha: Difference factor
        :return:
        '''
        print("load_diff_between_frm")
        cap = cv2.VideoCapture(self.video_path)  # Open video file
        diff = []
        frm = 0
        pre_image = np.array([])
        curr_image = np.array([])

        while True:
            frm = frm + 1

            # success literally indicates success, data is the image data of the current frame; .read reads a frame of image and moves to the next frame
            # cv2.imwrite('./img/' + str(num) + ".jpg", data)
            success, data = cap.read()
            if not success:
                break
            #  here is the main part
            #  record every data
            if frm == 1:
                pre_image = data
                curr_image = data
            else:
                pre_image = curr_image
                curr_image = data
            #  Differentiate
            diff.append(abs_diff(pre_image, curr_image))
            #  Loop in the end

            if frm % 100 == 0:
                print('Detect Frame:', str(frm))
        cap.release()

        if smooth:
            diff = exponential_smoothing(alpha, diff)
        #  normalize data
        self.diff = np.array(diff)
        mean = np.mean(self.diff)
        dev = np.std(self.diff)
        self.diff = (self.diff - mean) / dev

        self.pick_idx()
        return

    def pick_idx(self):
        '''
        Get the index which accord to the frame we want(peek in the window)
        :return:
        '''
        print("pick_idx")
        for i, d in enumerate(self.diff):
            ub = len(self.diff) - 1
            lb = 0
            if not i - self.window // 2 < lb:
                lb = i - self.window // 2
            if not i + self.window // 2 > ub:
                ub = i + self.window // 2

            comp_window = self.diff[lb:ub]
            if d >= np.max(comp_window):
                self.idx.append(i)

        tmp = np.array(self.idx)
        tmp = tmp + 1  # to make up the gap when diff
        self.idx = tmp.tolist()
        print("Extract the Frame Index:" + str(self.idx))

    def save_key_frame(self):
        '''
        Save the key frame image
        :return:
        '''
        print("save_key_frame")
        cap = cv2.VideoCapture(self.video_path)  # open video file
        frm = 0
        idx = set(self.idx)
        while True:
            frm = frm + 1
            success, data = cap.read()
            if not success:
                break
            if frm in idx:
                # print('Extracting idx:'+str(frm))
                cv2.imwrite(self.img_path + '/' + str(frm) + ".jpg", data)
                idx.remove(frm)
            if not idx:
                print('Done！')
                break

    def plot_diff_time(self):
        '''
        Plot the distribution of the difference along to the frame increasing.
        :return:
        '''
        plt.plot(self.diff, '-b')
        plt.plot(np.array(self.idx) - 1, [self.diff[i-1] for i in self.idx], 'or')
        plt.xlabel('Frame Pair Index')
        plt.ylabel('Difference')
        plt.legend(['Each Frame', 'Extract Frame'])
        plt.title("The Difference for Each Pair of Frame")
        plt.plot()
        plt.show()


if __name__ == '__main__':
    sa = []

    for root, dirs, files in os.walk(r"C:/Users/21300/Desktop/Anime_scene_dataset"):
        for file in files:

            source_path = os.path.join(root, file)
            dir_path = 'C:/Users/21300/Desktop/Anime_scene_dataset/' + file.strip('.avi').strip('.mp4') + '/'

            print("The source file directory is", root)
            print("The source file path is", source_path)
            print("The destination file path is", dir_path)

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            kfg = KeyFrameGetter(source_path, dir_path, 100)
            a = time.time()
            kfg.load_diff_between_frm(alpha=0.07)  # load parameters of the model
            b = time.time()
            sa.append(b - a)
            print(sa)
            kfg.save_key_frame()  # Save the corresponding image in the index list