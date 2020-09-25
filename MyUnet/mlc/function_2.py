import os
import re
import glob
import cv2
import pickle
import matplotlib as plt
import numpy as np
import random
from scipy import ndimage as ndi
from mlc.io import read_raw, vtk_to_numpy, shiftscale_uchar, numpy_to_vtk,write_raw
from mlc.function import load_image,load_pickle
from PIL import Image
from keras import backend as K
np.set_printoptions(threshold=400000000)



def true_positive(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true * y_pred, 1), K.floatx()))

def true_negative(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true + y_pred, 0), K.floatx()))

def false_positive(y_true, y_pred):
    return K.sum(K.cast(K.less(y_true, y_pred), K.floatx()))

def false_negative(y_true, y_pred):
    return K.sum(K.cast(K.greater(y_true, y_pred), K.floatx()))

def IoU(y_true, y_pred):
    y_pred = K.round(y_pred)
    return true_positive(y_true, y_pred) / (false_negative(y_true, y_pred)+true_positive(y_true, y_pred)+false_positive(y_true, y_pred))


def check_existfile(path):
    if os.path.exists(path) == False :
       os.makedirs(path)

#領域抽出
def Extraction():
    path_1 = "D:/Users/takami.h/Desktop/Winf2019/image/method/0037.jpg"
    path_2 = "D:/Users/takami.h/Desktop/Winf2019/image/method/0026_label.jpg"    

    image_1 = cv2.imread(path_1,-1) #original
    label = cv2.imread(path_2,0)#label


    label = label[:, :, np.newaxis]
    label[label > 250] = 0
    label[label > 0] = 1

    label_2 = np.concatenate((label,label),axis=2)
    label = np.concatenate((label_2,label),axis=2)

    im = label*image_1

    cv2.imwrite("D:/Users/takami.h/Desktop/Winf2019/image/method/extraction.jpg",im)

def show(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def movie_to_image(num_cut,dataname):
    """
    動画から画像に変換
    (保存先->動画が置いてあるディレクトリに動画名ファイル作成し保存)
    params
    ----------------------------------
    num_cut :  int
        -間引くフレームの数
    dataname    :string
        -動画名
    """
    video_path = "E:/image/ultrasonics/JSUM-20181003/moving_image/1_hemangioma/" + dataname + ".avi"     #動画のパス
    output_path = "E:/output/JSUM-20181003/" + dataname  #保存先パス

    print("E:/output/JSUM-20181003/" + dataname )

    check_existfile(output_path)

    capture = cv2.VideoCapture(video_path)

    img_count = 1       #保存する候補画像数
    frame_count = 0     #読み込んだフレーム数

    #フレーム画像がある限るループ
    while(capture.isOpened()):
        print("\r{0:d}".format(img_count),end="")
        ret,frame = capture.read()
        if ret == False:
            break

        if frame_count % num_cut == 0:
            img_filename = output_path + "/{0:04d}.".format(img_count) + "jpg"
            cv2.imwrite(img_filename,frame)
            img_count +=1
        frame_count += 1

    capture.release()


if __name__ == "__main__":

   movie_to_image(1,"image11_with_tumor")
   #Extraction()
   #test_1()
