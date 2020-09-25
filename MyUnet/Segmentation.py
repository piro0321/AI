

# -*- coding: utf-8 -*-
import os
import re
import cv2
import keras
import csv
import sys
import numpy as np
from utils.self_unet import unet_2d, unet_2d_GAM
import tensorflow as tf

def show(image):
    """
    #   画像表示用関数
    Parameter
    ---------
    image   :numpy型
            -numpyのimage
    result
    ------
    画像表示
    """
    cv2.imshow("test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Segmentation(image,model,target_size = (640,640,1)):

    target = cv2.resize(image,(target_size[0],target_size[1]))
    target = target.reshape((1,target_size[0],target_size[1],1))
    target = target / 255

    pred = model.predict(target)
    pred = np.reshape(pred, (target_size[0],target_size[1],1), order='F')
    
    Threshold = 0.7
    pred[pred >= Threshold] = 1
    pred[pred < Threshold] = 0

    pred= np.asarray(pred, dtype=np.uint8)
    pred = cv2.resize(pred,(image.shape[1],image.shape[0]))

    return pred*255


#渡すよう
def tumor_validation(raw_image,seg_img,annotation,cnn_model):
    
    print(seg_img.shape)
    tumor_coord = np.zeros(seg_img.shape)
    for n,coord in enumerate(annotation):
        #座標受け取り
        #x1,x2,y1,y2 = coord
        tumor_coord[y1:y2,x1:x2] = 1
        result = tumor_coord * seg_image

        tumor_area= np.count_nonzero(tumor_coord >=1)#腫瘍領域の面積    
        result_area = np.count_nonzero(result >= 1)#重複する面積

        Inclusion_rate = result_area / tumor_area
        print("Inclusion_rate   :",Inclusion_rate)
        #CNNの処理


def test_amed():
    np.set_printoptions(threshold=400000000)
    pred = cv2.imread("./image/test/test.png",cv2.IMREAD_GRAYSCALE)
    print(pred.shape)
    tumor_coord = np.zeros(pred.shape)
    x1,x2,y1,y2  = 200,500,200,500
    show(tumor_coord)
    #座標受け取り
    #x1,x2,y1,y2 = coord
    tumor_coord[y1:y2,x1:x2] = 1
    result = tumor_coord * pred

    tumor_area= np.count_nonzero(tumor_coord >= 1)#腫瘍領域の面積
    result_area = np.count_nonzero(result >= 1)#重複する面積


    Inclusion_rate = result_area / tumor_area
    print(Inclusion_rate)
    show(pred)
    show(result)
    input()

if __name__ == "__main__":
    
    test_amed()

    input()
    target_size = (640,640,1)
    #model
    seg_model = unet_2d_GAM(input_shape=target_size)
    seg_model.load_weights("./model/unet_2d_weights.hdf5")

    Segmentation(image,seg_model,target_size)


