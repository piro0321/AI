
import os
import re
import glob
import cv2
import pickle
import matplotlib as plt
import numpy as np
from PIL import Image
from mlc.function import show
from keras import backend as K
np.set_printoptions(threshold=400000000)



def show_IoU():
    """
    #   #jpgのみにしてください
    """
    datapath = "./image/precision/"
    filepath = "epicenter_center"

    #labelを持ってくる
    label_path = (datapath+filepath + "/label/*jpg").replace('\n', '')
    files = glob.glob(label_path)    
    image_path = (datapath+filepath + "/image/*png").replace('\n', '')
    fimage = glob.glob(image_path)   

    #ファイル名のみ取得
    pred = os.listdir(datapath+filepath + "/label")


    #取得
    for j, n in enumerate(pred):
        #読み込み
        fpred = ("./unet_result/pred/"+n).replace('\n', '')  
        #for file in files:
        flabel = files[j]

        #print(fpred)
        #print(fimage[j])
        #print(flabel)
        input()
        pred = cv2.imread(fpred,0)
        image = cv2.imread(fimage[j],0)
        label = cv2.imread(flabel,0)

        
       # show(pred)
       # show(image)
       # show(label)

        #os.system('cls')
   
        #precison 計算
        w,h = pred.shape


        label[label >240] = 0
        label[label > 0] = 255

        pred[pred > 250] = 255
        pred[pred <= 250] = 0
        pred_s = np.copy(pred)/255
        label_s = np.copy(label)/255
    
        """
        plabel  :pred label (編集結果画像)
        val     :val  image　(予測したマスク画像)
        label   :acc label　（正解マスク画像）
            -正解データ

    　　    赤 0,0,255
    　　    青 255,0,0
    　　    緑 0,255,0
        """

        TP,FP,FN = 0,0,0

        for x in range(h):
            for y in range(w):
                #正解ラベル
                if label[y,x] == 255:
                    if pred[y,x] == 255:
                        TP +=1
                #検出結果

                #未検出
                if label[y,x] == 255 and pred[y,x] == 0:
                    FP += 1
                #誤検出
                if label[y,x] == 0 and pred[y,x] == 255:
                    FN += 1


        IOU = TP / (TP + FP+ FN) 
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F_value = TP/(TP+(FP+FN)/2)

        with open(root +"/value.txt",mode='a') as f_dataset: 
            f_dataset.write("dataname :"+filename + dataname)
            f_dataset.write("\n IOU      :"+str(IOU))
            f_dataset.write("\n recall   :"+str(recall))
            f_dataset.write("\n precision:"+str(precision))
            f_dataset.write("\n F_value  :"+str(F_value))
            f_dataset.write("\n----------------------------------------------------------------------\n")

        print("\nIOU = ",IOU)
        print("\nrecall = ",recall)
        print("\nprecision = ",precision)
        print("\nF_value = ",F_value)





if __name__ == "__main__":

   show_IoU()
   #movie_to_image(10,videopath)

