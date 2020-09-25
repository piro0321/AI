# -*- coding: utf-8 -*-

    #   # 前処理・後処理の機能をまとめたファイル
"""
#   dataset作成   #
pklデータの連結等
"""
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
from PIL import Image
from mlc.function import make_pickle,stop,check_existfile,concatenate_pkl,Read_Parameter


##  メインの関数

def save_sizedata(size,dataname):
    """
    元画像のサイズを保存する
    Parameter
    ---------
    size    :numpyのlist
            -超音波画像
    label   :numpyのlist
            -ラベル画像
    color   :tuple
            -色の数値　例）blue = (255,0,0)
    epsilon :float型 
            -輪郭の近似値,値が低いほど細かく近似する

    result
    -------
    image   :original image
    label   :輪郭描写後の画像
    """

        #datasetの詳細をテキストで保存
    with open(root_path + "sizedata.txt",mode='w') as f_dataset: 
        for t in range(len(task)):
            f_dataset.write(dataname[t]+ ":" +size[t] )

#pklデータ作成
def make_dataset(root,task,size):
    """
    #   #   Dataset　の作成
    -//aka/share/amed/amed_liver_labelのデータを使用

    Parameter
    -----------
    root    :strings
            -データセットを作成するためのデータまでのパス
    task    :list型
            -データセット作成で使用するディレクトリ名を格納したリスト
    size    :tuple型
            -データセットの画像サイズ

    Result
    --------
    「./image」にデータセット作成
     -image.pkl
     -label.pkl
    """
    x,y = size
    num_list = []
    image_size,image_name = [],[]
    mode = "image"
    for s in range(len(task)):

        path = Path(root + task[s])
        print("\n\n"+task[s])
        for k in range(2):#image,label
            list = []
            z = 0
            paths = path[k]+ '/*.jpg' 
            files = glob.glob(paths)
            if not files:
                paths = path[k]+ '/*.png' 
                files = glob.glob(paths)
            a = path[k + 2].split('/')

            for p in range(len(a)):
                if a[p] in ["image"]:
                    mode = "image"
                if a[p] in ["label"]:
                    mode = "label"
                else:
                    continue

            for f in files:

                z = z + 1

                image = Image.open(f)
                image = image.convert('L')
                
                image_name.append(os.path.basename(f))
                image_size.append(image.size)

                image_resize = image.resize((x, y))
                image = np.array(image_resize)

                #check image write
                if mode == "image":
                    cv2.imwrite(path[k+2]+"/{0:04d}.jpg".format(z) , image)
                if mode == "label":
                    #image = np.where(image > 0,255,0)# 条件満たす 255 それ以外 0
                    #image = np.where(image == 255,0,255)# 条件満たす 255 それ以外 0
                    image[image > 200] = 0
                    image[image > 0] = 255
                    cv2.imwrite(path[k+2]+"/{0:04d}.jpg".format(z) , image)

                image = np.expand_dims(image,axis=-1)
                #data type convert (image => float   :  label => int)
                if mode == "image":
                    image = np.asarray(image, dtype=np.uint8)
                    addPath = "/image.pkl"
                if mode == "label":
                    image = np.asarray(image, dtype=np.uint8)
                    addPath = "/label.pkl"
                list.append(image)
                print("\r{0:d}".format(z),end="")

            #save pickle data
            list = np.asarray(list)
            save_path = path[4] + "/pkl"
            check_existfile(save_path)
            save_path = save_path + addPath

            f = open(save_path,'wb')
            pickle.dump(list,f)
            print(" ")
        num_list.append(z)

        #画像サイズの情報を保存する
        size_path = os.path.dirname(path[0])
        print("size_path",size_path)
        with open(size_path + "/sizedata.txt",mode='w') as f_sizedata: 
            for sn in range(len(image_name)):
                f_sizedata.write(image_name[sn]+ ":" +str(image_size[sn])+"\n" )
        image_size,image_name = [],[]

    dataset_image,dataset_label = [],[]
    for i in range(len(task)):
        dataset_image.append(root + task[i] + "/pkl/image.pkl") 
        dataset_label.append(root + task[i] + "/pkl/label.pkl")

    #データを連結
    ori_image = load_pkl(dataset_image[0])
    ori_label = load_pkl(dataset_label[0])

    for i in range(len(dataset_image) - 1):
        #image
        add_image = load_pkl(dataset_image[i + 1])
        ori_image = concatenate_pkl(ori_image,add_image)

        #label
        add_label = load_pkl(dataset_label[i + 1])
        ori_label = concatenate_pkl(ori_label,add_label)


    root_path = "D:/Users/takami.h/Desktop/AMED_proiect/U-net/AmedSegmentation/image/"
    f_image = open(root_path + "image.pkl",'wb')
    f_label = open(root_path + "label.pkl",'wb')
    
    with open(root_path + "image.pkl",mode='wb') as f_image:    
        pickle.dump(ori_image,f_image) 
    with open(root_path + "label.pkl",mode='wb') as f_label:    
        pickle.dump(ori_label,f_label)        

    #datasetの詳細をテキストで保存
    with open(root_path + "dataset.txt",mode='w') as f_dataset: 
        f_dataset.write("data Path  :" + root)
        f_dataset.write("\n画像枚数   :" + str(len(ori_image)))
        f_dataset.write("\n解像度     :" + str(ori_image[0].shape))
        f_dataset.write("\n各フォルダの画像枚数")
        for t in range(len(task)):
            f_dataset.write("\n     -"+task[t] + " :" + str(num_list[t]))

##  部分処理の関数
def load_pkl(filename):
    with open(filename,mode='rb') as f:
       pkl = pickle.load(f)
    return pkl

def Path(path):

    """
    #   #   パスの作成   #   #
   
    Parameters
    --------------
    train   :string
        -画像データパス
    val   :string
        -ラベル画像データパス
    
    return 
    --------------
    ()    :タプル
        -（train_image , val_image , train_label ,val_label）

    """

    #ディレクトリ無ければ作る
    image_path = path + "/image"
    label_path = path + "/label"
    rimage_path = image_path + "/resize"
    rlabel_path = label_path + "/resize"

    check_existfile(image_path)
    check_existfile(label_path)
    check_existfile(rimage_path)
    check_existfile(rlabel_path)

    return (image_path,label_path,rimage_path,rlabel_path,path)

if __name__ == "__main__":

    """
    ※注意事項※
    1.rootを指定する事
        -dataまでのroot path
    2.taskを指定する事
        -どのファイルからdataset作るか指定する事
    """
    root = "//aka/share/amed/amed_liver_label/"
    task = ["Image06",
            "Image07",
            "Image09",
            "Image10",
            "Image11",
            "Image17",
            "Image15_with_tumor",
            "Amed_train_image"]

    size = (560,560)
    make_dataset(root,task,size)

