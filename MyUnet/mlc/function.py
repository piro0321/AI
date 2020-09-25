# -*- coding: utf-8 -*-

    #   # いろんな処理・機能をまとめたファイル
    #------------------------------------------
    #便利な処理をまとめた[function.py]関数，他のファイルからimportして使用する
    #------------------------------------------
import os
import re
import glob
import cv2
import pickle
import numpy as np
import subprocess
from scipy import ndimage as ndi
from PIL import Image
np.set_printoptions(threshold=400000000)

#1
def check_existfile(path):
    """
    Parameter
    ---------
    path    :string
            -ディレクトリの存在を確認したいパス
    """
    if os.path.exists(path) == False :
       os.makedirs(path)
#2
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
#3
def Roi_Reduction(coord,reduction_rate):
    """
    大きいROIを0.6倍に縮小する関数

    Parameter
    ---------
    coord   :tuple
            -roiの座標

    Return
    ------
    remake_roi  :tuple
            -修正後の座標
    """
    reduction_rate = (1-reduction_rate)/2
    x1,y1,x2,y2 = coord

    reduction_x = (x2-x1)*reduction_rate
    reduction_y = (y2-y1)*reduction_rate

    x1 = int(x1 + reduction_x)
    y1 = int(y1 + reduction_y)
    x2 = int(x2 - reduction_x)
    y2 = int(y2 - reduction_y)

    return (x1,y1,x2,y2)
#4
def draw_contours(image,label,color = (0,0,255),epsilon = 0.0001 ):
    """
    ラベルの輪郭を抽出
    Parameter
    ---------
    image   :numpyのlist
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

    val = np.zeros(image.shape)
    #輪郭抽出
    label = label*255
    ret,thresh = cv2.threshold(label,0,255,0)
    label, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approx_contours = []
    for k, cnt in enumerate(contours):
        # 輪郭の周囲の長さを計算する。
        arclen = cv2.arcLength(cnt, True)
        # 輪郭を近似する。
        approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.0001 * arclen, closed=True)
        approx_contours.append(approx_cnt)
        cv2.drawContours(image, approx_contours,-1,color,3)
        cv2.drawContours(val, approx_contours,-1,(1,1,1),-1)

    return image,label,val
#5
def load_image_from_txt(dict):
    """
    #   datasetのテキストを読み取り，対応した画像をリストで返す #

    Parameter
    ---------
    text_path   :string
                -datasetまでのpath

    return
    ---------
    cimage_list     :カラー画像のリスト
    gimage_list     :グレー画像のリスト
    imagename_list  :画像のdataname
    """
    text_path = dict["mydataset_path"]
    data_path = dict["amedabdomen_path"]
    cimage_list,gimage_list ,imagename_list= [],[],[]

    with open(text_path) as f:
        for i,line in enumerate(f):
            #line       :hcc/007454_9300119000010448
            #root_path  :007454_9300119000010448
            root_path = os.path.basename(line)
            file_path = (data_path+line + "/"+ "*_denoised.png").replace('\n', '')

            files = glob.glob(file_path)
            for f in files:
                #image_path     :1.2.840.113619.2.256.50119124685.1563429440.1915_denoised.png
                image_path = os.path.basename(f)

                cimage = cv2.imread(f,cv2.IMREAD_COLOR)
                gimage = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
                gimage = np.expand_dims(gimage,axis=-1)


            imagename_list.append((line+"/"+image_path).replace('\n', ''))


            cimage_list.append(cimage)
            gimage_list.append(gimage)
            print("\r{0:d}".format(i),end="")
         
            if i == 20:
                break
    return cimage_list,gimage_list,imagename_list
#6
def Make_Mytest_Datasettext(dict):
    """
    #   テスト用データの再構築     #
    -アーチファクトの画像を除いたテストデータを元にリストを再構築
    -
    """
    print("Start -Make_Mytest_Datasettext-")

    root_path = dict["amedtest_path"]
    new_dataset_path = dict["remaketestdata_path"]

    input("teisi")
    my = []
    new = []

    label = os.listdir(new_dataset_path+"image")
    same = []
    mysame = []

    with open(root_path, 'r') as f:
        for l  in f:
            new.append(l.rstrip('\n'))
    for j, n in enumerate(new):
        print('\r{:5d}/{:5d}/{:5d}'.format(len(same),j+1, len(new)), end='')
        #呼び込んで
        denoised_path = glob.glob(os.path.join(dict["amedabdomen_path"], n, '*_denoised.png'))[0]
        denoised = os.path.basename(denoised_path)
        #denoised :image name
        for i, l in enumerate(label):
            
            #テスト画像の中に含まれていたら
            if denoised == l:

                #same.append(n+"  "+ denoised + '\n')
                same.append(n+"\n")
        
    #結果出力
    #書き出し
    with open("./my_testdata.txt",mode='w') as f: 
        f.writelines(same)


    print("\nEnd process.")
#7
def check_sameimage_train(dict):
    """
    #   画像のdatanameから同じ画像を別のフォルダへ移動する   #
    -画像のimage_pathに画像数が多いほうを指定する
    -
    """
    root_path = "//aka/share/amed/amed_liver_label/Amed_train_image/"
    image_path = root_path + "image"
    label_path = root_path + "label"

    my = []
    new = []

    image = os.listdir(image_path)
    label = os.listdir(label_path)
    same = []
    differ = []

    for j, n in enumerate(image):
        image_name = os.path.splitext(os.path.basename(image[j]))[0]
        check = 0
        for i, l in enumerate(label):

            print('\r{:5d}/{:5d}'.format(j, len(image)), end='')

            label_name = os.path.splitext(os.path.basename(label[i]))[0]
            #テスト画像の中に含まれていたら


            if image_name== label_name:
                same.append(image[i] + '\n')
                check = 1
                break

        if check != 1:
            differ.append(image[j] + '\n')
            check_existfile(root_path + "differ")

            dif_image = cv2.imread(image_path + "/" +image[j])
            cv2.imwrite(root_path + "differ/"+image[j],dif_image)
            
            os.remove(image_path + "/" +image[j])
        
    #結果出力
    #書き出し
    with open(root_path + "same.txt",mode='w') as f: 
        f.writelines("テストに学習画像が含まれた数    :"+str(len(same))+"\n")
        f.writelines(same)

    with open(root_path + "differ.txt",mode='w') as aaa: 
        aaa.writelines("テストに学習画像が含まれてない数    :"+str(len(differ))+"\n")

        aaa.writelines(differ)


    print("学習画像が含まれた数    :"+str(len(same))+"\n")
#8
def movie_to_image(num_cut,videopath):
    """
    #   # 動画から画像に変換
    
    Parameter
    ----------
    num_cut     :int
                -間引くフレームの数
    videopath   :string
                -動画までのpath
    result
    -------
    動画を指定したフレームでカットし画像として保存
    """
    dataname =  os.path.splitext(os.path.basename(videopath))[0]
    video_path = videopath     #動画のパス
    output_path = "./image_data/video_to_image/" +dataname #保存先パス

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
#9
def concatenate_pkl(list,add):
    """
    #  listを読み込んで連結   #

    Parameters
    --------------
    list    :list型
        -連結するための配列
    list_add:list型
        -後ろに連結する配列
    Return
    --------------
    output  :list型
        -連結後のlist
    """

    list = list.transpose(1,0,2,3) #軸を変更 xyzw -> yxzw
    list_add = add.transpose(1,0,2,3) 

    output = np.hstack((list,list_add))
    output = output.transpose(1,0,2,3) 

    return output
#10
def Read_Parameter(path):
    """
    #   Parameter.txtから初期値を呼び出す

    Param
    -----------
    path    :string
            -パスを受け取る
    result
    ------
    textのデータを読み込んだ辞書を作成

    Parameter.txtの構造
    -------------------
    keys    :values
        ----注意点----
        -keysの後ろはtab(\t)にすること
        -文字列は""　で囲むこと
    """

    """辞書メモ
    dict.keys()     :登録した辞書のkeysを読み込む   forで一つずつ
    dict.values()   :登録した辞書のvalueを読み込む  forで一つずつ
    dict.items()    :keys,valueを受け取る            for mykey,myvalue in dict.items()
    """
    dict = {}
    with open(path,encoding = 'utf-8' ) as file:

        for line in file:
            line = line.split("\t:")
            line[1]=line[1].rstrip('\n')
            dict.update({line[0]:line[1]})

    return dict
#11
def readme():
    """
    #   #実行すれば見たいread meファイルが見れます
    -完全遊びで作ったプログラムなので不必要なら消しちゃって
    """
    print("Start -Read Me- function\n")
    path = "D:/Users/takami.h/Desktop/Amedproject/readme_file/"
    print("What text do you read?")
    print("1.Readme for all programs")
    print("2.About Parameter.txt")
    print("other.Exit this program")
    a = input("select number: ")
    a = int(a)
    if a is 1:
        path = path + "readme.txt"
    elif a is 2:
        path = path + "about_Parameter.txt"
    else:
        return 
    print(a)
    subprocess.Popen([path],shell=True)


if __name__ == "__main__":
    readme()