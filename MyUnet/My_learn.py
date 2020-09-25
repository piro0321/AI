
------------------------------------

import os
import re
import cv2
import glob
import pickle
import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.self_unet import unet_2d, unet_2d_GAM
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
np.set_printoptions(threshold=400000000)

import datetime

size = (512,512)

#-
def load_image_s(filename):
    print("Load Pickle Data.")
    with open(filename,mode='rb') as f:
        image = pickle.load(f)
    print("Load Complete.")
    return image


def make_list(data_path,size):
    count = 0
    list = []
    format = ".jpg" 
    #format = ".bmp"
    for root, dirs, files in os.walk(data_path):
        for fn in files:
            count+=1
            print("\r{0:d}".format(count),end="")
            bn, ext = os.path.splitext(fn)
            if ext not in [format]:
                continue

            filename = os.path.join(root, fn)
            inputs = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            inputs = inputs

            inputs = cv2.resize(inputs,size)
            inputs = np.expand_dims(inputs,axis=-1)
            inputs = np.asarray(inputs, dtype=np.float32)

            list.append(inputs)

    return list
#-


def check_existfile(path):
    if os.path.exists(path) == False :
       os.makedirs(path)
#-
def load_image(path,size):
    """
    parameter
    ------------
    path    :string
        -画像の位置

    return
    ------------
    color   :numpyのlist
            -カラー画像
    gray    :numpyのlist
            -グレ―画像
    """ 
    count = 0
    color,gray  = [],[]
    image_info,dataname_info = [],[]
    format = ".jpg"
    for root, dirs, files in os.walk(path):
        for fn in files:
            count+=1
            bn, ext = os.path.splitext(fn)
            if ext not in [format]:
                continue
            filename = os.path.join(root, fn)
            print(fn)
            input()

            #カラー画像
            clr_input = cv2.imread(filename,cv2.IMREAD_COLOR)
            image_info.append(clr_input.shape)#元画像サイズを格納
            clr_input = cv2.resize(clr_input,size)
            
            #clr_input = np.expand_dims(clr_input,axis=-1)
            #clr_input = np.asarray(clr_input, dtype=np.float32)

            #グレー画像
            gray_input = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            gray_input = cv2.resize(gray_input,size)
            
            gray_input = np.expand_dims(gray_input,axis=-1)
            #gray_input = np.asarray(gray_input, dtype=np.float32)
            
            
            color.append(clr_input)
            gray.append(gray_input)
            print("\r{0:d}".format(count),end="")
    """
    save_path = "./image_data/predict_gray.pkl"
    f = open(save_path,'wb')
    pickle.dump(gray,f)
    save_path = "./image_data/predict_clr.pkl"
    f = open(save_path,'wb')
    pickle.dump(color,f)
    """
    print("load Image List Complete")
    return color,gray,image_info
#-
def show(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#-


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
    
    #以下5行確認用
    #size = (400,400)
    #filename = "E:/output/image_data/Liver_area/Image09/label"
    #images,labels,sizedata = load_image(filename,size)
    #label[label > 200] = 0
    #label[label > 20] = 255

    val = np.zeros(image.shape)
    #輪郭抽出
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

    return image,label,val



def train():
    """
    #   #   ネットワークモデルの学習をするための関数
    -Network model  :U-net
    -save weight only
    
    ##注意点
    -学習時の画像サイズを必ず記録する事
    """


    """
    画像読み込み
    前処理(サイズ統一、終わってたらスルー)
    学習(fit)
    検証(predict)
    精度と学習モデル保存
    """
    #Setup Parameter
    size = (512,512)
    target_size = (size[0] , size[1], 1)
    batch_size = 4
    epoch = 50
    classes = 2
    gen_num = 5
    seed = 10

    #実行時間の計測用（処理実行開始時間）
    dt_now = datetime.datetime.now()    
    days = dt_now.year+dt_now.month + dt_now.day


    #Load model
    #model = unet_2d(input_shape = target_size)
    model = unet_2d_GAM(input_shape = target_size)
    model.summary()

    #Setup dataset
    image = load_image_s('./image/image.pkl') 
    label = load_image_s('./image/label.pkl')

    #preprocessing -前処理-
    #前処理に使うのはやめた方がいいかもしれない
    #image,label = Echo_Extraction(image,label)
    #for ss in range(len(image)):
    #    cv2.imwrite("./image/check/image/{0:04d}.jpg".format(ss) , image[ss])
    #    cv2.imwrite("./image/check/label/{0:04d}.jpg".format(ss), label[ss])

    #正規化
    image = image/255
    label = label/255
    train_image, val_image = train_test_split(image, test_size=0.3,random_state = seed,shuffle = True)
    train_label, val_label = train_test_split(label, test_size=0.3,random_state = seed,shuffle = True)
    
    #check dataset
    """
    t_image = train_image * 255
    t_label = train_label * 255
    v_image = val_image * 255
    v_label = val_label * 255

    for i in range(len(train_image)):
        cv2.imwrite("./image/check/train_image/{0:04d}.jpg".format(i) , t_image[i])
        cv2.imwrite("./image/check/train_label/{0:04d}.jpg".format(i), t_label[i])
    for i in range(len(val_image)):
        cv2.imwrite("./image/check/val_image/{0:04d}.jpg".format(i), v_image[i])
        cv2.imwrite("./image/check/val_label/{0:04d}.jpg".format(i), v_label[i])
    """

    #Data Augumentation
    nb_data = len(train_image) * gen_num
    val_nb_data = len(val_image) * gen_num
    #train_gen = ImageDataGenerator(preprocessing_function = myprocess_pixel_value,width_shift_range=[-10,10],height_shift_range=[-10,10])
    train_gen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1)
    val_gen = ImageDataGenerator()
    
    #Setup CallBacks
    ModelCheckpoint_ = ModelCheckpoint('./model/unet_2d_weights.hdf5', monitor='val_loss', save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=1)
    ReduceLROnPlateau_ = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.002) 
    CSVLogger_ = keras.callbacks.CSVLogger("./model/callback/log_learning.csv", separator=',', append=False)
    TensorBoard_ = keras.callbacks.TensorBoard(log_dir="./model/log",histogram_freq=1)

    #Set CallBack
    #callbacks = [ModelCheckpoint_,ReduceLROnPlateau_,early_stopping,CSVLogger_]
    callbacks = [ModelCheckpoint_,early_stopping,CSVLogger_]
    #callbacks = [ModelCheckpoint_]
    
    #Learning
    #(generator=train_gen.flow(train_image, train_label,batch_size=batch_size,save_to_dir="E:/output/data_augumentation/1", save_prefix='img', save_format='jpg')
    model.fit_generator(generator=train_gen.flow(train_image,train_label, batch_size=batch_size),
                    steps_per_epoch=nb_data,
                    epochs=epoch,
                    validation_data=val_gen.flow(val_image, val_label,batch_size=batch_size),
                    validation_steps=val_nb_data,
                    max_queue_size=10,
                    callbacks=callbacks)

    #log保存
    with open("./model/model_detail.txt",mode='w') as f_dataset: 

        f_dataset.write("タイトル     : 適当になんか")
        f_dataset.write("作成日時     : " + days)
        f_dataset.write("画像サイズ   : " + size)
        f_dataset.write("seed         : " + seed)
        f_dataset.write("使用モデル   : unet_2d_GAM")
        f_dataset.write("\n画像枚数   :" + str(len(image)))
        f_dataset.write("\n     - train_image : " + str(len(train_image)))
        f_dataset.write("\n     - val_image   : " + str(len(val_image)))

    #以下検証
    #重みの保存位置修正
    model.load_weights('./model/unet_2d_weights.hdf5')
    try:
        score = model.evaluate(val_image, val_label, batch_size=3,verbose=1)
        print('Test loss     ',score[0])
        print('Test accuracy ',score[1])
    except:
        print("Error")
    finally:
        print("End program")
#-