# -*- coding: utf-8 -*-

import datetime
#   # 学習，評価，テスト用のプログラム
# ------------------------------------------
# ネットワークモデルの学習，検証を行う．主にtest_amedを使用
# ------------------------------------------

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
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
np.set_printoptions(threshold=400000000)


size = (512, 512)

# -


def load_image_s(filename):
    print("Load Pickle Data.")
    with open(filename, mode='rb') as f:
        image = pickle.load(f)
    print("Load Complete.")
    return image
# -


def Echo_Extraction(gray, seglabel):
    print("Start processing -Contour Extraction")
    """
    #   #   超音波画像領域の切り出し　精度不安定
    Parameter
    ---------
    gray     :numpy image
            -グレー画像
    seglabel :numpy image label
            -訓練画像のラベル

    return
    ------
    image_list  :超音波画像領域を切り出した画像
    label_list  :↑を切り出すためのマスク画像


    """
    image_list, label_list = [], []
    for k in range(len(gray)):
        print("\r{0:d}".format(k), end="")

        # 画像呼び出し
        img = np.copy(gray[k])
        seg_label = np.copy(seglabel[k])
        # 特徴点考察
# ========================周囲
        label_preprocess = np.zeros(img.shape)

        ret_1, thresh_1 = cv2.threshold(img, 200, 255, 0)  # 閾値(200,255のものを残す)
        image_1, contours_1, hierarchy_1 = cv2.findContours(
            thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 色塗り
        for i in range(0, len(contours_1)):
            if len(contours_1[i]) > 0 and cv2.contourArea(contours_1[i]) < 100000:
                label_preprocess = cv2.drawContours(
                    label_preprocess, contours_1, i, 1, thickness=-1)


# ========================超音波領域
        ret, thresh = cv2.threshold(img, 4, 255, 0)  # 閾値(2値化)

        # 周囲の推測領域を除外する
        thresh[label_preprocess[:, :, 0] > 0] = 0
        image, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        w, h, _ = img.shape
        label = np.zeros(img.shape)
        keep_label = np.zeros(img.shape)

        max_area = w*h*0.85
        min_area = w * 50
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                if cv2.contourArea(contours[i]) < min_area:
                    continue
                if cv2.contourArea(contours[i]) > max_area:
                    keep_label = cv2.drawContours(
                        keep_label, contours, i, 1, thickness=-1)
                if cv2.contourArea(contours[i]) < max_area:
                    label = cv2.drawContours(
                        label, contours, i, 1, thickness=-1)

        # 画素の白い部分の数（腫瘍領域の四角形の面積)
        check = np.count_nonzero(label[:, :]*img[:, :] > 0)
        if check < 26000:
            label = np.copy(keep_label)

        # save data
        """
        img     :オリジナル画像
        label   :予測結果
        """
        #save_image(path_img,img,num = k)
        #save_image(path_label,label,num = k)

        # image

        re_image = img*label/255
        # label
        re_label = seg_label*label/255
        image_list.append(re_image)
        label_list.append(re_label)
    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)

    return image_list, label_list
# -


def myprocess_pixel_value(img):
    """
    #   #   Data Augmentation のオリジナルの処理
    -ここではガンマ値の変化を加える関数
    """
    # 線形濃度変換
    gamma = 0.8

    # 画素値の最大値
    imax = img.max()

    # ガンマ補正
    img = imax * (img / imax)**(1/gamma)
    return img
# -


def make_list(data_path, size):
    count = 0
    list = []
    format = ".jpg"
    #format = ".bmp"
    for root, dirs, files in os.walk(data_path):
        for fn in files:
            count += 1
            print("\r{0:d}".format(count), end="")
            bn, ext = os.path.splitext(fn)
            if ext not in [format]:
                continue

            filename = os.path.join(root, fn)
            inputs = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            inputs = inputs

            inputs = cv2.resize(inputs, size)
            inputs = np.expand_dims(inputs, axis=-1)
            inputs = np.asarray(inputs, dtype=np.float32)

            list.append(inputs)

    return list
# -


def check_existfile(path):
    if os.path.exists(path) == False:
        os.makedirs(path)
# -


def load_image(path, size):
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
    color, gray = [], []
    image_info, dataname_info = [], []
    format = ".jpg"
    for root, dirs, files in os.walk(path):
        for fn in files:
            count += 1
            bn, ext = os.path.splitext(fn)
            if ext not in [format]:
                continue
            filename = os.path.join(root, fn)
            print(fn)
            input()

            # カラー画像
            clr_input = cv2.imread(filename, cv2.IMREAD_COLOR)
            image_info.append(clr_input.shape)  # 元画像サイズを格納
            clr_input = cv2.resize(clr_input, size)

            #clr_input = np.expand_dims(clr_input,axis=-1)
            #clr_input = np.asarray(clr_input, dtype=np.float32)

            # グレー画像
            gray_input = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            gray_input = cv2.resize(gray_input, size)

            gray_input = np.expand_dims(gray_input, axis=-1)
            #gray_input = np.asarray(gray_input, dtype=np.float32)

            color.append(clr_input)
            gray.append(gray_input)
            print("\r{0:d}".format(count), end="")
    """
    save_path = "./image_data/predict_gray.pkl"
    f = open(save_path,'wb')
    pickle.dump(gray,f)
    save_path = "./image_data/predict_clr.pkl"
    f = open(save_path,'wb')
    pickle.dump(color,f)
    """
    print("load Image List Complete")
    return color, gray, image_info
# -


def show(image):
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# -


def draw_contours(image, label, color=(0, 0, 255), epsilon=0.0001):
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

    # 以下5行確認用
    #size = (400,400)
    #filename = "E:/output/image_data/Liver_area/Image09/label"
    #images,labels,sizedata = load_image(filename,size)
    #label[label > 200] = 0
    #label[label > 20] = 255

    val = np.zeros(image.shape)
    # 輪郭抽出
    ret, thresh = cv2.threshold(label, 0, 255, 0)
    label, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approx_contours = []
    for k, cnt in enumerate(contours):
        # 輪郭の周囲の長さを計算する。
        arclen = cv2.arcLength(cnt, True)
        # 輪郭を近似する。
        approx_cnt = cv2.approxPolyDP(
            cnt, epsilon=0.0001 * arclen, closed=True)
        approx_contours.append(approx_cnt)
        cv2.drawContours(image, approx_contours, -1, color, 3)

    return image, label, val


"""--以下、メインの処理関数
"""


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
    # Setup Parameter
    size = (512, 512)
    target_size = (size[0], size[1], 1)
    batch_size = 4
    epoch = 50
    classes = 2
    gen_num = 5
    seed = 10

    # 実行時間の計測用（処理実行開始時間）
    dt_now = datetime.datetime.now()
    days = dt_now.year+dt_now.month + dt_now.day

    # Load model
    #model = unet_2d(input_shape = target_size)
    model = unet_2d_GAM(input_shape=target_size)
    model.summary()

    # Setup dataset
    image = load_image_s('./image/image.pkl')
    label = load_image_s('./image/label.pkl')

    # preprocessing -前処理-
    # 前処理に使うのはやめた方がいいかもしれない
    #image,label = Echo_Extraction(image,label)
    # for ss in range(len(image)):
    #    cv2.imwrite("./image/check/image/{0:04d}.jpg".format(ss) , image[ss])
    #    cv2.imwrite("./image/check/label/{0:04d}.jpg".format(ss), label[ss])

    # 正規化
    image = image/255
    label = label/255
    train_image, val_image = train_test_split(
        image, test_size=0.3, random_state=seed, shuffle=True)
    train_label, val_label = train_test_split(
        label, test_size=0.3, random_state=seed, shuffle=True)

    # check dataset
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

    # Data Augumentation
    nb_data = len(train_image) * gen_num
    val_nb_data = len(val_image) * gen_num
    #train_gen = ImageDataGenerator(preprocessing_function = myprocess_pixel_value,width_shift_range=[-10,10],height_shift_range=[-10,10])
    train_gen = ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1)
    val_gen = ImageDataGenerator()

    # Setup CallBacks
    ModelCheckpoint_ = ModelCheckpoint(
        './model/unet_2d_weights.hdf5', monitor='val_loss', save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ReduceLROnPlateau_ = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.002)
    CSVLogger_ = keras.callbacks.CSVLogger(
        "./model/callback/log_learning.csv", separator=',', append=False)
    TensorBoard_ = keras.callbacks.TensorBoard(
        log_dir="./model/log", histogram_freq=1)

    # Set CallBack
    #callbacks = [ModelCheckpoint_,ReduceLROnPlateau_,early_stopping,CSVLogger_]
    callbacks = [ModelCheckpoint_, early_stopping, CSVLogger_]
    #callbacks = [ModelCheckpoint_]

    # Learning
    # (generator=train_gen.flow(train_image, train_label,batch_size=batch_size,save_to_dir="E:/output/data_augumentation/1", save_prefix='img', save_format='jpg')
    model.fit_generator(generator=train_gen.flow(train_image, train_label, batch_size=batch_size),
                        steps_per_epoch=nb_data,
                        epochs=epoch,
                        validation_data=val_gen.flow(
                            val_image, val_label, batch_size=batch_size),
                        validation_steps=val_nb_data,
                        max_queue_size=10,
                        callbacks=callbacks)

    # log保存
    with open("./model/model_detail.txt", mode='w') as f_dataset:
        f_dataset.write("作成日時     : " + days)
        f_dataset.write("画像サイズ   : " + size)
        f_dataset.write("seed         : " + seed)
        f_dataset.write("使用モデル   : unet_2d_GAM")
        f_dataset.write("\n画像枚数   :" + str(len(image)))
        f_dataset.write("\n     - train_image : " + str(len(train_image)))
        f_dataset.write("\n     - val_image   : " + str(len(val_image)))

    # 以下検証
    # 重みの保存位置修正
    model.load_weights('./model/unet_2d_weights.hdf5')

    score = model.evaluate(val_image, val_label, batch_size=3, verbose=1)

    print('Test loss     ', score[0])
    print('Test accuracy ', score[1])
# -


def Varidation():
    print("start predict")
    size = (512, 512)
    target_size = (size[0], size[1], 1)
    filename = "image11_with_tumor"
    path = "./image_data/"+filename
    cimage, gimage, sizedata = load_image(path, size)
    os.system("cls")

    # with open("./image_data/predict_gray.pkl",mode='rb') as f:
    #    gimage = pickle.load(f)
    # with open("./image_data/predict_clr.pkl",mode='rb') as f:
    #   cimage = pickle.load(f)

    #model = unet_2d(input_shape=target_size)
    model = unet_2d_GAM(input_shape=target_size)
    model.load_weights('./model/unet_2d_weights.hdf5')
    os.system("cls")

    for i in range(len(cimage)):
        mean = 0
        print("\r{0:d}".format(i), end="")

        w, h, c = gimage[i].shape
        target = gimage[i].reshape((1, w, h, 1))
        target = target / 255
        pred = model.predict(target)[0]
        pred = np.reshape(pred, (w, h, 1), order='F')

        # 閾値

        #mean = np.mean(pred)
        mean = 0.95

        # print(pred)
        pred[pred >= mean] = 255
        pred[pred < mean] = 0

        pred = np.asarray(pred, dtype=np.uint8)

        val = cimage[i]

        """
        val     :検出した領域の輪郭をoriginal画像と重ねた画像
        label   :検出した領域の輪郭
        val     :検出した領域
        """
        val, label, _ = draw_contours(val, pred)

        cv2.imwrite("./result/test/"+filename +
                    "/target/{0:04d}.jpg".format(i), cimage[i])
        cv2.imwrite("./result/test/"+filename +
                    "/pred/{0:04d}.jpg".format(i), pred)
        cv2.imwrite("./result/test/"+filename +
                    "/val/{0:04d}.jpg".format(i), val)
# -


def test_amed():
    """
    """
    print("start predict")
    size = (512, 512)

    target_size = (size[0], size[1], 1)
    filename = "image06"
    path = "./image_data/"+filename
    cimage, gimage, sizedata = load_image(path, size)

    # with open("./image_data/predict_gray.pkl",mode='rb') as f:
    #    gimage = pickle.load(f)
    # with open("./image_data/predict_clr.pkl",mode='rb') as f:
    #    cimage = pickle.load(f)

    #model = unet_2d(input_shape=target_size)
    model = unet_2d_GAM(input_shape=target_size)
    model.load_weights('./model/unet_2d_weights.hdf5')
    os.system("cls")

    for i in range(len(cimage)):
        mean = 0

        start = time.time()

        print("\r{0:d}".format(i), end="")
        w, h, c = gimage[i].shape
        target = cv2.resize(gimage[i], size)
        target = target.reshape((1, 512, 512, 1))
        target = target / 255

        pred = model.predict(target)[0]
        pred = np.reshape(pred, (512, 512, 1), order='F')

        # Heat Map 作成
        #pred_heat = pred[:,:,0]

        gray = np.copy(gimage[i])
        # 疑似カラーを付与
        heatmap = np.round(pred*255)
        heatmap = heatmap.astype('uint8')
        heatmap = cv2.applyColorMap(heatmap, 2)

        # 画像合成
        heatmap = cv2.resize(heatmap, (h, w))

        # チャネル追加
        gray_2 = np.concatenate((gray, gray), axis=2)
        gray = np.concatenate((gray_2, gray), axis=2)

        heatmap = cv2.addWeighted(gray, 0.3, heatmap, 0.7, 0)

        check_existfile("./result/test/"+filename+"/heatmap")
        cv2.imwrite("./result/test/"+filename +
                    "/heatmap/{0:04d}.jpg".format(i), heatmap)

        #mean = np.mean(pred)
        mean = 0.7
        # 推測
        pred[pred >= mean] = 1
        pred[pred < mean] = 0

        pred = np.asarray(pred, dtype=np.uint8)
        val = np.copy(cimage[i])

        """
        val     :検出した領域の輪郭をoriginal画像と重ねた画像
        label   :検出した領域の輪郭
        val     :検出した領域
        """

        pred_resize = cv2.resize(pred, (h, w))
        pred_resize = np.reshape(pred_resize, (w, h, 1), order='F')

        val, label, _ = draw_contours(val, pred_resize)

        val_resize = cv2.resize(val, (h, w))

        pred_resize_2 = np.concatenate((pred_resize, pred_resize), axis=2)
        pred_resize = np.concatenate((pred_resize_2, pred_resize), axis=2)

        #result_image = pred_resize*cimage[i]

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        #cv2.imwrite("//aka/share/amed/amed_liver_label/test_result/target/{0:04d}.jpg".format(i) , cimage[i])
        #cv2.imwrite("//aka/share/amed/amed_liver_label/test_result/pred/{0:04d}.jpg".format(i) , pred_resize*255)
        #cv2.imwrite("//aka/share/amed/amed_liver_label/test_result/val/{0:04d}.jpg".format(i) , val_resize)

        cv2.imwrite("./result/test/"+filename +
                    "/target/{0:04d}.jpg".format(i), cimage[i])
        cv2.imwrite("./result/test/"+filename +
                    "/pred/{0:04d}.jpg".format(i), pred_resize*255)
        cv2.imwrite("./result/test/"+filename +
                    "/val/{0:04d}.jpg".format(i), val_resize)


if __name__ == "__main__":

    print("Start Program.")
    # train()
    # Varidation()
    # test_amed()
