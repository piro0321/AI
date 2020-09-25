# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, concatenate, Dropout,BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D,Lambda, Multiply,Add

from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
from mlc.function_2 import IoU

def Gate_Attention_module(inputs):
    
    x = inputs.shape[1]
    y = inputs.shape[2]
    d = inputs.shape[3]
    out1 = Conv2D(int(d), 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    out2 = Conv2D(int(d), 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(out1)
    out3 = Multiply()([inputs, out2])
    #out3 = Lambda(lambda x: K.dot(inputs,out2), output_shape=(640))(out2)
    out4 = Add()([out3, inputs]) 
    #out4 = out3 + inputs

    return out4

def unet_3d(input_shape=(144, 144, 144, 1)):
    """

    """
    #入力画像サイズ．
    optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
    input = Input(input_shape)

    conv1 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling3D(size=(2, 2, 2))(drop5)
    conv6 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([drop4, conv6])
    conv6 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    conv7 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, conv7])
    conv7 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    conv8 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, conv8])
    conv8 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    conv9 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, conv9])
    conv9 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=input, outputs=conv10)
    model.compile(optimizer = Adam(lr=1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer =optimizer, loss = 'binary_crossentropy', metrics
    #= ['accuracy'])

    return model

def unet_2d_GAM(input_shape=(144, 144, 1)):
    """

    """

    #optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
    #入力画像サイズ．
    inputs = Input(input_shape)
    fsize = 3
    conv1 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Gate_Attention_module(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Gate_Attention_module(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Gate_Attention_module(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Gate_Attention_module(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Gate_Attention_module(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([drop4, conv6])
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Gate_Attention_module(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, conv7])
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Gate_Attention_module(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, conv8])
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Gate_Attention_module(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, conv9])
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    #conv10 = Conv2D(1, 1, activation='softmax')(conv9)


    model = Model(input=inputs, output=conv10)
    model.compile(optimizer = Adam(), loss = 'binary_crossentropy',metrics = ['accuracy',IoU])
    #model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics
    #= ['accuracy'])
    #model.compile(optimizer = Adam(lr=1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def unet_2d(input_shape=(144, 144, 1)):
    """

    """

    optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
    #入力画像サイズ．
    inputs = Input(input_shape)
    fsize = 3
    conv1 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([drop4, conv6])
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, conv7])
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, conv8])
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, conv9])
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, fsize, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    #conv10 = Conv2D(1, 1, activation='softmax')(conv9)


    model = Model(input=inputs, output=conv10)
    model.compile(optimizer = Adam(lr=1e-5), loss = 'binary_crossentropy',metrics = ['accuracy',IoU])
   # model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics= ['accuracy'])
    #model.compile(optimizer = Adam(lr=1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


if __name__ == "__main__":
    model = unet_3d(input_shape=(144, 144, 144, 1))
    model.summary()

    train = np.zeros((10, 144, 144, 144, 1), dtype=np.float32)
    mask = np.ones((10, 144, 144, 144, 1), dtype=np.float32)

    print(train.shape)

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(train, mask, batch_size=2, epochs=5, callbacks=[model_checkpoint])