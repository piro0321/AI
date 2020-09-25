# -*- coding: utf-8 -*-

import os
import numpy as np

from keras.models import Model
from keras.layers import add
from keras.layers import Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, Deconv2D, Cropping2D
from keras.layers import Activation


def fcn_vgg_8s(input_shape=(224, 224, 3), classes=21):
    img_shape = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu',padding='same')(img_shape)
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p3 = x
    p3 = Conv2D(classes, (1, 1), activation='relu')(p3)

    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p4 = x
    p4 = Conv2D(classes, (1, 1), activation='relu')(p4)
    p4 = Deconv2D(classes, (4, 4), strides=(2, 2), padding='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)

    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p5 = x
    p5 = Conv2D(classes, (1, 1), activation='relu')(p5)
    p5 = Deconv2D(classes, (8, 8), strides=(4, 4), padding='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)

    merged = add([p3, p4, p5])
    x = Deconv2D(classes, (16, 16), strides=(8, 8), padding='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)

    x = Reshape((input_shape[0] * input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    x = Reshape((input_shape[0], input_shape[1], classes))(x)

    model = Model(img_shape, x)

    return model