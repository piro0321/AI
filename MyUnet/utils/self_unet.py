import numpy as np
from keras.models import Model
from keras.layers.core import Activation, Reshape
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, concatenate, Multiply, Add
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
from keras.metrics import binary_accuracy, categorical_accuracy
from keras import backend as K

SGD = SGD()
Adagrad =  Adagrad()
AdaDelta = Adadelta()
RMSprop = RMSprop()
optimizer = AdaDelta
def create_convblock(input, chs, dropout=False, normal=True, pooling=True,GAM=False):
    x = input
    fsize = 3
    x = Conv2D(chs, fsize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(chs, fsize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if dropout == True:
        x = Dropout(0.4)(x)
        drop = x
    if normal == True:
        x = BatchNormalization()(x)
    y = x

    if GAM == True:
        x = Gate_Attention_module(x)

    if pooling == True:
        x = MaxPooling2D(pool_size = (2, 2))(x)
    if dropout == True:
        return x, drop
    else:
        return x, y

def create_deconvblock(input, chs, connection=None,GAM=False):
    x = input
    fsize = 3
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(chs, fsize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if connection != None:
        x = concatenate([x, connection])
    x = Conv2D(chs, fsize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(chs, fsize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if GAM == True:
        x = Gate_Attention_module(x)

    return x

def Gate_Attention_module(inputs):
    
    x,y,z = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    fsize = 3
    x = inputs
    x = Conv2D(int(z), fsize, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(int(z), fsize, activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Multiply()([inputs, x])
    x = Add()([x, inputs]) 

    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_2d(input_shape=(400, 400, None), classes=None,GAM=False):
    input = Input(input_shape)

    #conv1, n_pool1 = create_convblock(input, 16)
    #conv2, n_pool2 = create_convblock(conv1, 32)
    #conv3, n_pool3 = create_convblock(conv2, 64)
    #conv4, drop4 = create_convblock(conv3, 128, dropout = True)
    #conv5, drop5 = create_convblock(conv4, 256, dropout = True)
    #conv6, drop6 = create_convblock(conv5, 512, dropout = True, pooling =
    #False)

    #deconv7 = create_deconvblock(conv6, 256, connection = drop5)
    #deconv8 = create_deconvblock(deconv7, 128, connection = drop4)
    #deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3)
    #deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    #deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    conv1, n_pool1 = create_convblock(input, 16)
    conv2, n_pool2 = create_convblock(conv1, 32)
    conv3, n_pool3 = create_convblock(conv2, 64)
    conv4, drop4 = create_convblock(conv3, 128, dropout = True)
    conv5, drop5 = create_convblock(conv4, 256, dropout = True, pooling = False)
    
    deconv8 = create_deconvblock(conv5, 128, connection = drop4)
    deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3)
    deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    if classes != None:
        #deconv12 = Conv3D(2, 3, activation = 'relu', padding = 'same',
        #kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv2D(classes, 1, padding = 'valid')(deconv11)
        #output = Reshape(input_shape[0] * input_shape[1] * input_shape[2],
        #classes)(deconv13)
        output = Activation('softmax')(deconv13)
        model = Model(inputs = input, outputs = output)
        #model.compile(optimizer = optimizer, loss =
        #'categorical_crossentropy', metrics = [categorical_accuracy])
        model.compile(optimizer = Adam(lr=1e-3), loss = 'categorical_crossentropy', metrics = [categorical_accuracy])

        return model
    else:
        deconv12 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv2D(1, 1, activation = 'sigmoid')(deconv12)

    model = Model(inputs = input, outputs = deconv13)
    #model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics
    #= ['accuracy'])
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


def unet_2d_GAM(input_shape=(400, 400, None), classes=None):
    input = Input(input_shape)

    #conv1, n_pool1 = create_convblock(input, 16)
    #conv2, n_pool2 = create_convblock(conv1, 32)
    #conv3, n_pool3 = create_convblock(conv2, 64)
    #conv4, drop4 = create_convblock(conv3, 128, dropout = True)
    #conv5, drop5 = create_convblock(conv4, 256, dropout = True)
    #conv6, drop6 = create_convblock(conv5, 512, dropout = True, pooling =
    #False)

    #deconv7 = create_deconvblock(conv6, 256, connection = drop5)
    #deconv8 = create_deconvblock(deconv7, 128, connection = drop4)
    #deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3)
    #deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    #deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    conv1, n_pool1 = create_convblock(input, 16,GAM=True)
    conv2, n_pool2 = create_convblock(conv1, 32,GAM=True)
    conv3, n_pool3 = create_convblock(conv2, 64,GAM=True)
    conv4, drop4 = create_convblock(conv3, 128, dropout = True,GAM=True)
    conv5, drop5 = create_convblock(conv4, 256, dropout = True, pooling = False)
    
    deconv8 = create_deconvblock(conv5, 128, connection = drop4,GAM=True)
    deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3,GAM=True)
    deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    if classes != None:
        #deconv12 = Conv3D(2, 3, activation = 'relu', padding = 'same',
        #kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv2D(classes, 1, padding = 'valid')(deconv11)
        #output = Reshape(input_shape[0] * input_shape[1] * input_shape[2],
        #classes)(deconv13)
        output = Activation('softmax')(deconv13)
        model = Model(inputs = input, outputs = output)
        #model.compile(optimizer = optimizer, loss =
        #'categorical_crossentropy', metrics = [categorical_accuracy])
        model.compile(optimizer = Adam(lr=1e-3), loss = 'categorical_crossentropy', metrics = [categorical_accuracy])

        return model
    else:
        deconv12 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv2D(1, 1, activation = 'sigmoid')(deconv12)

    model = Model(inputs = input, outputs = deconv13)
    #model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics= ['accuracy'])
    #model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = [mean_IoU])
    #model.compile(optimizer = optimizer_AdaDelta, loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


def self_unet_3d(input_shape=(64, 128, 128, None), classes=None):
    optimizer = SGD(decay = 1e-6, momentum = 0.9, nesterov = True)
    input = Input(input_shape)

    #conv1, n_pool1 = create_convblock(input, 16)
    #conv2, n_pool2 = create_convblock(conv1, 32)
    #conv3, n_pool3 = create_convblock(conv2, 64)
    #conv4, drop4 = create_convblock(conv3, 128, dropout = True)
    #conv5, drop5 = create_convblock(conv4, 256, dropout = True)
    #conv6, drop6 = create_convblock(conv5, 512, dropout = True, pooling =
    #False)

    #deconv7 = create_deconvblock(conv6, 256, connection = drop5)
    #deconv8 = create_deconvblock(deconv7, 128, connection = drop4)
    #deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3)
    #deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    #deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    conv1, n_pool1 = create_convblock(input, 16)
    conv2, n_pool2 = create_convblock(conv1, 32)
    conv3, n_pool3 = create_convblock(conv2, 64)
    conv4, drop4 = create_convblock(conv3, 128, dropout = True)
    conv5, drop5 = create_convblock(conv4, 256, dropout = True, pooling = False)
    
    deconv8 = create_deconvblock(conv5, 128, connection = drop4)
    deconv9 = create_deconvblock(deconv8, 64, connection = n_pool3)
    deconv10 = create_deconvblock(deconv9, 32, connection = n_pool2)
    deconv11 = create_deconvblock(deconv10, 16, connection = n_pool1)
    
    if classes != None:
        #deconv12 = Conv3D(2, 3, activation = 'relu', padding = 'same',
        #kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv3D(classes, 1, padding = 'valid')(deconv11)
        #output = Reshape(input_shape[0] * input_shape[1] * input_shape[2],
        #classes)(deconv13)
        output = Activation('softmax')(deconv13)
        model = Model(inputs = input, outputs = output)
        #model.compile(optimizer = optimizer, loss =
        #'categorical_crossentropy', metrics = [categorical_accuracy])
        model.compile(optimizer = Adam(lr=1e-3), loss = 'categorical_crossentropy', metrics = [categorical_accuracy])

        return model
    else:
        deconv12 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(deconv11)
        deconv13 = Conv3D(1, 1, activation = 'sigmoid')(deconv12)

    model = Model(inputs = input, outputs = deconv13)
    #model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics
    #= ['accuracy'])
    model.compile(optimizer = Adam(lr=1e-3), loss = 'binary_crossentropy', metrics = [binary_accuracy])

    return model
