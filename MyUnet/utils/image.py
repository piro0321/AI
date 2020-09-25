# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import scipy.ndimage as ndi
from progressbar import ProgressBar


def read_txt(path):
    f = open(path, 'r')

    img_list = []
    label_list = []
    for line in f:
        line = line.split("\n")
        img, label = line[0].split(", ")
        img_list.append(img)
        label_list.append(label)

    return img_list, label_list


def binary_convert(labels, size, nb_class):
    y = np.zeros((size, size, nb_class))
    for i in range(nb_class):
        y[labels==i, i] = 1

    return y


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=2, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)

    return x


def random_zoom(img, label, zoom_range, gen_num):
    data_img = np.expand_dims(img, axis=0)
    data_label = np.expand_dims(label, axis=0)

    for i in range(gen_num):
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        h, w = img.shape[0], img.shape[1]
        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
        x = apply_transform(img, transform_matrix, 2, 'nearest', 0.)
        x = np.expand_dims(x, axis=0)
        y = apply_transform(label, transform_matrix, 2, 'nearest', 0.)
        y = np.expand_dims(y, axis=0)

        data_img = np.append(data_img, x, axis=0)
        data_label = np.append(data_label, y, axis=0)

    data_img = np.append(data_img, x, axis=0)
    data_label = np.append(data_label, y, axis=0)

    return data_img, data_label


def random_shift(img, label, wrg, hrg, gen_num):
    data_img = np.expand_dims(img, axis=0)
    data_label = np.expand_dims(label, axis=0)

    for i in range(gen_num):
        h, w = img.shape[0], img.shape[1]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        transform_matrix = translation_matrix 
        x = apply_transform(img, transform_matrix, 2, 'nearest', 0.)
        x = np.expand_dims(x, axis=0)
        y = apply_transform(label, transform_matrix, 2, 'nearest', 0.)
        y = np.expand_dims(y, axis=0)

        data_img = np.append(data_img, x, axis=0)
        data_label = np.append(data_label, y, axis=0)

    data_img = np.append(data_img, x, axis=0)
    data_label = np.append(data_label, y, axis=0)

    return data_img, data_label

#データ拡張
def data_augmentation(img, label, gen_num):
    x_1, y_1 = random_zoom(img, label, (0.8, 1.2), gen_num)
    x_2, y_2 = random_shift(img, label, 0.2, 0.2, gen_num)

    x = np.append(x_1, x_2, axis=0)
    y = np.append(y_1, y_2, axis=0)

    return x, y


def read_image(data_path, size):
    x = Image.open(data_path)
    x = x.resize((size, size), Image.ANTIALIAS)
    x = np.asarray(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)

    return x


def read_image_generator(data_path, label_path, size, classes, generator=False, mode='test'):
    if generator == True:
        gen_num = 10
        data_num = len(data_path) * gen_num * 2
    else:
        gen_num = 1
        data_num = len(data_path)

    x = np.zeros((data_num, size, size, 1), dtype=np.float32)
    y = np.zeros((data_num, size, size, classes), dtype=np.int32)

    count = 0
    p = ProgressBar(max_value=len(data_path))
    for i in range(len(data_path)):
        p.update(i+1)
        img = read_image(data_path[i], size)
        label_img = read_image(label_path[i], size)

        if mode == 'train':
            img, label_tmp = data_augmentation(img, label_img, gen_num)
            img = np.asarray(img, dtype=np.float32) / 255
            
            for j in range(gen_num):
                label = label_tmp[j].reshape((label_tmp[j].shape[0], label_tmp[j].shape[1]))
                label = np.array(label, dtype=np.int32)
                y[count] = binary_convert(label, size, classes)
                x[count] = img[j]
                count += 1

        else:
            img = np.asarray(img, dtype=np.float32) / 255
            x[i] = img

            label = label_img.reshape((label_img.shape[0], label_img.shape[1]))
            label = np.array(label, dtype=np.int32)
            y[i] = binary_convert(label, size, classes)

    p.finish()

    return x, y
