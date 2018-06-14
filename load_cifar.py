# 用于读取cifar10
# 输入为“cifar-10-batches-py”文件夹的路径，输出为train_images，train_label，test_images,test_label
# 用法示例train_images, train_label, test_images, test_label = load( './cifar-10-batches-py')

import numpy as np
import os
import gzip
import pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return images,labels


def load( data_dir):
    train_images, train_label = cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], data_dir)
    test_images, test_label = cifar_generator(['test_batch'], data_dir)
    return train_images, train_label, test_images, test_label