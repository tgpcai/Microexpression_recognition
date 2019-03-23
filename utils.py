import collections
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes, random_seed


def load_data(data_file):
    data = pd.read_csv(data_file)
    pixels = data['pixels'].tolist()
    width = 48
    height = 48
    faces = []
    for pixel_sequence in pixels:
        # 从csv中获取人脸的数据
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        # 把脸的数据变为48*48像素，利用plt.imshow即可打印出图片
        face = np.asarray(face).reshape(width, height)
        faces.append(face)
    # 把faces从列表变为三维矩阵。(35887,)----->(35887,48,48)
    faces = np.asarray(faces)
    # 添加维度，将faces从(35887,48,48)------>(35887,48,48,1)
    faces = np.expand_dims(faces, -1)
    # one-hot编码，把属于该类表情置1，其余为0，并转换为矩阵
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions


class DataSet(object):
    def __init__(self, images, labels, reshape=True, dtype=dtypes.float32, seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        if reshape:
            # 将images(35887,48,48,1)变为(35887,2304)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],images.shape[1]*images.shape[2])

        # 类型转换，并进行灰度处理
        if dtype == dtypes.float32:
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        # 设置私有属性
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self.num_examples

    @property
    def epochs_completed(self):
        self._epochs_completed

    # 批量获取训练数据
    def next_batch(self, batch_size,shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            # 打乱顺序
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # 当剩余的数据不够一次batch_size，就在之前的数据中随机选取并进行组合
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def input_data(train_dir, dtype = dtypes.float32, reshape = True, seed=None):
    training_size = 28709
    validation_size = 3589
    test_size = 3589

    train_faces, train_emotions = load_data(train_dir)
    print("Data load success!")

    # 验证数据
    validation_faces = train_faces[training_size: training_size + validation_size]
    validation_emotions = train_emotions[training_size: training_size + validation_size]

    # 测试数据
    test_faces = train_faces[training_size + validation_size:]
    test_emotions = train_emotions[training_size + validation_size:]

    # 训练数据
    train_faces = train_faces[: training_size]
    train_emotions = train_emotions[: training_size]

    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    train = DataSet(train_faces, train_emotions, reshape=reshape,)
    validation = DataSet(validation_faces, validation_emotions, dtype=dtype, reshape=reshape, seed=seed)
    test = DataSet(test_faces, test_emotions, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, validation=validation, test=test)

