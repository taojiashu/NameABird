import os

import numpy as np
from scipy.misc import imread, imresize


def load_image_labels(dataset_path=''):
    labels = {}

    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = int(class_id)

    return labels


def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}

    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path

    return paths


def load_train_test_split(dataset_path=''):
    train_images = []
    test_images = []

    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train > 0:
                train_images.append(image_id)
            else:
                test_images.append(image_id)

    return train_images, test_images


def format_dataset(dataset_path, image_path_prefix):
    image_paths = load_image_paths(dataset_path, image_path_prefix)
    image_labels = load_image_labels(dataset_path)
    train_images, test_images = load_train_test_split(dataset_path)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for image_ids, image, label in [(train_images, X_train, Y_train), (test_images, X_test, Y_test)]:
        for image_id in image_ids:
            image.append(imresize(imread(image_paths[image_id], mode="RGB"), (224, 224)))
            label.append(image_labels[image_id] - 1)

    return np.array(X_train).astype(np.float32), np.array(Y_train), np.array(X_test).astype(np.float32), np.array(Y_test)
