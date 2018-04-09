import os

import numpy as np
from scipy.misc import imread, imresize


# Modified from
# https://github.com/Hezi-Resheff/Oreilly-Blog/blob/master/01_Transfer_Learning_Multiple_Pre_Trained/cub_util.py
class CUB200(object):
    """
    Helper function to load the CUB-200 dataset
    """

    def __init__(self, path, image_size=(224, 224)):
        self._path = path
        self._size = image_size

    def _classes(self):
        return os.listdir(self._path)

    def _load_image(self, category, im_name):
        return imresize(imread(os.path.join(self._path, category, im_name), mode="RGB"), self._size)

    def load_dataset(self):
        classes = self._classes()
        all_images = []
        all_labels = []

        for c in classes:
            d = os.path.join(self._path, c)
            if os.path.isdir(d):
                class_images = os.listdir(d)

                for image_name in class_images:
                    all_images.append(self._load_image(c, image_name))
                    label = int(c[:3]) - 1  # label will be from 0 to 199
                    all_labels.append(label)
        return np.array(all_images).astype(np.float32), np.array(all_labels)
