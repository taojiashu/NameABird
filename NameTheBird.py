import time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread, imresize

from alexnet import AlexNet

bird_names = pd.read_csv('bird_classes.csv')
nb_classes = 200

x = tf.placeholder(tf.float32, (None, 224, 224, 3))
resized = tf.image.resize_images(x, (227, 227))

fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
saver = tf.train.Saver()
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver.restore(sess, "./checkpoint/model_4.12.ckpt")
print("Model restored.")

# Read Images
im1 = imresize(imread("Brewer_Blackbird.jpg", mode="RGB"), (224, 224)).astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imresize(imread("Prothonotary_Warbler.jpg", mode="RGB"), (224, 224)).astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (bird_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
