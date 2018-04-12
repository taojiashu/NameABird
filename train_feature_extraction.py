import os
import time
import warnings

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from alexnet import AlexNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings

warnings.filterwarnings("ignore") #Hide messy Numpy warnings

nb_classes = 200
epochs = 10
batch_size = 128

# Load data.
h5 = h5py.File('data.h5', 'r')
X_train = np.array(h5.get('X_train'))
X_test = np.array(h5.get('X_test'))
Y_train = np.array(h5.get('Y_train'))
Y_test = np.array(h5.get('Y_test'))
X_val = np.array(h5.get('X_val'))
Y_val = np.array(h5.get('Y_val'))
h5.close()

# Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 224, 224, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

keep_prob = tf.placeholder(tf.float32)
drop_out = tf.nn.dropout(fc7, keep_prob)

# Add the final layer for traffic sign classification.
shape = (drop_out.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
saver = tf.train.Saver()
logits = tf.nn.xw_plus_b(drop_out, fc8W, fc8b)

# Define loss, training, accuracy operations.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_op = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy', loss_op)

opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
init_op = tf.global_variables_initializer()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
tf.summary.scalar('accuracy', accuracy_op)

merged = tf.summary.merge_all()

# Train and evaluate the feature extraction model.


def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    k = 1
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch, keep_prob: k})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]


with tf.Session() as sess:
    sess.run(init_op)
    #saver.restore(sess, "./checkpoint/model.ckpt")
    #print("Model restored.")
    train_writer = tf.summary.FileWriter("./log/train", sess.graph)
    val_writer = tf.summary.FileWriter("./log/val")
    test_writer = tf.summary.FileWriter("./log/test")

    steps = 0
    for i in range(epochs):
        # training
        X_train, Y_train = shuffle(X_train, Y_train)
        t0 = time.time()
        k = 0.5
        for offset in range(0, X_train.shape[0], batch_size):
            steps += 1
            end = offset + batch_size
            summary, _ = sess.run([merged, train_op], feed_dict={features: X_train[offset:end], labels: Y_train[offset:end], keep_prob: k})
            train_writer.add_summary(summary, steps)

        val_loss, val_acc = eval_on_data(X_val, Y_val, sess)

        validation_acc_summary = tf.summary.scalar('val_accuracy', val_acc)
        validation_summary = sess.run(validation_acc_summary)
        val_writer.add_summary(validation_summary, steps)

        save_path = saver.save(sess, "./checkpoint/model_4.12.ckpt")
        print("Model saved in path: %s" % save_path)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

    _, test_acc = eval_on_data(X_test, Y_test, sess)
    test_acc_summary = tf.summary.scalar('test_accuracy', test_acc)
    test_summary = sess.run(test_acc_summary)
    test_writer.add_summary(test_summary)

    print("Test Accuracy =", test_acc)
    print("")
