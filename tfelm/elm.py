import tensorflow as tf
from tfelm.base_slfn import Fdnn

import time, math
import os
import numpy as np


class ELM(Fdnn):
    'Single Layer ELM object'

    def __init__(self, input_size,
                 output_size,
                 type='c',
                 l2norm=None,
                 name="elm"
                 ):

        super(__class__, self).__init__(input_size,
                                        output_size,
                                        name=name,
                                        )

        self.l2norm = l2norm
        if self.l2norm is None:
            self.l2norm = 1e-05  # from CuDNN.h

        self.HH_HT_op = None
        self.B_op = None
        self.type = type

    def compile(self):

        assert self.n_hidden_layer is 1, \
            "elm object supports only one hidden layer and before compiling one hidden layer should be created"

        super(__class__, self).compile()

        # define training structure
        with tf.name_scope("training_" + self.name):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[self.n_hidden_layer], dtype=tf.float32),
                                              tf.cast(self.l2norm, tf.float32)),
                                  name='HH', trainable=False)

            self.HT = tf.Variable(tf.zeros([self.n_neurons[self.n_hidden_layer], self.n_neurons[-1]]), name='HT', trainable=False)

            self.HH_HT_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.H[- 1], self.H[- 1], transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.H[- 1], self.y, transpose_a=True)), name='HH_HT_op'
            )

            self.B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT), name='B_op')

        self.sess.run([self.HH.initializer, self.HT.initializer])

    def train(self, tf_iterator, n_batches=None):

        next_batch = tf_iterator.get_next()

        t0 = time.time()

        batch = 1
        while True:
            try:
                start = time.time()
                # get next batch of data
                x_batch, y_batch = self.sess.run(next_batch)

                # Run the training op
                self.sess.run(self.HH_HT_op, feed_dict={self.x: x_batch,
                                                        self.y: y_batch})

                if n_batches is not None:
                    if batch % 25 == 0:
                        eta = (time.time() - start) * (n_batches - batch)
                        eta = '%d:%02d' % (eta // 60, eta % 60)
                        print("{}/{} ETA:{}".format(batch, n_batches, eta))
                    batch += 1

            except tf.errors.OutOfRangeError or IndexError:
                break

        self.sess.run(self.B_op)
        print("Training of ELM {} ended in {}:{}:{:5f}".format(self.name, math.floor((time.time() - t0) // 3600),
                                                               math.floor((time.time() - t0) % 3600 // 60),
                                                               ((time.time() - t0) % 3600 % 60)))
        print("#" * 100)

        self.saver = tf.train.Saver()

    def fit(self, x, y, batch_size=1024):

        print("Creating Dataset and Iterator from tensors")

        # from https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays
        # recommended method:

        n_batches = int(np.ceil(x.shape[0] / batch_size))

        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(batch_size=batch_size)
        iterator = dataset.make_initializable_iterator()

        self.sess.run(iterator.initializer, feed_dict={self.x: x,  # TODO is there a better way ?
                                                       self.y: y})

        self.train(iterator, n_batches)

        # re-initialize the iterator
        self.sess.run(iterator.initializer, feed_dict={self.x: x,
                                                       self.y: y})

        if self.type is 'c':
            train_perf = self.evaluate(tf_iterator=iterator, metric='acc')


        elif self.type is 'r':
            train_perf = self.evaluate(tf_iterator=iterator, metric='mse')

        return train_perf

    def evaluate(self, x=None, y=None, tf_iterator=None, metric='acc', batch_size=1024):

        if tf_iterator is None:
            # create iterator
            assert x is not None and y is not None, \
                "Both feature and labels arrays should be provided when an iterator is not passed to the function"

            dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(batch_size=batch_size)
            tf_iterator = dataset.make_initializable_iterator()

            self.sess.run(tf_iterator.initializer, feed_dict={self.x: x,  # TODO is there a better way ?
                                                              self.y: y})

        print("Evaluating network performance")

        next_batch = tf_iterator.get_next()

        if metric is 'acc':
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_out, 1))
            eval_metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        elif metric is 'mse':
            eval_metric = tf.reduce_mean(tf.squared_difference(self.y_out, self.y, name='mse'))

        else:
            ValueError("Invalid performance metric, use mse or acc")

        metric_vect = []

        while True:
            try:
                x_batch, y_batch = self.sess.run(next_batch)

                metric_vect.append(self.sess.run(eval_metric, feed_dict={self.x: x_batch, self.y: y_batch}))
            except tf.errors.OutOfRangeError:
                break

        mean_metric = np.mean(metric_vect)

        if metric is 'acc':
            print('Accuracy: %.7f' % mean_metric)
        elif metric is 'mse':
            print('MSE: .7f' % mean_metric)

        return mean_metric

    def predict(self, x=None, tf_iterator=None, batch_size=1024):

        if tf_iterator is None:
            # create iterator
            assert x is not None, \
                "Feature array should be provided when an iterator is not passed to the function"

            dataset = tf.data.Dataset.from_tensor_slices((self.x)).batch(batch_size=batch_size)
            tf_iterator = dataset.make_initializable_iterator()

            self.sess.run(tf_iterator.initializer, feed_dict={self.x: x})

        print("Predicting...")

        next_batch = tf_iterator.get_next()

        y_out = []
        while True:
            try:
                x_batch = self.sess.run(next_batch)
                y_out.extend(self.sess.run(self.y_out, feed_dict={self.x: x_batch}))

            except tf.errors.OutOfRangeError:
                break

        print("Done")

        return np.asarray(y_out)

    def reset(self):

        self.compile()

    def save(self, ckpt_path=None):

        if ckpt_path is None:
            ckpt_path = os.path.join(os.getcwd(), self.name)

        self.saver.save(self.sess, ckpt_path, write_meta_graph=True)
        print("Model saved in path: %s" % ckpt_path)

    def load(self, ckpt_path=None):

        if ckpt_path is None:
            ckpt_path = os.path.join(os.getcwd(), self.name + '.ckpt')

        saver = tf.train.import_meta_graph(ckpt_path + '.meta')

        saver.restore(self.sess, ckpt_path)

        # TODO fix load
        ''' 
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("input_" + self.name + ':' + "x:0")
        self.y = graph.get_tensor_by_name("y")
        self.y_out = graph.get_operation_by_name("y_out:0")
        self.HH_HT_op = graph.get_operation_by_name("HH_HT_op:0")
        self.B_op = graph.get_operation_by_name("B_op:0")

        print("Model restored.")
        '''


