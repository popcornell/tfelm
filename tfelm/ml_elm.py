from tfelm.elm import ELM
import numpy as np
import tensorflow as tf
from datetime import datetime
import time, math


class ML_ELM(ELM):

    def __init__(self, input_size,
                 output_size,
                 type='c',
                 name="ml_elm",
                 ):

        super(__class__, self).__init__(input_size,
                                        output_size,
                                        l2norm=None,  # will be overridden
                                        type=type,
                                        name=name,
                                        )

        self.l2norm = []  # overrides super attribute
        self.ae_B = []
        self.ae_y_out = []
        self.ae_Hw = []
        self.ae_H = []

    def add_layer(self, n_neurons, activation=tf.tanh, w_init='default', b_init='default', l2norm=None):

        super(__class__, self).add_layer(n_neurons, activation, w_init, b_init)

        if l2norm is None:
            l2norm = 1e-05

        self.l2norm.append(l2norm)

    def _compile_ae(self, layer):

        with tf.name_scope("autoenc_of_" + self.name + ("_n_%d" % layer)):
            if self.w_initializer[layer] is 'default' or self.b_initializer[layer] is 'default':
                init_w = tf.random_normal(shape=[self.n_neurons[layer], self.n_neurons[layer + 1]],
                                          stddev=tf.sqrt(tf.div(2.,
                                                                tf.add(tf.cast(self.n_neurons[layer - 1], 'float32'),
                                                                       tf.cast(self.n_neurons[layer + 1], 'float32')))))

                if self.b_initializer[layer] is not None:
                    init_b = tf.random_normal(shape=[self.n_neurons[layer + 1]],
                                              stddev=tf.sqrt(tf.div(2.,
                                                                    tf.add(
                                                                        tf.cast(self.n_neurons[layer - 1], 'float32'),
                                                                        tf.cast(self.n_neurons[layer + 1],
                                                                                'float32')))))

                    self.Hb.append(tf.Variable(init_b, trainable=False))
                else:
                    self.Hb.append(None)

                self.ae_Hw.append(tf.Variable(init_w, trainable=False))

            else:
                print("Using custom initialization for AE-ELM: {} and layer number {}/{}".format(self.name, layer + 1,
                                                                                                 self.n_hidden_layer))

                with tf.name_scope("custom_initialization_" + ("_%d" % layer)):

                    self.ae_Hw.append(self.w_initializer[layer])

                    assert self.ae_Hw[layer] or self.Hb[layer] is 'default', "Both w_initializer and b_initializer " \
                                                                             "should be provided when using custom initialization"

                    assert sorted(self.ae_Hw[layer].shape.as_list()) == sorted([self.n_neurons[layer],
                                                                                self.n_neurons[layer + 1]]), \
                        "Invalid shape for hidden layer weights tensor"

                    if self.b_initializer[layer] is not None:
                        self.Hb.append(self.b_initializer[layer])
                        assert self.Hb[layer].shape.as_list()[0] == self.n_neurons[layer + 1], \
                            "Invalid shape for hidden layer biases tensor"
                    else:
                        self.Hb.append(None)

            if layer == 0:
                if self.Hb[layer] is not None:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.x, self.ae_Hw[layer]) + self.Hb[layer]))
                else:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.x, self.ae_Hw[layer])))
            else:
                if self.Hb[layer] is not None:
                    self.ae_H.append(
                        self.activation[layer](tf.matmul(self.H[layer - 1], self.ae_Hw[layer]) + self.Hb[layer]))
                else:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.ae_Hw[layer])))

                    # take as input the activation of the first layer of the final structure

            # initialization
            if self.Hb[layer] is not None:
                self.sess.run([self.ae_Hw[layer].initializer, self.Hb[layer].initializer])

            else:
                self.sess.run([self.ae_Hw[layer].initializer])

            with tf.name_scope('output_layer_' + self.name):
                self.ae_B.append(tf.Variable(tf.zeros(shape=[self.n_neurons[layer + 1], self.n_neurons[layer]]),
                                             dtype='float32', trainable=False))

                self.ae_y_out.append(tf.matmul(self.ae_H[layer], self.ae_B[layer]))

            with tf.name_scope("mean_squared_error"):
                if layer == 0:
                    metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer],
                                                                  self.x, name='mse'))
                else:
                    metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer],
                                                                  self.H[layer - 1], name='mse'))

            print("AE Network parameters have been initialized")

    def _compile_layer(self, layer):

        self.Hw.append(tf.transpose(self.ae_B[layer]))

        with tf.name_scope("hidden_layer_of_" + self.name + ("_n_%d" % layer)):
            # no biases
            if layer == 0:
                self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer])))
            else:
                self.H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer])))

    def compile(self):
        pass

    def _train_ae(self, layer, tf_iterator, n_batches=None):

        self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[layer + 1], dtype=tf.float32),
                                          tf.cast(self.l2norm[layer], tf.float32)),
                              name='HH', trainable=False)

        self.HT = tf.Variable(tf.zeros([self.n_neurons[layer + 1], self.n_neurons[layer]]), name='HT', trainable=False)

        if layer == 0:

            train_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.ae_H[layer],
                                                 self.ae_H[layer],
                                                 transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.ae_H[layer], self.x,
                                                 transpose_a=True))
            )

        else:

            train_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.ae_H[layer],
                                                 self.ae_H[layer],
                                                 transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.ae_H[layer], self.H[layer - 1],
                                                 transpose_a=True))
            )

        B_op = tf.assign(self.ae_B[layer], tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        self.sess.run([self.HH.initializer, self.HT.initializer])

        next_batch = tf_iterator.get_next()

        t0 = time.time()
        print("{} Start training...".format(datetime.now()))

        batch = 1
        while True:
            try:
                start = time.time()
                # get next batch of data

                x_batch, y_batch = self.sess.run(next_batch)

                # Run the training op

                self.sess.run(train_op, feed_dict={self.x: x_batch})
                if n_batches is not None:
                    if batch % 25 == 0:
                        eta = (time.time() - start) * (n_batches - batch)
                        eta = '%d:%02d' % (eta // 60, eta % 60)
                        print("{}/{} ETA:{}".format(batch, n_batches, eta))

                    batch += 1
            except tf.errors.OutOfRangeError:
                break

        self.sess.run(B_op)

        print("Training of AE {} ended in {}:{}:{:5f}".format(self.name, math.floor((time.time() - t0) // 3600),
                                                              math.floor((time.time() - t0) % 3600 // 60),
                                                              ((time.time() - t0) % 3600 % 60)))

    def _eval_ae(self, tf_iterator, layer):

        with tf.name_scope("mean_squared_error"):
            if layer == 0:
                metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer], self.x, name='mse'))
            else:
                metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer], self.H[layer - 1], name='mse'))

            metric_vect = []

            next_batch = tf_iterator.get_next()

            print("Evaluating AE performance...")

            while True:
                try:
                    x_batch, _ = self.sess.run(next_batch)

                    metric_vect.append(self.sess.run(metric, feed_dict={self.x: x_batch}))
                except tf.errors.OutOfRangeError:
                    break

            mean_metric = np.mean(metric_vect)

            print('MSE: %.7f' % mean_metric)
            print('#' * 100)

    def train(self, tf_iterator, tf_iterator_init_op, n_batches=None):

        t0 = time.time()
        print("{} ML-ELM Start training...".format(datetime.now()))

        for layer in range(self.n_hidden_layer - 1):
            tf_iterator_init_op()
            self._compile_ae(layer)
            self._compile_layer(layer)
            print('Training AE %d/%d' % (layer + 1, self.n_hidden_layer - 1))

            with tf.name_scope("ae_training_n_%d_of_%s" % (layer, self.name)):
                self._train_ae(layer, tf_iterator, n_batches)
                tf_iterator_init_op()
                self._eval_ae(tf_iterator, layer)

        tf_iterator_init_op()
        # initialize and compile last layer

        print("Initializing last layer ELM...")

        with tf.name_scope("ELM_layer_" + self.name + ("_%d" % self.n_hidden_layer)):
            if self.w_initializer[-1] is 'default' or self.b_initializer[-1] is 'default':

                init_w = tf.random_normal(shape=[self.n_neurons[-3], self.n_neurons[-2]],
                                          stddev=tf.sqrt(tf.div(2.,
                                                                tf.add(tf.cast(self.n_neurons[-3], 'float32'),
                                                                       tf.cast(self.n_neurons[-1], 'float32')))))
                '''
                init_w = tf.glorot_normal_initializer()
                '''

                if self.b_initializer[-1] is not None:

                    init_b = tf.random_normal(shape=[self.n_neurons[-2]],
                                              stddev=tf.sqrt(tf.div(2.,
                                                                    tf.add(tf.cast(self.n_neurons[-3], 'float32'),
                                                                           tf.cast(self.n_neurons[-1],
                                                                                   'float32')))))

                    self.Hb.append(tf.Variable(init_b, trainable=False))
                else:
                    self.Hb.append(None)

                self.Hw.append(tf.Variable(init_w, trainable=False))

            else:
                print("Using custom initialization for ELM: {} and layer number {}/{}".format(self.name,
                                                                                              self.n_hidden_layer,
                                                                                              self.n_hidden_layer))

                with tf.name_scope("custom_initialization_" + ("_%d" % self.n_hidden_layer)):
                    self.Hw.append(self.w_initializer[-1])
                    assert self.Hw[-1] or self.Hb[-1] is 'default', "Both w_initializer and b_initializer " \
                                                                    "should be provided when using custom initialization"
                    assert sorted(self.Hw[-1].shape.as_list()) == sorted([self.n_neurons[-3],
                                                                          self.n_neurons[-2]]), \
                        "Invalid shape for hidden layer weights tensor"

                    if self.b_initializer[-1] is not None:  # check
                        self.Hb.append(self.b_initializer[-1])
                        assert self.Hb[-1].shape.as_list()[0] == self.n_neurons[-2], \
                            "Invalid shape for hidden layer biases tensor"
                    else:
                        self.Hb.append(None)

            if self.Hb[-1] is not None:

                self.sess.run([self.Hw[-1].initializer, self.Hb[-1].initializer])

            else:
                self.sess.run([self.Hw[-1].initializer])

            if self.Hb[-1] is not None:
                self.H.append(self.activation[-1](tf.matmul(self.H[-1], self.Hw[-1]) + self.Hb[-1]))
            else:
                self.H.append(self.activation[-1](tf.matmul(self.H[-1], self.Hw[-1])))

        with tf.name_scope('output_layer_' + self.name):
            self.B = tf.Variable(tf.zeros(shape=[self.n_neurons[-2], self.n_neurons[-1]]),
                                 dtype='float32', trainable=False)
            self.y_out = tf.matmul(self.H[-1], self.B)

        # last layer training

        # define training structure for last layer
        print("Training last layer ELM...")

        with tf.name_scope("training_" + self.name):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[-2], dtype=tf.float32),
                                              tf.cast(self.l2norm[-1], tf.float32)),
                                  name='HH', trainable=False)

            self.HT = tf.Variable(tf.zeros([self.n_neurons[-2], self.n_neurons[-1]]), name='HT', trainable=False)

            self.HH_HT_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.H[-1], self.H[-1],
                                                 transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.H[-1], self.y, transpose_a=True))
            )

            self.B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        # initialize variables
        self.sess.run([self.HH.initializer, self.HT.initializer])

        super(__class__, self).train(tf_iterator, n_batches)

        print("#" * 100)

        print(
            "TOTAL Training of ML_ELM {} ended in {}:{}:{:5f}".format(self.name, math.floor((time.time() - t0) // 3600),
                                                                      math.floor((time.time() - t0) % 3600 // 60),
                                                                      ((time.time() - t0) % 3600 % 60)))
        print("#" * 100)


    def fit(self, x, y, batch_size=1024):

        assert self.n_hidden_layer > 1, "Before compiling the network at least two hidden layers should be created"

        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(batch_size=batch_size)
        iterator = dataset.make_initializable_iterator()

        nb = int(np.ceil(x.shape[0] / batch_size))

        iterator_init_op = lambda: self.sess.run(iterator.initializer, feed_dict={self.x: x, self.y: y})

        self.train(iterator,iterator_init_op,nb)

        iterator_init_op()

        if self.type is 'c':
            train_perf = self.evaluate(tf_iterator=iterator, metric='acc')


        elif self.type is 'r':
            train_perf = self.evaluate(tf_iterator=iterator, metric='mse')

        return train_perf


def main():
    import keras
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import accuracy_score

    mnist = keras.datasets.mnist
    train, test = mnist.load_data(os.getcwd() + "/elm_tf_test" + "mnist.txt")
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28 * 28)

    input_size = 784
    output_size = 10

    batch_size = 1024
    n_epochs = 1

    datagen = ImageDataGenerator(
        # rotation_range=1,
        # width_shift_range=0.05,
        # height_shift_range=0.05
    )

    datagen.fit(x_train)

    def gen():
        n_it = 0
        batches_per_epochs = len(x_train) // batch_size
        for x, y in datagen.flow(x_train, y_train, batch_size=batch_size):
            x = x.reshape(batch_size, 28 * 28)
            if n_it % 100 == 0:
                print("generator iteration: %d" % n_it)
            yield x, y
            n_it += 1
            if n_it >= batches_per_epochs * n_epochs:
                break

    data = tf.data.Dataset.from_generator(generator=gen,
                                          output_shapes=((batch_size, 28 * 28,), (batch_size, 10,)),
                                          output_types=(tf.float32, tf.float32))

    iterator = data.make_initializable_iterator()

    # Parameters
    input_size = 28 * 28
    output_size = 10

    # https://www.wolframalpha.com/input/?i=plot+sign(x)*((k*abs(x)))%2F(k-a*abs(x)%2B1),+x%3D-255..255,++k%3D-100,+a%3D1
    def tunable_sigm(t):
        k = -130.
        a = 1.
        half = tf.div(tf.multiply(k, tf.abs(t)), tf.add(1., tf.add(k, -a * tf.abs(t))))
        return tf.multiply(tf.sign(t), half)

    init_w = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[input_size, 700]))
    init_b = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[700]))
    init_w2 = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[700, 700]))
    # def sigm_zero_mean(x):
    #   return tf.subtract(1., tf.sigmoid(x))
    # def dual_relu(x):
    #   return tf.sign(x)*tf.nn.relu(x)
    # , savedir=(os.getcwd() + "/ml_elm")

    n_batches = n_epochs * (len(x_train) // batch_size)

    ml_elm1 = ML_ELM(input_size, output_size)
    ml_elm1.add_layer(4000, activation=tf.tanh)
    ml_elm1.add_layer(4000, activation=tf.tanh)
    ml_elm1.add_layer(4000, activation=tf.tanh)
    iterator_init_op = lambda: ml_elm1.sess.run(iterator.initializer)
    ml_elm1.train(iterator, iterator_init_op, n_batches=n_batches)
    iterator_init_op()
    print("TRAIN perf")
    ml_elm1.evaluate(tf_iterator=iterator)
    print("TEST perf")
    ml_elm1.evaluate(x_test, y_test)

    # os.system('tensorboard --logdir=%s' % os.getcwd() + "/ml_elm")
    # del ml_elm1
    '''
    elm1 = elm(784, 10)
    elm1.add_layer(4000)
    elm1.compile()
    elm1.train(x_train, y_train)
    acc = elm1.evaluate(x_test, y_test)
    print(acc)
    '''

    '''
    import hpelm
    elm = hpelm.ELM(x_train.shape[1], 784, classification="r", accelerator="GPU", precision='single', batch=500)
    elm.add_neurons(4000, 'sigm')
    print(str(elm))
    # Training model
    elm.train(x_train, x_train, 'r')
    y_train_predicted = elm.predict(x_train)
    print('Training Error: ', (elm.error(x_train, y_train_predicted)))
    '''


if __name__ == '__main__':
    main()
