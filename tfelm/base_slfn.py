import tensorflow as tf


class Fdnn(object):
    'Base class for a feed-foward neural network with Dense Layers'

    def __init__(
            self,
            input_size,
            output_size,
            name='fnn'

    ):

        self.n_neurons = [input_size, output_size]
        self.w_initializer = []
        self.b_initializer = []
        self.activation = []
        self.name = name
        self.HH = None
        self.HT = None
        self.n_hidden_layer = 0
        self.H = []
        self.Hw = []
        self.Hb = []
        self.B = None
        self.y_out = None

        # start tensorflow session
        self.sess = tf.Session()

        # define graph inputs
        with tf.name_scope("input_" + self.name):
            self.x = tf.placeholder(dtype='float32', shape=[None, self.n_neurons[0]], name='x')
            self.y = tf.placeholder(dtype='float32', shape=[None, self.n_neurons[-1]], name='y')

    def add_layer(self, n_neurons, activation=tf.sigmoid, w_init='default', b_init='default'):
        # add an hidden layer
        self.n_neurons.insert(-1, n_neurons)
        self.activation.append(activation)
        self.w_initializer.append(w_init)
        self.b_initializer.append(b_init)
        self.n_hidden_layer += 1

    def compile(self):

        assert self.n_hidden_layer is not 0, "Before compiling the network at least one hidden layer should be created"
        for layer in range(self.n_hidden_layer):



            with tf.name_scope("hidden_layer_" + self.name + ("_%d" % layer)):
                if self.w_initializer[layer] is 'default' or self.b_initializer[layer] is 'default':

                    if layer == 0:
                        init_w = tf.random_normal(shape=[self.n_neurons[layer], self.n_neurons[layer + 1]],
                                                  stddev=tf.sqrt(
                                                      tf.div(2., tf.cast(self.n_neurons[layer], 'float32'))))
                    else:

                        init_w = tf.random_normal(shape=[self.n_neurons[layer], self.n_neurons[layer + 1]],
                                                  stddev=tf.sqrt(
                                                      tf.div(2., tf.cast(self.n_neurons[layer - 1], 'float32'))))

                    if self.b_initializer[layer] is not None:

                        if layer == 0:
                            init_b = tf.random_normal(shape=[self.n_neurons[layer + 1]],
                                                      stddev=tf.sqrt(
                                                          tf.div(2.,
                                                                 tf.cast(self.n_neurons[layer], 'float32'))))

                        else:

                            init_b = tf.random_normal(shape=[self.n_neurons[layer + 1]],
                                                      stddev=tf.sqrt(
                                                          tf.div(2.,
                                                                 tf.cast(self.n_neurons[layer - 1], 'float32'))))

                        self.Hb.append(tf.Variable(init_b, trainable=False))
                    else:
                        self.Hb.append(None)

                    self.Hw.append(tf.Variable(init_w, trainable=False))

                else:
                    print("Using custom inizialization for ELM: {} and layer number {}/{}".format(self.name, layer + 1,
                                                                                                  self.n_hidden_layer))

                    with tf.name_scope("custom_initialization_" + ("_%d" % layer)):
                        self.Hw.append(self.w_initializer[layer])
                        assert self.Hw[layer] or self.Hb[layer] is 'default', "Both w_initializer and b_initializer " \
                                                                              "should be provided when using custom initialization"
                        assert sorted(self.Hw[layer].shape.as_list()) == sorted([self.n_neurons[layer],
                                                                                 self.n_neurons[layer + 1]]), \
                            "Invalid shape for hidden layer weights tensor"

                        if self.b_initializer[layer] is not None:  # check
                            self.Hb.append(self.b_initializer[layer])
                            assert self.Hb[layer].shape.as_list()[0] == self.n_neurons[layer + 1], \
                                "Invalid shape for hidden layer biases tensor"
                        else:
                            self.Hb.append(None)

                if layer == 0:
                    if self.Hb[layer] is not None:
                        self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer]) + self.Hb[layer]))
                    else:
                        self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer])))
                else:
                    if self.Hb[layer] is not None:
                        self.H.append(
                            self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer]) + self.Hb[layer]))
                    else:
                        self.H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer])))

                # initialization
                if self.Hb[layer] is not None:
                    self.sess.run([self.Hw[layer].initializer, self.Hb[layer].initializer])

                else:
                    self.sess.run([self.Hw[layer].initializer])

        with tf.name_scope('output_layer_' + self.name):
            self.B = tf.Variable(tf.zeros(shape=[self.n_neurons[self.n_hidden_layer], self.n_neurons[-1]]),
                                 dtype='float32', trainable=False)
            self.y_out = tf.matmul(self.H[self.n_hidden_layer - 1], self.B, name='y_out')

        print("%s has been initialized" % self.name)

    def get_Hw_Hb(self, layer_number=-1):
        Hw = self.Hw[layer_number].eval(session=self.sess)  # get hidden layer weights matrix
        if self.Hb[layer_number] is not None:
            Hb = self.Hb[layer_number].eval(session=self.sess)  # get hidden layer biases
        else:
            Hb = None
        return Hw, Hb

    def get_B(self):
        return self.B.eval(session=self.sess)

    def get_HH(self):
        return self.HH.eval(session=self.sess)

    def __del__(self):
        self.sess.close()
