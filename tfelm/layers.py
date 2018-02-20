import tensorflow as tf
import os


def bias_tensor(shape, binit, trainable=True):
    """Initialize biases.
    shape (int): bias tensor shape
    trainable (bool): set the bias to be trainable or not.
                      For ELM it should be set to not trainable.
    binit (tuple): first value specify the distribution
                   second value specifies stddev of distribution.
    """
    if binit[0] is tf.constant:
       initial = binit[0](value=binit[1], shape=shape)
    else:
       initial = binit[0](stddev=binit[1], shape=shape)
    return tf.Variable(initial, trainable)


def weight_tensor(shape, winit, trainable=True):
    """Initialize weights.
        shape (int): weight tensor shape
        trainable (bool): set the bias to be trainable or not.
                          For ELM it should be set to not trainable.
        binit (tuple): first value specify the distribution
                       second value specifies stddev of distribution.
        """
    weights =  winit[0](shape, stddev=winit[1])
    return tf.Variable(weights, trainable)


def variable_summaries(var):
    """(for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, name,
           trainable = True,
           act=tf.nn.relu,
           winit = (tf.random_normal,1),
           binit= (tf.constant,0.1)
          ):

    """for output layer set act=tf.identity and binit=(tf.constant,0)
    """
    with tf.name_scope(name):

        with tf.name_scope('weights'):
            weights = weight_tensor([input_dim, output_dim], winit, trainable)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_tensor([output_dim], binit, trainable)
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        activation_name = "activation_%s" %(name)
        tf.summary.histogram(activation_name, activations)
    return activations


def output_layer(input_tensor, input_dim, output_dim, name,
                  trainable = True,
                 winit = (tf.random_normal,1)
                ):

    """for output layer with bias create a nn_layer with sact=tf.identity
    """
    with tf.name_scope(name):

        with tf.name_scope('weights'):
            weights = weight_tensor([input_dim, output_dim], winit, trainable)
            variable_summaries(weights)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights)
            tf.summary.histogram('pre_activations', preactivate)

    return preactivate

def main(): # testing purpose only

    LOGDIR = os.path.join(os.getcwd(), "test_tf_elm_log")

    tf.reset_default_graph()
    sess = tf.Session()

    input_size = 32
    output_size = 2
    #input tensor placeholder
    x = tf.placeholder(tf.float32, shape=[None, input_size], name="x")

    with tf.name_scope("elm_layer"):
         elm_layer= nn_layer(x, input_size, output_size,
                         name='elm1',
                         trainable=False,
                         act=tf.nn.sigmoid,
                         binit=(tf.random_normal, 3*(1/input_size)**0.5)
                         )

    with tf.name_scope("nn_layer"):
          dense_layer= nn_layer(x, input_size, output_size,
                         name='nn1',
                         trainable=True,
                         act=tf.nn.sigmoid
                         )  # unitary stddev for initialization be careful!

    # tensorboard graph

    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    print("Check graph on tensorboard:")

    print('tensorboard --logdir=%s' % LOGDIR)



if __name__ == '__main__':
   main()
