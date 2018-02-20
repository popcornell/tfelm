import tensorflow as tf

from tfelm.layers import nn_layer,output_layer


def slfn(x , y_,  hparams):

    # TODO regularization
    output_size = int(y_.shape[1])
    input_size = int(x.shape[1])

    hidden_layer_size = hparams['hidd']
    assert hidden_layer_size > 0, "The size of hidden layer must be greater than 0"

    trainable = hparams['TR']


    # for tensorboard visualization, 1 for Grayscale this in load mnist
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)


    # create hidden layer
    with tf.name_scope("hidden_layer"):

        hidden = nn_layer(x, input_size, hidden_layer_size, name='hidden',
                          trainable=trainable,
                          act=hparams['act'],
                          winit=hparams['winit'],
                          binit=hparams['binit']
                         )


    # output layer
    with tf.name_scope("output_layer"):

        logits = output_layer(hidden, hidden_layer_size, output_size,
                          name='output_layer')


        # e.g. output layer with bias
        '''
        logits = nn_layer(hidden, hidden_layer_size,output_size,
                          name='output', act=tf.identity, winit=hparams['winit'],
                          binit=hparams['binit']
                          )
        '''

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_), name="cross_entropy")

    with tf.name_scope("train"):
        train_step = hparams['optim'][0](*hparams['optim'][1]).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return  train_step, accuracy, cross_entropy