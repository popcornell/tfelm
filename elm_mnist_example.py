import tensorflow as tf
import os
import itertools

# mnist data
from mnist import mnist,LOG_DIR
from mnist import mnist_sprites_test_path,mnist_metadata_test_path
from mnist import mnist_sprites_train_path,mnist_metadata_train_path

# elm/nn layer definition
from tf_elm import nn_layer,output_layer


# for embeddings
from embeddings import create_image_embedding_metadata
from tensorflow.contrib.tensorboard.plugins import projector

# globals
input_size = 784
output_size = 10

run_var = 0


# standard initialization

binit = (tf.random_normal, 3*(1/input_size)**0.5)
winit = binit # same for biases and weights

hyperpar = {'lr': [1E-2],
            'hidd': [1000,500], # number of hidden layers
            'TR' : [False,True], # trainable hidden layer: False for ELM layer
            'winit': [winit], # hidden layer weight initialization
            'binit': [binit], # hidden layer bias initialization
            'nstep': [2000], # number of training steps
            'act' : [tf.nn.leaky_relu],
            #'e_stop': [False], # use early stopping #TODO
            'bsize' : [100], # batch size
            #'cr_val': [0] # use cross-validation 0 is false #TODO
           }


def slfn_model(input_size, output_size, hparams):

    # TODO regularization

    hidden_layer_size = hparams['hidd']
    assert hidden_layer_size > 0, "The size of hidden layer must be greater than 0"

    learning_rate = hparams['lr']
    trainable = hparams['TR']
    n_steps = hparams['nstep']
    batch_size = hparams['bsize']


    tf.reset_default_graph() # reset graph
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, input_size], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, output_size], name="labels")

    # for tensorboard visualization, 1 for Grayscale this in load mnist
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)


    # create hidden layer
    hidden = nn_layer(x, input_size, hidden_layer_size, name='elm',
                   trainable=trainable,
                   act=hparams['act'],
                   winit=hparams['winit'],
                   binit=hparams['binit']
                   )

    # output layer
    logits = output_layer(hidden, hidden_layer_size, output_size,
                           name='output'
                          )

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_), name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    hparams_str = '/Run_%d' % (run_var)

    # embeddings for training visualization
    summ = tf.summary.merge_all()
    embedding_input = tf.reshape(x, [-1,input_size])
    embedding = tf.Variable(tf.zeros([hidden_layer_size, input_size]), name="test_run_%d" % (run_var))
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_DIR + hparams_str)
    writer.add_graph(sess.graph)

    # print to txt file current hyperparameters for convenience
    with open(LOG_DIR + hparams_str+"/hyperparam.txt", "w") as text_file:
         print(hparams, file=text_file)


    # setup config for  tensorboard projector
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path =  mnist_sprites_test_path
    embedding_config.metadata_path =  mnist_metadata_test_path

    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    # TODO better training section, still it is enough to get insight into ELMs and compare them to MLPs

    for i in range(n_steps):
        batch = mnist.train.next_batch(batch_size)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y_: batch[1]})
            writer.add_summary(s, i)
        if i % 500 == 0:
            print("Train Accuracy on train set at step %d is: %.2f" % (i, train_accuracy))
            sess.run(assignment, feed_dict={x: mnist.test.images[:hidden_layer_size],
                                             y_: mnist.test.labels[:hidden_layer_size]})
            saver.save(sess, os.path.join(LOG_DIR + hparams_str, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


def embeddings_init(max_hidden_size):
    # create mnist metadata and sprites
    # metadata and sprites should be greater than hidden layer size
    # this is because we want to be able to visualize mnist data in the hidden layer space
    # to get insight on neural network internal representation of data

    # NOTE: tensorboard can output a warning indicating that
    # Number of tensors does not match the numer of lines in metadata,
    # however it will not be a problem till the former is lesser than the latter

    # create metadata for train embeddings
    x = mnist.train.images[:max_hidden_size]
    y = mnist.train.labels[:max_hidden_size]


    create_image_embedding_metadata(input_data=x,
                                    labels=y,
                                    embedding_metadata_path=mnist_metadata_train_path,
                                    sprites_path=mnist_sprites_train_path
                                    )


    # create metadata for test embeddings
    x = mnist.test.images[:max_hidden_size]
    y = mnist.test.labels[:max_hidden_size]

    create_image_embedding_metadata(input_data=x,
                                    labels=y,
                                    embedding_metadata_path=mnist_metadata_test_path,
                                    sprites_path=mnist_sprites_test_path
                                    )



def main():


        max_hidden_layer_size = max(hyperpar['hidd'])

        # create embeddings metadata
        embeddings_init(max_hidden_layer_size)


        # generate every possible combination of hyperparameters
        keys, values = zip(*hyperpar.items())

        global run_var

        for v in itertools.product(*values):
            comb = dict(zip(keys, v))

            print("Run: %d" % (run_var))
            print("Training with hyperparameter set:")
            print(comb)

            slfn_model(input_size, output_size, hparams=comb)
            run_var = run_var+1

        print('tensorboard --logdir=%s' % LOG_DIR)

        """
        import os
        os.system('killall tensorboard')
        os.system('fuser 6006/tcp -k')  # free default tensorboard port
        os.system('tensorboard --logdir=%s' % LOG_DIR)
        """

if __name__ == '__main__':
        main()





