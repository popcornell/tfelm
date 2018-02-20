import tensorflow as tf
import itertools

# mnist data
from mnist_load import mnist,LOG_DIR
from mnist_load import mnist_sprites_test_path,mnist_metadata_test_path
from mnist_load import mnist_sprites_train_path,mnist_metadata_train_path

# import models
from tfelm.networks import slfn

# for embeddings in tensorboard
from utilities.embeddings import create_image_embedding_metadata


# training
from tfelm.training import k_fold_training,validation_training

# GLOBALS
input_size = 784
output_size = 10

# global variable to track number of runs
run_var = 0

##################
# HYPERPARAMETERS
##################

# WEIGHTS AND BIASES INIT
winit = (tf.random_normal, (6/(input_size+output_size))**0.5)
binit = winit

# hyperparameters dictionary

# it is passed both to the training defined in training.py and to the model defined in networks.py
# All possible combinations of hyperparameters will be evaluated. The hyperparametrs should be contained in a list

# The optim hyperparameter which refers to the optimizer which will be used in the training phase, should be
# written as a tuple, the first argument is the name of the optimizer, the second is a list used to pass arguments to the
# optimizer function. Arguments are passed by position. E.G. in the code below, 1E-2 refers to the learning rate as
# the first arg of tf.train.AdamOptimizer is the learning rate.

# The winit and binit hyperparameters should be written also as a tuple. The first argument is the distribution,
# the second is the stddev. It is possible also to have a tf.constant and as a second argument a float with the value
# of the constant

hyperpar = {'optim':[(tf.train.AdamOptimizer, [1E-2] )],  # second tuple arg should be an iterable e.g. a list
            'hidd': [500,1000], # number of hidden layers
            'TR' : [False,True], # trainable hidden layer: False for ELM layer
            'winit': [winit], # hidden layer weight initialization
            'binit': [binit], # hidden layer bias initialization
            'act' : [tf.nn.sigmoid], # activation for hidden layer
            'estop': [20], # use early stopping (np.infinity or very large number if not required)
            'epo':[200], # number of Epochs
            'bsize' : [100], # batch size
            'cr_val': [1], # use cross-validation 1 or 0 for validation set only training
            'model': [slfn] # model selection

           }


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

        global run_var

        # generate every possible combination of hyperparameters
        keys, values = zip(*hyperpar.items())

        for v in itertools.product(*values):
            comb = dict(zip(keys, v))

            print("Run: %d" % (run_var)) # Run number
            print("Training with hyperparameter set:")
            print(comb)

            k = int(comb['cr_val'])
            if k > 1:  # cross eval:

                k_fold_training(k, mnist.train.images, mnist.train.labels,
                                mnist.test.images, mnist.test.labels,
                                log_dir=LOG_DIR,
                                hpar_comb=comb,
                                input_size=784,
                                output_size=10,
                                name = 'Run_%d' % run_var,
                                emb_metadata_path = mnist_metadata_test_path,
                                sprites_meta_path=mnist_sprites_test_path,
                                check_interval=100
                                )

            else: # use mnist validation set

                validation_training(mnist.train,
                                    mnist.validation.images,mnist.validation.labels,
                                    mnist.test.images, mnist.test.labels,
                                    log_dir=LOG_DIR,
                                    hpar_comb=comb,
                                    input_size=784,
                                    output_size=10,
                                    name='run_%d' % run_var,
                                    emb_metadata_path=mnist_metadata_test_path,
                                    sprites_meta_path=mnist_sprites_test_path,
                                    check_interval=100
                                   )

            print("RUN number %d COMPLETED\n\n" % run_var)

            run_var +=  1  # next run

        print('tensorboard --logdir=%s' % LOG_DIR)

        """
        import os
        os.system('killall tensorboard')
        os.system('fuser 6006/tcp -k')  # free default tensorboard port
        os.system('tensorboard --logdir=%s' % LOG_DIR)
        """


if __name__ == '__main__':
        main()

