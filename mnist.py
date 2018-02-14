import os
from tensorflow.contrib.learn import datasets
from embeddings import create_image_embedding, create_image_embedding_metadata

LOG_DIR = os.path.join(os.getcwd(), "MNIST_LOG")

mnist = datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)


embeddings_path = os.path.join(os.getcwd())

# training set metadata and sprites paths

mnist_sprites_train_path = os.path.join(embeddings_path, 'mnistdigits_train.png')
mnist_metadata_train_path = os.path.join(embeddings_path, 'mnist_metadata_train.tsv')


# test set metadata and sprites paths

mnist_sprites_test_path = os.path.join(embeddings_path, 'mnistdigits_test.png')
mnist_metadata_test_path = os.path.join(embeddings_path, 'mnist_metadata_test.tsv')



def main():  # testing purpose only

    #imports here because only for testing
    import tensorflow as tf
    import numpy as np
    from tensorflow.contrib.tensorboard.plugins import projector

    # take n istances from training set to be visualized in tboard projector
    x, y = mnist.train.next_batch(500)

    print("first image label for training set is:")
    print(np.argmax(y[0]))

    # visualize mnist data in tensorboard projector
    config = projector.ProjectorConfig()

    # create image metadata
    create_image_embedding_metadata(input_data=x,
                                    labels=y,
                                    embedding_metadata_path=mnist_metadata_train_path,
                                    sprites_path=mnist_sprites_train_path
                                    )

    create_image_embedding(name='MNIST_train',
                                          input_data=x,
                                          image_shape= [28, 28],
                                          embedding_dir=LOG_DIR,
                                          metadata_path= mnist_metadata_train_path,
                                          sprites_path=mnist_sprites_train_path,
                                          config = config
                                          )

    # take n istances from training set to be visualized in tboard projector
    x, y = mnist.test.next_batch(500)

    print("first image label for test set is:")
    print(np.argmax(y[0]))

    # visualize mnist data in tensorboard projector

    # create image metadata
    create_image_embedding_metadata(input_data=x,
                                    labels=y,
                                    embedding_metadata_path=mnist_metadata_test_path,
                                    sprites_path=mnist_sprites_test_path
                                    )



    create_image_embedding(name='MNIST_test',
                           input_data=x,
                           image_shape=[28, 28],
                           embedding_dir=LOG_DIR,
                           metadata_path=mnist_metadata_test_path,
                           sprites_path=mnist_sprites_test_path,
                           config = config
                           )

    ## send to tensorboard
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)


    print("Check projection on tensorboard:")

    print('tensorboard --logdir=%s' % LOG_DIR)

    # or run this to avoid copy-pasting into terminal
    """
    os.system('killall tensorboard')
    os.system('fuser 6006/tcp -k')  # free default tensorboard port
    os.system('tensorboard --logdir=%s' % LOG_DIR)
    """

if __name__ == '__main__':
    main()



