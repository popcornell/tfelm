import os
from tensorflow.contrib.learn import datasets
from embeddings import create_image_embedding, create_image_embedding_metadata

LOG_DIR = os.path.join(os.getcwd(), "MNIST_LOG")

mnist = datasets.mnist.read_data_sets("MNIST_data/", one_hot=True)

embeddings_path = os.path.join(os.getcwd())

mnist_sprites_path = os.path.join(embeddings_path, 'mnistdigits.png')
mnist_metadata_path = os.path.join(embeddings_path, 'mnist_metadata.tsv')



def main():  # testing purpose only

    import tensorflow as tf


    # visualize mnist data in tensorboard projector
    x, y = mnist.train.next_batch(500)
    # 1000 istances

    # create image metadata
    create_image_embedding_metadata(input_data=x,
                                    labels=y,
                                    embedding_metadata_path=mnist_metadata_path,
                                    sprites_path=mnist_sprites_path
                                    )

    create_image_embedding(name='MNIST',
                                          input_data=x,
                                          image_shape= [28, 28],
                                          embedding_dir=LOG_DIR,
                                          metadata_path= mnist_metadata_path,
                                          sprites_path=mnist_sprites_path
                                          )

    # send to tensorboard
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
    os.system('tensorboard --logdir=%s' % LOGDIR)
    """

if __name__ == '__main__':
    main()



