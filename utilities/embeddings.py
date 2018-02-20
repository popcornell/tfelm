import numpy as np

from tensorflow import Variable
from tensorflow.contrib.tensorboard.plugins import projector
from matplotlib.pyplot import imsave
import tensorflow as tf


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def create_image_embedding(name, input_data, embedding_dir, metadata_path, sprites_path, image_shape, config):

    embedding_var = Variable(input_data, name=name)

    # projector config should be called in top module or parent function
    # otherwise it will be overwritten and only the embedding relative with the last call
    # will be displayed

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata and sprite images
    embedding.metadata_path =  metadata_path
    embedding.sprite.image_path =  sprites_path

    embedding.sprite.single_image_dim.extend(image_shape)
    writer = tf.summary.FileWriter(embedding_dir)

    projector.visualize_embeddings(writer, config)


def create_image_embedding_metadata(input_data, labels, embedding_metadata_path, sprites_path ):

    with open(embedding_metadata_path, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, np.argmax(labels[index])))

    to_visualise = input_data
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    imsave(sprites_path, sprite_image, cmap='gray')


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits




