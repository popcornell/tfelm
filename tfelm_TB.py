from tfelm.elm import ELM
import tensorflow as tf
import numpy as np
import keras
import os
import time
import itertools


def load_mnist():
    from keras.datasets import mnist
    print('Loading MNIST dataset')
    train, test = mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28 * 28).astype('float32')
    x_test = x_test.reshape(-1, 28 * 28).astype('float32')
    img_size = 28
    img_channels = 1
    return x_train, x_test, y_train, y_test, img_size, img_channels


def load_cifar():
    from keras.datasets import cifar10
    print("Loading Dataset: CIFAR10")
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float32')
    x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float32')
    img_size = 32
    img_channels = 3
    return x_train, x_test, y_train, y_test, img_size, img_channels


def load_SVHN():
    """RGB 32x32 SVHN dataset, stored on HDD"""
    import h5py
    path = "C:\\Users\\Andrea\\PycharmProjects\\EMBEDDED\\ELM\\data"
    print('Loading SVHN dataset...')
    # Open the file as readonly
    h5f = h5py.File(path + '/SVHN_single.h5', 'r')
    # Load the training, test and validation set
    x_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    x_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    h5f.close()
    # Make train and test set a multiple of 100
    deleted_train = np.random.choice(y_train.shape[0], 388, replace=False)
    x_train = np.delete(x_train, deleted_train, axis=0)
    y_train = np.delete(y_train, deleted_train, axis=0)
    deleted_test = np.random.choice(y_test.shape[0], 32, replace=False)
    x_test = np.delete(x_test, deleted_test, axis=0)
    y_test = np.delete(y_test, deleted_test, axis=0)

    print('Training set', x_train.shape, y_train.shape)
    print('Test set', x_test.shape, y_test.shape)
    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)
    img_size = 32
    img_channels = 3
    return x_train.astype('float32'), x_test.astype('float32'), y_train, y_test, img_size, img_channels


def gen_keras_pipeline(x_train, y_train, x_test, y_test, size, channels, batch_size= 1000):
    """Input pipeling for keras image preprocessing"""

    from keras.preprocessing.image import ImageDataGenerator
    # pre-processing pipeline

    input_shape = [size for channels in range(2)]
    input_shape.insert(0, -1)
    input_shape.append(channels)

    x_train = x_train.reshape(input_shape)
    x_test = x_test.reshape(input_shape)

    datagen = ImageDataGenerator(
    #featurewise_std_normalization=True,
    #featurewise_center=True
    samplewise_center=True,
    samplewise_std_normalization=True
    # rotation_range=15,
    # width_shift_range=0.15,
    # height_shift_range=0.15,
    # shear_range=0.2,
    # channel_shift_range=0.2,
    # fill_mode='nearest'
    # horizontal_flip=False,
    # vertical_flip=False,
    # data_format='channels_last'
    )

    datagen.fit(x_train)

    def gen_train():
        n_it = 0
        batches_per_epochs = len(x_train) // batch_size
        for x, y in datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False):
            x = x.reshape(batch_size, size**2*channels)
            if n_it % 25 == 0:
                print("generator iteration: %d" % n_it)
            yield x, y
            n_it += 1
            if n_it >= batches_per_epochs * n_epochs:
                break

    def gen_test():
        n_it = 0
        batches_per_epochs = len(x_test) // batch_size
        for x, y in datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False):
            x = x.reshape(batch_size, size ** 2 * channels)
            if n_it % 25 == 0:
                print("generator iteration: %d" % n_it)
            yield x, y
            n_it += 1
            if n_it >= batches_per_epochs * n_epochs:
                break

    train_dataset = tf.data.Dataset.from_generator(generator=gen_train,
                                                   output_shapes=((batch_size, size**2*channels,),
                                                                  (batch_size, output_size,)),
                                                   output_types=(tf.float32, tf.float32))

    test_dataset = tf.data.Dataset.from_generator(generator=gen_test,
                                                   output_shapes=((batch_size, size**2*channels,),
                                                                  (batch_size, output_size,)),
                                                   output_types=(tf.float32, tf.float32))

    return train_dataset, test_dataset


def gen_pipeline(x_train, y_train, x_test, y_test, size, channels, batch_size= 1000):
    """Input pipeline, raw generators"""

    def gen_batch_train():
        n_it = 0
        num_samples = len(x_train)
        batches = range(0, num_samples, batch_size)
        for batch in batches:
            x_batch = x_train[batch:batch + batch_size]
            y_batch = y_train[batch:batch + batch_size]
            if n_it % 25 == 0:
                print("generator iteration: %d" % n_it)
            yield x_batch, y_batch
            n_it += 1

    def gen_batch_test():
        n_it = 0
        num_samples = len(x_test)
        batches = range(0, num_samples, batch_size)
        for batch in batches:
            x_batch = x_test[batch:batch + batch_size]
            y_batch = y_test[batch:batch + batch_size]
            if n_it % 25 == 0:
                print("generator iteration: %d" % n_it)
            yield x_batch, y_batch
            n_it += 1

    train_dataset = tf.data.Dataset.from_generator(generator=gen_batch_train,
                                                   output_shapes=((batch_size, size ** 2 * channels,),
                                                                  (batch_size, output_size,)),
                                                   output_types=(tf.float32, tf.float32))

    test_dataset = tf.data.Dataset.from_generator(generator=gen_batch_test,
                                                  output_shapes=((batch_size, size ** 2 * channels,),
                                                                 (batch_size, output_size,)),
                                                  output_types=(tf.float32, tf.float32))

    return train_dataset, test_dataset


def batch_normalization(train_dataset, test_dataset):
    """Batch normalization , mean 0, std 1"""

    def batch_norm(x, y):
        # x, y = x # x is a tuple
        mean, var = tf.nn.moments(x, axes=0)
        broadcast_shape = [1, 1]
        broadcast_shape[1] = x.shape[1]
        mean = tf.reshape(mean, broadcast_shape)
        var = tf.reshape(var, broadcast_shape)
        norm_images = tf.div(tf.subtract(x, mean), tf.sqrt(var))
        return norm_images, y

    train_dataset = train_dataset.map(lambda x, y: batch_norm(x, y))
    test_dataset = test_dataset.map(lambda x, y: batch_norm(x, y))
    return train_dataset, test_dataset


######################################################################################################################
# Get dataset
x_train, x_test, y_train, y_test, img_size, img_channels = load_SVHN()

# Data scaler
#from sklearn.preprocessing import StandardScaler
#prescaler = StandardScaler()
#x_train = prescaler.fit_transform(x_train)
#x_test = prescaler.transform(x_test)

######################################################################################################################
# Hyperparameters
input_size = img_size**2 * img_channels
output_size = 10
n_neurons = (1000, 5000, 8000, 15000,)
batch_size = 500
n_epochs = 1
repeate_run = 5
norm = repeate_run*(10**0,)

#Orthogonal init
#ortho_w = tf.orthogonal_initializer()
#uni_b = tf.variance_scaling_initializer(distribution='uniform')


init = (['hpelm_init', 'hpelm_init'],)

######################################################################################################################
start = time.time()

# use gen_pipeline() or keras_gen_pipeline()
train_dataset , test_dataset = gen_pipeline(x_train, y_train, x_test, y_test, img_size, img_channels, batch_size)
# compute the batch normalization over the dataset
train_dataset, test_dataset = batch_normalization(train_dataset, test_dataset)

# Create iterator from dataset structure
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

# Create init op for the iterator
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

run_time = []
train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(n_neurons, init, norm))
for v in itertools.product(n_neurons, init, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: neurons= ', v[0], 'init=', v[1], 'norm=', v[2], 'act= sigmoid')
    t0 = time.time()
    model = ELM(input_size=input_size, output_size=output_size, l2norm=v[2])
    if v[1][1] is not 'default':
        with tf.variable_scope('hpelm_init', reuse=tf.AUTO_REUSE):
            # hpelm like init
            init_w = tf.random_normal(shape=[input_size, v[0]], stddev=3. * tf.sqrt(tf.div(1., input_size)))
            init_b = tf.random_normal(shape=[v[0]], stddev=1.)
            v[1][0] = tf.Variable(init_b, dtype=tf.float32, name='Hb', trainable=False)
            v[1][1] = tf.Variable(init_w, dtype=tf.float32, name='Hw', trainable=False)
    model.add_layer(v[0], activation=tf.sigmoid, w_init=v[1][1], b_init=v[1][0])
    model.compile()

    model.sess.run(train_init_op)
    model.train(iterator, n_batches=n_epochs * (len(x_train) // batch_size))
    #model.sess.run(train_init_op)
    #train_acc.append(model.evaluate(tf_iterator=iterator))

    model.sess.run(test_init_op)
    test_acc.append(model.evaluate(tf_iterator= iterator, batch_size=batch_size))
    print('Test accuracy: ', test_acc[run])

    #B = model.get_B()
    #Hw, Hb = model.get_Hw_Hb()
    #y_out = model.iter_predict(x_test, y_test)
    #tf.reset_default_graph()
    del model
    run_time.append( time.time() - t0)
    print('Run time: ', run_time[run])
    run += 1

print('\nDone training!')
print('Total time: ', time.time() - start)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][2])
print('Best net test accuracy: ', test_acc[best_net])

# mean accuracy
# be careful when multi-parameter are chosen, need to decimate A[::10] of repeate_run to obtain correct accuracy for every hyperpar
if repeate_run > 1:
    print('Mean acc from %d run: %f' %(run_comb.__len__(), sum(test_acc)/len(test_acc)))

'''AYYLMAO
for i in range(4):
    print(np.array(test_acc[i*10:i*10+10:1]).mean(), '+-',np.array(test_acc[i*10:i*10+10:1]).std()*100 )'''