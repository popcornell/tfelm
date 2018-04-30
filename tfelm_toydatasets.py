import numpy as np
import tensorflow as tf
from tfelm.elm import ELM
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.datasets import load_iris, fetch_olivetti_faces, load_breast_cancer
from Misc.syntetic_data import spirals
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import time
import itertools


def prepare_data(X, Y):
    """Scale data and apply one hot encoding to labels"""
    num_class = len(np.unique(Y))
    x_train, x_test, y_train, y_test = train_test_split(X.astype('float32'),
                                                        Y.astype('float32'), shuffle=True,
                                                        test_size=0.2)
    print('Dataset info:')
    print('x_train shape ', x_train.shape)
    print('y_train shape ', y_train.shape)
    print('x_test shape ', x_test.shape)
    print('y_test shape ', y_test.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = to_categorical(y_train, num_classes=num_class)
    y_test = to_categorical(y_test, num_classes=num_class)

    return x_train, y_train, x_test, y_test, num_class

######################################################################################################################
# Get dataset
#olivetti = fetch_olivetti_faces()
#x_train, y_train, x_test, y_test, num_class = prepare_data(olivetti.data, olivetti.target)

#data, target = load_breast_cancer(return_X_y=True)
data, target = spirals(5000, noise=0.1)

x_train, y_train, x_test, y_test, num_class = prepare_data(data, target)

# Hyperparameters
input_size = x_train.shape[1]
output_size = num_class
n_neurons = (100, )
batch_size = 1000
n_epochs = 1
repeate_run = 10
norm = repeate_run*(10**0, )
init = (['hpelm_init', 'hpelm_init'],)
######################################################################################################################
start = time.time()

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

    train_acc.append(model.fit(x_train, y_train, batch_size= batch_size))
    print('Train accuracy', train_acc[run])
    test_acc.append(model.evaluate(x=x_test, y=y_test, batch_size=batch_size))
    print('Test accuracy: ', test_acc[run])

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

'''print('AYYYLMAO')
for i in range(3):
    print(np.array(test_acc[i*10:i*10+10:1]).mean()*100, '+-',np.array(test_acc[i*10:i*10+10:1]).std()*100)'''