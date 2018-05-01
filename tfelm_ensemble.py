from tfelm.elm import ELM
import tensorflow as tf
import numpy as np
import keras
import os
import time
import itertools
import random


def correct_prediction(labels, prediction):
    """Reshape vector from one-hot-encode format and create a boolean array where each image is correctly classified"""

    cls_true = labels.argmax(1)
    cls_pred = prediction.argmax(1)
    correct = (cls_true == cls_pred)
    return correct


def ensemble_prediction():
    pred_labels = []
    test_accuracies = []
    train_accuracies = []

    for i in range(n_estimator):
        test_acc = correct_prediction(y_test, y_test_predicted[i])
        test_acc = test_acc.mean()
        test_accuracies.append(test_acc)

        train_acc = correct_prediction(y_train, y_train_predicted[i])
        train_acc = train_acc.mean()
        train_accuracies.append(train_acc)

        msg = "Network: {0}, Accuracy on Training-Set: {1:.6f}, Test-Set: {2:.6f}"
        print(msg.format(i, train_acc, test_acc))

        pred_labels = np.array(y_test_predicted)

    return pred_labels, test_accuracies, train_accuracies


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


######################################################################################################################
# Get dataset
x_train, x_test, y_train, y_test, img_size, img_channels = load_cifar()

# Data scaler
from sklearn.preprocessing import StandardScaler
prescaler = StandardScaler()
x_train = prescaler.fit_transform(x_train)
x_test = prescaler.transform(x_test)

######################################################################################################################
# Hyperparameters
input_size = img_size**2 * img_channels
output_size = 10
n_neurons = 5000
batch_size = 1000
n_epochs = 1
n_estimator = 40
norm = 10**3
init = ['default', 'default']
act = (tf.sigmoid, tf.tanh)

######################################################################################################################
# use gen_pipeline() or keras_gen_pipeline()
train_dataset , test_dataset = gen_pipeline(x_train, y_train, x_test, y_test, img_size, img_channels, batch_size)
# compute the batch normalization over the dataset
#train_dataset, test_dataset = batch_normalization(train_dataset, test_dataset)

# Create iterator from dataset structure
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

# Create init op for the iterator
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)


# Training networks
print()
print("Bagging %d ELM classificator\n" % n_estimator)
y_train_predicted = []
y_test_predicted = []
tstart = time.time()
for i in range(n_estimator):
    print('Estim #%d/%d' % (i+1, n_estimator))
    t0 = time.time()
    model = ELM(input_size=input_size, output_size=output_size, l2norm=norm)
    model.add_layer(n_neurons, activation=random.choice(act), w_init=init[0], b_init=init[1])
    model.compile()

    model.sess.run(train_init_op)
    model.train(iterator, n_batches=n_epochs * (len(x_train) // batch_size))
    model.sess.run(train_init_op)
    y_train_predicted.append(model.predict(tf_iterator=iterator, batch_size=batch_size))

    model.sess.run(test_init_op)
    y_test_predicted.append(model.predict(tf_iterator= iterator, batch_size=batch_size))

    del model
    print('Run time: ', time.time() - t0)
    print()

print("Training done in ", (time.time() - tstart), "seconds!!")
print("###############################################################################################")

# pred labels on test-set
pred_labels, test_accuracies, train_accuracies = ensemble_prediction()

print("\nMean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

# BUILDING ENSABLE
# TODO: selector and stacking
ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)  # one-hot-reverted
ensemble_correct = (ensemble_cls_pred == y_test.argmax(1))
ensemble_incorrect = np.logical_not(ensemble_correct)

# best network
best_net = np.argmax(test_accuracies)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == y_test.argmax(1))
best_net_incorrect = np.logical_not(best_net_correct)

# Ensemble and Best network comparison
print("\nBest Net correct estimated instances: ", np.sum(best_net_correct))
print("Ensamble correct estimated instances: ", np.sum(ensemble_correct))

ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)
print("Best Net better classification: ", best_net_better.sum())
print("Ensemble better classification: ", ensemble_better.sum())

ensemble_acc = correct_prediction(y_test, ensemble_pred_labels)
ensemble_acc = ensemble_acc.mean()
best_net_acc = test_accuracies[best_net]
print("\nEnsemble accuracy: ", ensemble_acc * 100)
print("Best net accuracy: ", test_accuracies[best_net] * 100)

print("\nCOMPARISON HARD-VOTING VS MEAN BAGGING")
hard_voting_cls_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=pred_labels.argmax(2))
hard_voting_correct = (hard_voting_cls_pred == y_test.argmax(1))
hard_voting_incorrect = np.logical_not(hard_voting_correct)
print("Hard voting correct estimated istances: ", np.sum(hard_voting_correct))
print("Mean better classification: ", ensemble_better.sum())
hard_voting_better = np.logical_and(ensemble_incorrect, hard_voting_correct)
mean_better = np.logical_and(ensemble_correct, hard_voting_incorrect)
print("Hard Voting better classification: ", hard_voting_better.sum())
print("Mean Bagging better classification: ", mean_better.sum())
hard_voting_acc = (hard_voting_cls_pred == y_test.argmax(1)).mean()
print("Hard voting accuracy: ", hard_voting_acc * 100)
print("Mean accuracy: ", ensemble_acc * 100)
print("###############################################################################################\n")

######################################################################################################################
######################################################################################################################
# 1-layer STACKING
# Build aggregator on top of predicted value
print('Building ELM aggregator on top of estim values')
train_agg = np.swapaxes(np.array(y_train_predicted), 0, 1).reshape(-1, n_estimator*output_size)
test_agg = np.swapaxes(np.array(pred_labels), 0, 1).reshape(-1, n_estimator*output_size)

agg_scaler = StandardScaler()
train_agg_scaled = agg_scaler.fit_transform(train_agg)
test_agg_scaled = agg_scaler.transform(test_agg)


print('Aggregator hypar: neurons= 2000')
model = ELM(input_size=n_estimator*output_size, output_size=output_size, l2norm=10**0)
model.add_layer(2000, activation=tf.sigmoid, w_init='default', b_init='default')
model.compile()
model.fit(train_agg_scaled, y_train, batch_size= 1000)
agg_accuracy = model.evaluate(x=test_agg_scaled, y=y_test, batch_size=1000)
del model

print("###############################################################################################")
print('RECAP:')
print("Single net mean-accuracy: {0:.3f} % ".format(np.mean(test_accuracies)))
print("Ensemble mean accuracy: {0:.3f} % " .format(ensemble_acc * 100))
print("Hard voting ensemble accuracy: {0:.3f} % " .format(hard_voting_acc * 100))
print('Aggregator accuracy: {0:.3f} % ' .format(agg_accuracy*100))



