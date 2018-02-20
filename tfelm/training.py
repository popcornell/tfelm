import numpy as np
import tensorflow as tf

from utilities.dataset import Dataset
from tensorflow.contrib.tensorboard.plugins import projector



# GLOBALS
n_steps = 0
checks_since_last_progress = 0


def k_fold_training(k, training_set_instances, training_set_labels, test_set_instances, test_set_labels,
                    input_size , output_size,
                    hpar_comb , log_dir, name,
                    emb_metadata_path,
                    sprites_meta_path,
                    check_interval=100
                    ):
    global n_steps
    global checks_since_last_progress
    batch_size = hpar_comb['bsize']
    n_epochs = int(hpar_comb['epo'])

    # these lists will be used to append results for each fold and compute the mean accuracy
    k_val_acc = []
    k_val_loss = []
    k_tr_loss = []
    k_tr_acc = []
    k_test_loss = []
    k_test_acc = []

    inst_per_fold = training_set_instances.shape[0] // k
    print("Beginning training\nK_FOLDS: %d" % k)
    print("Instances per fold: %d" % (inst_per_fold))
    print("#" * 10)

    # split dataset

    for i in range(k):

        if i == k - 1:
            val_fold = training_set_instances[(inst_per_fold * i):]
            train_fold = np.delete(training_set_instances, np.s_[inst_per_fold * i:], 0)
            val_lab_fold = training_set_labels[(inst_per_fold * i):]
            train_lab_fold = np.delete(training_set_labels, np.s_[inst_per_fold * i:], 0)
        else:
            val_fold = training_set_instances[inst_per_fold * i:inst_per_fold * (i + 1)]
            train_fold = np.delete(training_set_instances, np.s_[inst_per_fold * i:inst_per_fold * (i + 1)], 0)
            val_lab_fold = training_set_labels[inst_per_fold * i:inst_per_fold * (i + 1)]
            train_lab_fold = np.delete(training_set_labels, np.s_[inst_per_fold * i:inst_per_fold * (i + 1)], 0)

        train_fold = Dataset(np.concatenate((train_fold, train_lab_fold), 1))
        # labels are appended into second dim 784 --> 794 probably not an optimal solution

        tf.reset_default_graph()  # reset graph from previous fold
        sess = tf.Session()

        x = tf.placeholder(tf.float32, shape=[None, input_size], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, output_size], name="labels")

        # define model with hyperparameters
        # model type itself can be set in hyperparameters dictionary
        [train_step, accuracy, loss_val] = hpar_comb['model'](x, y_, hparams=hpar_comb)

        # weights, biases and activation summaries for the model
        var_summ = tf.summary.merge_all()

        # validation set summaries
        val_summ = [tf.summary.scalar("val_cross_entropy", loss_val),
                    tf.summary.scalar("val_accuracy", accuracy)]

        summ_val = tf.summary.merge(val_summ)

        # training set summaries

        train_summ = [tf.summary.scalar("train_cross_entropy", loss_val),
                      tf.summary.scalar("train_accuracy", accuracy)]

        summ_train = tf.summary.merge(train_summ)

        # test set summaries
        test_summ = [tf.summary.scalar("test_cross_entropy", loss_val),
                     tf.summary.scalar("test_accuracy", accuracy)]

        summ_test = tf.summary.merge(test_summ)

        # embeddings
        path_str = log_dir + '/' + name + '_Kfold_n%d' % (i)

        # embeddings for training visualization

        embedding_input = tf.reshape(x, [-1, input_size])
        embedding = tf.Variable(tf.zeros([hpar_comb['hidd'], input_size]), name=name+ '_Fold%d' % i)
        assignment = embedding.assign(embedding_input)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(path_str)
        writer.add_graph(sess.graph)

        # print to txt file current hyperparameters for convenience
        with open( path_str + "/hyperparam.txt", "w") as text_file:

            print(hpar_comb, file=text_file)

        # setup config for  tensorboard projector
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.sprite.image_path = sprites_meta_path
        embedding_config.metadata_path = emb_metadata_path

        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


        n_steps = 0
        checks_since_last_progress = 0
        best_val_loss = np.infty

        # check interval only for printing

        if check_interval >= (train_fold._num_examples // batch_size):
            print("the check interval should be less than %d" % train_fold._num_examples // batch_size)

        for epoch in range(n_epochs):

            for iteration in range(train_fold._num_examples // batch_size):

                batch = train_fold.next_batch(batch_size)

                sess.run(train_step, feed_dict={x: batch[:, :input_size], y_: batch[:, input_size:]})

                if iteration % check_interval == 0:

                    [val_accuracy, val_loss_val, s_val] = sess.run([accuracy, loss_val, summ_val],
                                                                   feed_dict={x: val_fold,
                                                                              y_: val_lab_fold})

                    writer.add_summary(s_val, n_steps)

                    # append to list to analyze the global k-fold results

                    print("FOLD: %d" % i)
                    print("Accuracy on Validation set at iteration %d at Epoch %d (total steps: %d) is: %.4f" %
                          (iteration, epoch, n_steps, val_accuracy * 100))

                    print("Cost function on Validation set at iteration %d at Epoch %d (total steps: %d) is: %.4f"
                          % (iteration, epoch, n_steps, val_loss_val * 100))
                    print("#" * 10)
                    print("")

                    [train_accuracy, train_loss_val, s_train, s_var] = sess.run([accuracy, loss_val, summ_train, var_summ],
                                                                                feed_dict={x: batch[:, :input_size],
                                                                                            y_: batch[:, input_size:]})

                    # append summaries
                    writer.add_summary(s_train, n_steps)
                    # weights and biases summaries
                    writer.add_summary(s_var, n_steps)

                    # for embeddings
                    sess.run(assignment, feed_dict={x: test_set_instances[:hpar_comb['hidd']],  # hidden layer size
                                                    y_: test_set_labels[:hpar_comb['hidd']]})

                    print("Train_accuracy at iteration %d at Epoch %d is: %.3f" % (
                        iteration, epoch, train_accuracy * 100))
                    print("Train_loss_val at iteration %d at Epoch %d is: %.3E" % (
                        iteration, epoch, train_loss_val * 100))
                    print("#" * 10)
                    print("")

                    if val_loss_val < best_val_loss:

                        best_val_loss = val_loss_val
                        checks_since_last_progress = 0
                        saver.save(sess, (path_str + "/model.ckpt"), global_step=n_steps)
                        best_model_params_step = n_steps

                        best_train_accuracy = train_accuracy
                        best_train_loss = train_loss_val
                        best_val_accuracy = val_accuracy


                    else:
                        # early stopping
                        checks_since_last_progress += 1
                        print("<" * 10)
                        print("checks since last progress: %d" % checks_since_last_progress)
                        print("<" * 10)
                        print("")

                n_steps += 1 # increase n step

            print("Epoch {}, best_train accuracy: {:.4f}%, best_valid. accuracy: {:.4f}%, valid. best loss: {:.6f}"
                                 .format(epoch, best_train_accuracy * 100, best_val_accuracy * 100, best_val_loss))

            if checks_since_last_progress >=  hpar_comb['estop']:

                print("Early stopping!")

                if best_model_params_step:
                    print('Best model parameters at step: %d' % best_model_params_step)

                    saver.restore(sess,  path_str+"/model.ckpt-%d" % (best_model_params_step))
                    print("Restored to best model!")

                    break

        [acc_test, test_loss_val, s_test] = sess.run([accuracy, loss_val, summ_test], feed_dict={x: test_set_instances,
                                                                                                 y_: test_set_labels})

        writer.add_summary(s_test, n_steps)

        k_tr_acc.append(best_train_accuracy)
        k_tr_loss.append(best_train_loss)
        k_val_acc.append(best_val_accuracy)
        k_val_loss.append(best_val_loss)
        k_test_acc.append(acc_test)
        k_test_loss.append(test_loss_val)

        print("#" * 50)

        rep = ('ENDED TRAINING ON K-FOLD number: %d\n'
               'Final Accuracy on test set: %.4f\nLoss on test_set: %.4f\n\n'
               'Best Accuracy on train set: %.4f\n'
               'Best Loss on train set: %.4f\n\n'
               'Best Accuracy on val set: %.4f\n'
               'Best Loss on val set: %.4f\n\n'
               ) % (i, acc_test * 100, test_loss_val, best_train_accuracy * 100, best_train_loss,
                    best_val_accuracy * 100, best_val_loss)

        print("#" * 50)

        print(rep)


        # K-fold text summary
        # tf summary accepts only a tensor in input

        # final report for tensorboard

        # hyperparameters text summaries

        hyperpar_tensor = tf.constant(str(list(hpar_comb.items())))  # TODO better
        hyperpar_properties = tf.summary.text("Hyperpar_properties", hyperpar_tensor)
        summary = sess.run(hyperpar_properties)
        writer.add_summary(summary)

        # can join these probably

        final_rep_tensor = tf.constant(rep)
        final_rep_tensor = tf.summary.text("K_fold_final_report", final_rep_tensor)
        summary = sess.run(final_rep_tensor)
        writer.add_summary(summary)

        n_steps = 0


    # K-FOLD CROSSVAL FINAL REPORT

    print("<" * 100)
    print(">" * 100)
    final_rep = ("K-FOLD CROSS-VALIDATION COMPLETED\nNumber of k-folds: %d\n\n"
                 "TRAIN SET\n"
                 "K-FOLD accuracy on train test:%s\n"
                 "K-FOLD loss on train test: %s\n\n"

                 "VAL SET\n"
                 "K-FOLD accuracy on validation test:%s\n"
                 "K-FOLD loss on validation test: %s\n\n"

                 "TEST SET\n"
                 "K-FOLD accuracy on test test:%s\n"
                 "K-FOLD loss on test test: %s\n\n"

                 "MEAN VALUES\n"
                 "Train Accuracy: %.4f\n"
                 "Validation Accuracy: %.4f\n"
                 "Test Accuracy: %.4f\n"

                 ) % (k, k_tr_acc, k_tr_loss, k_val_acc,
                      k_val_loss, k_test_acc,
                      k_test_loss, np.mean(k_tr_acc) * 100, np.mean(k_val_acc) * 100, np.mean(k_test_acc) * 100)

    print(final_rep)

    # print to tensorboard text

    tf.reset_default_graph()  # reset graph
    sess = tf.Session()

    path_str = log_dir + '/' + name +'_SUMM'
    writer = tf.summary.FileWriter(path_str)

    hyperpar_tensor = tf.constant(str(list(hpar_comb.items())))  # TODO better
    hyperpar_properties = tf.summary.text("Hyperpar_properties", hyperpar_tensor)
    summary = sess.run(hyperpar_properties)
    writer.add_summary(summary)

    final_rep_tensor = tf.constant(final_rep)
    final_rep_tensor = tf.summary.text("K_fold_final_report", final_rep_tensor)
    summary = sess.run(final_rep_tensor)

    writer.add_summary(summary)


def validation_training(training_set,
                        val_set_instances, val_set_labels,
                        test_set_instances, test_set_labels,
                        input_size , output_size,
                        hpar_comb , log_dir, name,
                        emb_metadata_path,
                        sprites_meta_path,
                        check_interval=100
                        ):

    # largely the same as k-fold function
    global n_steps
    global checks_since_last_progress
    batch_size = hpar_comb['bsize']
    n_epochs = int(hpar_comb['epo'])

    print("Beginning training with Validation set")
    print("#" * 10)

    # split dataset

    tf.reset_default_graph()  # reset graph from previous fold
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, input_size], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, output_size], name="labels")

    # define model with hyperparameters
    # model type itself can be set in hyperparameters dictionary
    [train_step, accuracy, loss_val] = hpar_comb['model'](x, y_, hparams=hpar_comb)

    # weights, biases and activation summaries for the model
    var_summ = tf.summary.merge_all()

    # validation set summaries
    val_summ = [tf.summary.scalar("val_cross_entropy", loss_val),
                tf.summary.scalar("val_accuracy", accuracy)]

    summ_val = tf.summary.merge(val_summ)

    # training set summaries

    train_summ = [tf.summary.scalar("train_cross_entropy", loss_val),
                  tf.summary.scalar("train_accuracy", accuracy)]

    summ_train = tf.summary.merge(train_summ)

    # test set summaries
    test_summ = [tf.summary.scalar("test_cross_entropy", loss_val),
                 tf.summary.scalar("test_accuracy", accuracy)]

    summ_test = tf.summary.merge(test_summ)

    # embeddings
    path_str = log_dir + '/' + name

    # embeddings for training visualization

    embedding_input = tf.reshape(x, [-1, input_size])
    embedding = tf.Variable(tf.zeros([hpar_comb['hidd'], input_size]), name=name )
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(path_str)
    writer.add_graph(sess.graph)

    # print to txt file current hyperparameters for convenience
    with open(path_str + "/hyperparam.txt", "w") as text_file:
        print(hpar_comb, file=text_file)

    # setup config for  tensorboard projector
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = sprites_meta_path
    embedding_config.metadata_path = emb_metadata_path

    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    n_steps = 0
    checks_since_last_progress = 0
    best_val_loss = np.infty


    for epoch in range(n_epochs):

        for iteration in range(training_set.num_examples // batch_size):

            batch = training_set.next_batch(batch_size)

            # training

            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

            if iteration % check_interval == 0:
                [val_accuracy, val_loss_val, s_val] = sess.run([accuracy, loss_val, summ_val],
                                                               feed_dict={x: val_set_instances, y_: val_set_labels})

                writer.add_summary(s_val, n_steps)

                # append to list to analyze the global k-fold results

                print("step: %d" % n_steps)
                print("Accuracy on Validation set at iteration %d at Epoch %d (total steps: %d) is: %.4f" %
                                                                     (iteration, epoch, n_steps, val_accuracy * 100))

                print("Cost function on Validation set at iteration %d at Epoch %d (total steps: %d) is: %.4f" %
                                                                     (iteration, epoch, n_steps, val_loss_val * 100))
                print("#" * 10)
                print("")

                [train_accuracy, train_loss_val, s_train, s_var] = sess.run([accuracy, loss_val, summ_train, var_summ],
                                                                            feed_dict={x: batch[0],y_: batch[1]})

                # append summaries
                writer.add_summary(s_train, n_steps)

                # weights and biases summary
                writer.add_summary(s_var, n_steps)

                # for embeddings
                sess.run(assignment, feed_dict={x: test_set_instances[:hpar_comb['hidd']],  # hidden layer size
                                                 y_: test_set_labels[:hpar_comb['hidd']]})

                print("Train_accuracy at iteration %d at Epoch %d is: %.3f" % (iteration, epoch, train_accuracy * 100))
                print("Train_loss_val at iteration %d at Epoch %d is: %.3E" % (iteration, epoch, train_loss_val * 100))
                print("#" * 10)
                print("")

                if val_loss_val < best_val_loss:

                    best_val_loss = val_loss_val
                    checks_since_last_progress = 0
                    saver.save(sess, (path_str + "/model.ckpt"), global_step=n_steps)
                    best_model_params_step = n_steps

                    best_train_accuracy = train_accuracy
                    best_train_loss = train_loss_val
                    best_val_accuracy = val_accuracy

                else:
                    # for early stopping
                    checks_since_last_progress += 1
                    print("<" * 10)
                    print("checks since last progress: %d" % checks_since_last_progress)
                    print("<" * 10)
                    print("")


            n_steps += 1 # increase number of step

        print("Epoch {}, best_train accuracy: {:.4f}%, best_valid. accuracy: {:.4f}%, valid. best loss: {:.6f}"
                                  .format(epoch, best_train_accuracy * 100, best_val_accuracy * 100, best_val_loss))

        if checks_since_last_progress >= hpar_comb['estop']:

            print("Early stopping!")

            if best_model_params_step:
                print('Best model parameters at step: %d' % best_model_params_step)

                saver.restore(sess, path_str + "/model.ckpt-%d" % (best_model_params_step))
                print("Restored to best model!")

                break

    [acc_test, test_loss_val, s_test] = sess.run([accuracy, loss_val, summ_test], feed_dict={x: test_set_instances,
                                                                                                  y_: test_set_labels})

    writer.add_summary(s_test, n_steps)

    print("#" * 50)

    rep = ('ENDED TRAINING and TESTING\n'
           'Final Accuracy on test set: %.4f\nLoss on test_set: %.4f\n\n'
           'Best Accuracy on train set: %.4f\n'
           'Best Loss on train set: %.4f\n\n'
           'Best Accuracy on val set: %.4f\n'
           'Best Loss on val set: %.4f\n\n'
           ) % (acc_test * 100, test_loss_val, best_train_accuracy * 100, best_train_loss,
                best_val_accuracy * 100, best_val_loss)

    print("#" * 50)

    print(rep)

    # FINAL REPORT on tensorboard

    # tf summary accepts only a tensor in input

    # final report for tensorboard

    # hyperparameters text summaries

    hyperpar_tensor = tf.constant(str(list(hpar_comb.items())))  # TODO better
    hyperpar_properties = tf.summary.text("Hyperpar_properties", hyperpar_tensor)
    summary = sess.run(hyperpar_properties)
    writer.add_summary(summary)

    # can join these probably

    final_rep_tensor = tf.constant(rep)
    final_rep_tensor = tf.summary.text("FINAL REPORT", final_rep_tensor)
    summary = sess.run(final_rep_tensor)
    writer.add_summary(summary)

    n_steps = 0





