import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np

def train(sess, data, model, config, logger):
    # initialization
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # to record loss of all iterations in one npz
    global_train_loss_list = []
    global_valid_loss_list = []
    global_test_loss_list = []
    global_train_acc_list = []
    global_valid_acc_list = []
    global_test_acc_list = []

    # calculate epoch info
    num_iter_per_epoch = int(np.ceil(config.train_total_size / config.train_batch_size))
    num_epochs = int(np.ceil(config.num_iter / num_iter_per_epoch))
    # train for each epoch
    for cur_epoch in range(int(model.cur_epoch_tensor.eval(sess)), num_epochs + 1, 1):

        #### Training ####
        # initial:use list to record each step(iteration)
        loss_list = []
        acc_list = []
        for iter in tqdm(range(num_iter_per_epoch)):
            cur_iter = model.global_step_tensor.eval(sess) + 1
            if cur_iter > config.num_iter:
                break
            loss, acc= train_step(sess, data, model, config)
            loss_list.append(loss)
            acc_list.append(acc)

            # Checkpoints save at 25%, 50%, 75%, 100%
            if cur_iter % (config.num_iter * 0.25) == 0:
                model.save(sess)

        training_loss = np.mean(loss_list)
        training_acc = np.mean(acc_list)
        # Print result + summary
        print("Iter: %i , Training Loss: %f, Training Accuracy: %f" % ( cur_iter, training_loss, training_acc))
        summary_dict = {}
        summary_dict['loss'] = training_loss
        summary_dict['acc'] = training_acc
        logger.summarize(cur_iter / num_iter_per_epoch, summarizer="train", summaries_dict=summary_dict)
        # npz training
        global_train_loss_list.append(training_loss)
        global_train_acc_list.append(training_acc)

        #### Validation ####
        # if cur_iter % config.validation_interval == 0:
        if cur_iter % num_iter_per_epoch == 0:
            valid_loss, valid_acc = evaluate(sess, data, model, 'valid', config)
            print("Validation Loss: %f, Validation Accuracy: %f" % (valid_loss, valid_acc))
            summary_dict = {}
            summary_dict['loss'] = valid_loss
            summary_dict['acc'] = valid_acc
            logger.summarize(cur_iter / num_iter_per_epoch, summarizer="valid", summaries_dict=summary_dict)
            # npz valid
            global_valid_loss_list.append(valid_loss)
            global_valid_acc_list.append(valid_acc)

        #### Test ####
        # 1.1.2 ask to evaluate on test set while training
        # if cur_iter % config.validation_interval == 0:
        if cur_iter % num_iter_per_epoch == 0:
            test_loss, test_acc = evaluate(sess, data, model, 'test', config)
            print("Test Loss: %f, Test Accuracy: %f" % (test_loss, test_acc))
            summary_dict = {}
            summary_dict['loss'] = test_loss
            summary_dict['acc'] = test_acc
            logger.summarize(cur_iter / num_iter_per_epoch, summarizer="test", summaries_dict=summary_dict)
            # npz test
            global_test_loss_list.append(test_loss)
            global_test_acc_list.append(test_acc)


        if cur_iter > config.num_iter:
            break

    # #### Test after training ###
    # test_loss, test_acc = evaluate(sess, data, model, 'test', config)
    # global_test_loss_list.append(test_loss)
    # global_test_acc_list.append(test_acc)

    np.savez(os.path.join("npz",("%s.npz")%(config.exp_name)), \
             train_loss=global_train_loss_list, \
             valid_loss=global_valid_loss_list, \
             test_loss=global_test_loss_list, \
             train_acc=global_train_acc_list, \
             valid_acc=global_valid_acc_list, \
             test_acc=global_test_acc_list)


def train_step(sess, data, model, config):

    batch_x, batch_y = data.next_batch('train')
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: True}
    _, loss, acc = sess.run([model.train_step, model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    return loss, acc


def evaluate(sess, data, model, split, config):

    batch_x, batch_y = data.next_batch(split)
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: False}
    loss, acc = sess.run([model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    return loss, acc
