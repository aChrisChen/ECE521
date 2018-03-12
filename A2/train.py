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

    # to solve memory leak caused by iterator.get_next(), put get_next() out of the loop
    train_next_element = data.train_iter.get_next()
    valid_next_element = data.valid_iter.get_next()
    test_next_element = data.test_iter.get_next()

    # train for each epoch
    for cur_epoch in range(model.cur_epoch_tensor.eval(sess), config.num_epochs + 1, 1):

        #### Training ####
        # initial:use list to record each step(iteration)
        loss_list = []
        acc_list = []
        for iter in tqdm(range(config.num_iter_per_epoch)):
            cur_iter = model.global_step_tensor.eval(sess) + 1
            if cur_iter > config.num_iter:
                break
            loss, acc = train_step(sess, data, model, train_next_element)
            loss_list.append(loss)
            acc_list.append(acc)
            global_train_loss_list.append(loss)
        if cur_iter > config.num_iter:
            break
        training_loss = np.mean(loss_list)
        training_acc = np.mean(acc_list)
        # Print result + summary
        print("Iter: %i , Training Loss: %f, Training Accuracy: %f" % ( cur_iter, training_loss, training_acc))
        summary_dict = {}
        summary_dict['loss'] = training_loss
        summary_dict['acc'] = training_acc
        logger.summarize(cur_iter, summarizer= "train", summaries_dict= summary_dict)

        ###### Validation ####
        if cur_iter % config.validation_interval == 0:
            valid_loss, valid_acc = evaluate(sess, data, model, 'valid', valid_next_element)
            global_valid_loss_list.append(valid_loss)
            print("Validation Loss: %f, Validation Accuracy: %f" % (valid_loss, valid_acc ))
            summary_dict = {}
            summary_dict['loss'] = valid_loss
            summary_dict['acc'] = valid_acc
            logger.summarize(cur_iter, summarizer="valid", summaries_dict=summary_dict)

    np.savez("loss.npz", train=global_train_loss_list, valid=global_valid_loss_list)


def train_step(sess, data, model, next):
    batch_x, batch_y = data.next_batch('train', next)
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: True}
    _, loss, acc = sess.run([model.train_step, model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    return loss, acc


def evaluate(sess, data, model, split, next):
    batch_x, batch_y = data.next_batch(split, next)
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: False}
    loss, acc = sess.run([model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    return loss, acc