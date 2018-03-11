import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np

def train(sess, data, model, config, logger):
    # initialization
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # train for each epoch
    for cur_epoch in range(model.cur_epoch_tensor.eval(sess), config.num_epochs + 1, 1):

        #### Training ####
        # initial:use list to record each step(iteration)
        loss_list = []
        acc_list = []
        for iter in tqdm(range(config.num_iter_per_epoch)):
            loss, acc = train_step(sess, data, model)
            loss_list.append(loss)
            acc_list.append(acc)
        training_loss = np.mean(loss_list)
        training_acc = np.mean(acc_list)
        # Print result + summary
        cur_iter = model.global_step_tensor.eval(sess)
        print("Iter: %i , Training Loss: %f, Training Accuracy: %f" % ( cur_iter, training_loss, training_acc))
        summary_dict = {}
        summary_dict['loss'] = training_loss
        summary_dict['acc'] = training_acc
        logger.summarize(cur_iter, summarizer= "train", summaries_dict= summary_dict)


        ###### Validation  ####
        if cur_iter % config.validation_interval == 0:
            valid_loss, valid_acc = evaluate(sess, data, model, 'valid')
            print("Validation Loss: %f, Validation Accuracy: %f" % (valid_loss, valid_acc ))
            summary_dict = {}
            summary_dict['loss'] = valid_loss
            summary_dict['acc'] = valid_acc
            logger.summarize(cur_iter, summarizer="valid", summaries_dict=summary_dict)


def train_step(sess, data, model):
    batch_x, batch_y = data.next_batch('train')
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: True}
    _, loss, acc = sess.run([model.train_step, model.total_loss, model.accuracy],
                            feed_dict= feed_dict)
    return loss, acc


def evaluate(sess, data, model, split):
    batch_x, batch_y = data.next_batch(split)
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: False}
    loss, acc = sess.run([model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    return loss, acc