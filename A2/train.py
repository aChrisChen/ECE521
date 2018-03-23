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
    for cur_epoch in range(model.cur_epoch_tensor.eval(sess), num_epochs + 1, 1):

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

        global_train_loss_list.append(loss)
        global_train_acc_list.append(acc)
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



        #### Validation ####
        if cur_iter % config.validation_interval == 0:
            valid_loss, valid_acc = evaluate(sess, data, model, 'valid', config)
            global_valid_loss_list.append(valid_loss)
            global_valid_acc_list.append(valid_acc)
            print("Validation Loss: %f, Validation Accuracy: %f" % (valid_loss, valid_acc))
            summary_dict = {}
            summary_dict['loss'] = valid_loss
            summary_dict['acc'] = valid_acc
            logger.summarize(cur_iter, summarizer="valid", summaries_dict=summary_dict)





    # #### Test after training ###
    # test_loss, test_acc = evaluate(sess, data, model, 'test', config)
    # global_test_loss_list.append(test_loss)
    # global_test_acc_list.append(test_acc)

    # np.savez(os.path.join("npz",config.dataset,("%d_%f_%f_%s_%s_%s.npz")%(config.num_iter, config.learning_rate, config.weight_decay, config.logistic, config.adam, config.exp_name)), \
    #          train_loss=global_train_loss_list, \
    #          valid_loss=global_valid_loss_list, \
    #          test_loss=global_test_loss_list, \
    #          train_acc=global_train_acc_list, \
    #          valid_acc=global_valid_acc_list, \
    #          test_acc=global_test_acc_list)


def train_step(sess, data, model, config):
    # turn on dropout
    model.mode = True

    batch_x, batch_y = data.next_batch('train')
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: True}
    if config.logistic:
        _, loss, acc = sess.run([model.train_step, model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    else:
        _, loss, target, output = sess.run([model.train_step, model.total_loss, model.y, model.output],
                            feed_dict=feed_dict)
        acc = np.mean((output > 0.5) == target)
    return loss, acc


def evaluate(sess, data, model, split, config):
    # turn off dropout
    model.mode = False

    batch_x, batch_y = data.next_batch(split)
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: False}
    if config.logistic:
        loss, acc = sess.run([model.total_loss, model.accuracy],
                            feed_dict=feed_dict)
    else:
        loss, target, output = sess.run([model.total_loss, model.y, model.output],
                            feed_dict=feed_dict)
        acc = np.mean((output > 0.5) == target)
    return loss, acc
