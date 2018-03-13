import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Logger():
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"))
        self.validate_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "validate"))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))


    def summarize(self, iter, summarizer="train", summaries_dict=None):
        # summary_writer
        if summarizer== "train":
            summary_writer = self.train_summary_writer
        elif summarizer == "validate":
            summary_writer = self.validate_summary_writer
        else:
            summary_writer = self.test_summary_writer

        # write result
        if summaries_dict is not None:
            summary_list = []

            for tag, value in summaries_dict.items():
                ## try, exception
                if len(value.shape)<=1:
                    self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name = tag)
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

            summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

            for summary in summary_list:
                summary_writer.add_summary(summary, iter)

            summary_writer.flush()

    '''
    This function plots one or two curves on one figure. Two curves mush share the same x-axis.
    fig_name: string, name of the saved figure
    dict1: dict, directory of the first npz dictionary
    tag1: string, tag of the first entry in dict to plot
    tag1_label: string, label of first entry, print in legend
    dict2: dict, directory of the second npz dictionary, default None
    tag2: string, tag of the second entry in the dict to plot, default None
    tag2_label: string, label of second entry, print in legend
    x_scale: int, scaling of x axis default 1
    new_plot: boolean, whether to plot on new figure, default to True
    log_x: boolean, logarithmic scale for x-axis, default to False
    x_label: string
    y_label: string
    title: string
    '''
    def plot_npz(self, fig_name, dict1, tag1, tag1_label, dict2=None, tag2=None, tag2_label=None, x_scale=1, new_plot=True, log_x=False, x_label=None, y_label=None, title=None):
        r1 = np.load(dict1)
        y1 = r1[tag1]
        x1 = np.arange(np.shape(y1)[0]) * x_scale + 1
        if dict2 != None:
            r2 = np.load(dict2)
            y2 = r2[tag2]
            x2 = np.arange(np.shape(y2)[0]) * x_scale + 1

        if new_plot:
            plt.clf()
        
        if log_x:
            plt.semilogx(x1, y1, label=tag1_label)
            if dict2 != None:
                plt.semilogx(x2, y2, label=tag2_label)
        else:
            plt.plot(x1, y1, label=tag1_label)
            if dict2 != None:
                plt.plot(x2, y2, label=tag2_label)
        plt.legend()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        plt.savefig(os.path.join(self.config.plot_dir, time+"_"+fig_name+".png"))