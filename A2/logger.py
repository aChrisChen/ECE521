import tensorflow as tf
import os

class Logger():
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"))
        self.validate_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "validate"))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))


    def summarize(self, iter, summarizer= "train", summaries_dict= None):
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