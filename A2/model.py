import tensorflow as tf
import os
import math

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step_tensor)

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.config.exp_name)
        if latest_checkpoint:
            print("Loading model checkpoint {} ... \n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name = 'cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name = 'global_step')

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError


class LogisticClassification(BaseModel):
    def __init__(self, config):
        super(LogisticClassification, self).__init__(config)

        self.BATCH_SIZE = self.config.batch_size
        self.input_size = self.config.input_size
        self.IMAGE_SIZE = int(math.sqrt(self.input_size))
        self.output_size = self.config.output_size

        self.build_model()
        self.loss()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        # TODO: error when this x placeholder used for evaluation, if the number of data points in validation set is less than self.BATCH_SIZE
        self.x = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE))
        self.y = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.output_size))

        self.input = tf.reshape(self.x, [-1, self.IMAGE_SIZE * self.IMAGE_SIZE])
        # TODO: why cannot set activation=tf.nn.sigmoid?
        self.output = tf.layers.dense(inputs=self.input, units=self.config.output_size, name='dense1')

    def loss(self):
        # loss
        if self.config.output_size == 2:
            self.cross_e = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.output))
        else:
            self.cross_e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        
        self.weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.output.name)[0] + '/kernel:0')
        self.weight_decay_loss = tf.nn.l2_loss(self.weights) * self.config.weight_decay
        
        self.total_loss = self.cross_e + self.weight_decay_loss
        # accuracy
        self.prediction = tf.argmax(self.output,1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.y,1)), tf.float32))

        # update parameters
        if self.config.adam:
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step= self.global_step_tensor)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step= self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class LinearRegression(BaseModel):
    def __init__(self, config):
        super(LinearRegression, self).__init__(config)

        self.BATCH_SIZE = self.config.batch_size
        self.input_size = self.config.input_size
        self.IMAGE_SIZE = int(math.sqrt(self.input_size))
        self.output_size = self.config.output_size

        self.build_model()
        self.loss()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE))
        self.y = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.output_size))

        self.input = tf.reshape(self.x, [-1, self.IMAGE_SIZE * self.IMAGE_SIZE])
        self.output = tf.layers.dense(inputs=self.input, units=self.config.output_size, name='dense1')

    def loss(self):
        # loss
        self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.output),reduction_indices=1)) * 0.5
        
        self.weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.output.name)[0] + '/kernel:0')
        self.weight_decay_loss = tf.nn.l2_loss(self.weights) * self.config.weight_decay
        
        self.total_loss = self.mse + self.weight_decay_loss
        
        # accuracy
        # this prediction value is for classification job
        self.prediction = tf.round(self.output)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))

        # update parameters
        if self.config.adam:
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step = self.global_step_tensor)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step = self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)