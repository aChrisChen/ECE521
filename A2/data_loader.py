import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

class DataLoader():
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess

        self.get_dataset()
        self.dataloader()


    def get_dataset(self):
        name_dataset = self.config.dataset
        if name_dataset == 'notMNIST_2':
            self.trainDataset, self.validDataset, self.testDataset = self.notMNISTDataset2()
        elif name_dataset == "notMNIST_10":
            self.trainDataset, self.validDataset, self.testDataset = self.notMNISTDataset10()
        elif name_dataset == 'FaceScrub':
            self.trainDataset, self.validDataset, self.testDataset = self.faceDataset(self.config.face_task)
        elif name_dataset == 'notMNIST_A3':
            self.trainDataset, self.validDataset, self.testDataset = self.notMNIST_A3()

        else:
            print('no dataset')


    def notMNISTDataset2(self):
        with np.load("data/notMNIST.npz") as data:
            Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)

        Data, Target = Data[randIndx], Target[randIndx]

        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        # one hot switch
        if self.config.logistic:
            trainTarget = self.oneHotEncoder(trainTarget)
            validTarget = self.oneHotEncoder(validTarget)
            testTarget = self.oneHotEncoder(testTarget)

        # auto detect data size
        self.config.image_size = np.shape(trainData)[1]
        self.config.train_total_size = np.shape(trainData)[0]
        self.config.valid_batch_size = np.shape(validData)[0]
        self.config.test_batch_size = np.shape(testData)[0]
        self.config.output_size = np.shape(trainTarget)[1]

        return [trainData, trainTarget], [validData, validTarget], [testData, testTarget]


    def notMNISTDataset10(self):
        with np.load("data/notMNIST.npz") as data:
            Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        # one hot switch
        if self.config.logistic:
            trainTarget = self.oneHotEncoder(trainTarget)
            validTarget = self.oneHotEncoder(validTarget)
            testTarget = self.oneHotEncoder(testTarget)

        # auto detect data size
        self.config.image_size = np.shape(trainData)[1]
        self.config.train_total_size = np.shape(trainData)[0]
        self.config.valid_batch_size = np.shape(validData)[0]
        self.config.test_batch_size = np.shape(testData)[0]
        self.config.output_size = np.shape(trainTarget)[1]

        return [trainData, trainTarget], [validData, validTarget], [testData, testTarget]


    def faceDataset(self, task):
        Data = np.load("data/data.npy") / 255.
        Target = np.load("data/target.npy")

        np.random.seed(45689)
        rnd_idx = np.arange(np.shape(Data)[0])
        np.random.shuffle(rnd_idx)

        trBatch = int(0.8*len(rnd_idx))
        validBatch = int(0.1*len(rnd_idx))

        trainData, validData, testData = Data[rnd_idx[1:trBatch],:], \
        Data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
        Data[rnd_idx[trBatch + validBatch+1:-1],:]

        trainTarget, validTarget, testTarget = Target[rnd_idx[1:trBatch], task], \
        Target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
        Target[rnd_idx[trBatch + validBatch + 1:-1], task]

        trainTarget = trainTarget[:, np.newaxis]
        validTarget = validTarget[:, np.newaxis]
        testTarget = testTarget[:, np.newaxis]

        # one hot switch
        if self.config.logistic:
            trainTarget = self.oneHotEncoder(trainTarget)
            validTarget = self.oneHotEncoder(validTarget)
            testTarget = self.oneHotEncoder(testTarget)

        # auto detect data size
        self.config.image_size = np.shape(trainData)[1]
        self.config.train_total_size = np.shape(trainData)[0]
        self.config.valid_batch_size = np.shape(validData)[0]
        self.config.test_batch_size = np.shape(testData)[0]
        self.config.output_size = np.shape(trainTarget)[1]

        return [trainData, trainTarget], [validData, validTarget], [testData, testTarget]


    def notMNIST_A3(self):
        with np.load("data/notMNIST.npz") as data:
            Data, Target = data["images"], data["labels"]
            np.random.seed(521)
            randIndx = np.arange(len(Data))
            np.random.shuffle(randIndx)
            Data = Data[randIndx] / 255.
            Target = Target[randIndx]
            trainData, trainTarget = Data[:15000], Target[:15000]
            validData, validTarget = Data[15000:16000], Target[15000:16000]
            testData, testTarget = Data[16000:], Target[16000:]

            trainTarget = self.oneHotEncoder(trainTarget)
            validTarget = self.oneHotEncoder(validTarget)
            testTarget = self.oneHotEncoder(testTarget)

            # auto detect data size
            self.config.image_size = np.shape(trainData)[1]
            self.config.train_total_size = np.shape(trainData)[0]
            self.config.valid_batch_size = np.shape(validData)[0]
            self.config.test_batch_size = np.shape(testData)[0]
            self.config.output_size = np.shape(trainTarget)[1]

            return [trainData, trainTarget], [validData, validTarget], [testData, testTarget]

    def dataloader(self):
        self.max_value = tf.placeholder(tf.int64, shape=[])

        features, labels = self.trainDataset[0], self.trainDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.train_batch_size)
        self.train_iter = dataset.make_initializable_iterator()
        self.train_next_element = self.train_iter.get_next()
        self.sess.run(self.train_iter.initializer, feed_dict={self.max_value: self.config.train_batch_size})

        features, labels = self.validDataset[0], self.validDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.valid_batch_size)
        self.valid_iter = dataset.make_initializable_iterator()
        self.valid_next_element = self.valid_iter.get_next()
        self.sess.run(self.valid_iter.initializer, feed_dict={self.max_value: self.config.valid_batch_size})

        features, labels = self.testDataset[0], self.testDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.test_batch_size)
        self.test_iter = dataset.make_initializable_iterator()
        self.test_next_element = self.test_iter.get_next()
        self.sess.run(self.test_iter.initializer, feed_dict={self.max_value: self.config.test_batch_size})


    def next_batch(self, split):
        if split == 'train':
            try:
                self.next = self.sess.run(self.train_next_element)
            except tf.errors.OutOfRangeError:
                self.sess.run(self.train_iter.initializer, feed_dict={self.max_value: self.config.train_batch_size})
                self.next = self.sess.run(self.train_next_element)
            return self.next
        if split == 'valid':
            try:
                self.next = self.sess.run(self.valid_next_element)
            except tf.errors.OutOfRangeError:
                self.sess.run(self.valid_iter.initializer, feed_dict={self.max_value: self.config.valid_batch_size})
                self.next = self.sess.run(self.valid_next_element)
            return self.next
        else:
            return self.sess.run(self.test_next_element)


    def oneHotEncoder(self, indice):
        onehot_encoder = OneHotEncoder(sparse=False)
        indice_encoded = indice.reshape(len(indice),1)
        onehot_encoded = onehot_encoder.fit_transform(indice_encoded)
        return onehot_encoded