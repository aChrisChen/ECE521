import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

class DataGenerator():
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess

        self.get_dataset()
        self.dataloader()


    def get_dataset(self):
        name_dataset = self.config.dataset
        if name_dataset == 'notMNIST':
            self.trainDataset, self.validDataset, self.testDataset = self.notMNISTDataset()
        elif name_dataset == 'FaceScrub':
            self.trainDataset, self.validDataset, self.testDataset = self.faceDataset(self.config.face_task)
        else:
            print('no dataset')


    def notMNISTDataset(self):
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
        trainTarget = self.oneHotEncoder(trainTarget)

        validData, validTarget = Data[3500:3600], Target[3500:3600]
        validTarget = self.oneHotEncoder(validTarget)

        testData, testTarget = Data[3600:], Target[3600:]
        testTarget = self.oneHotEncoder(testTarget)

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

        trainTarget = self.oneHotEncoder(trainTarget)
        validTarget = self.oneHotEncoder(validTarget)
        testTarget = self.oneHotEncoder(testTarget)

        return [trainData, trainTarget], [validData, validTarget], [testData, testTarget]

    def dataloader(self):
        # self.max_value = tf.placeholder(tf.int64, shape=[])

        features, labels = self.trainDataset[0], self.trainDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.batch_size)
        self.train_iter = dataset.make_initializable_iterator()

        features, labels = self.validDataset[0], self.validDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.valid_batch_size)
        self.valid_iter = dataset.make_initializable_iterator()

        features, labels = self.trainDataset[0], self.trainDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.test_batch_size)
        self.test_iter = dataset.make_initializable_iterator()


    def next_batch(self, split, next):
        if split == 'train':
            self.sess.run(self.train_iter.initializer)
            return self.sess.run(next)
        if split == 'valid':
            self.sess.run(self.valid_iter.initializer)
            return self.sess.run(next)
        else:
            self.sess.run(self.test_iter.initializer)
            return self.sess.run(next)


    def oneHotEncoder(self, indice):
        onehot_encoder = OneHotEncoder(sparse=False)
        indice_encoded = indice.reshape(len(indice),1)
        onehot_encoded = onehot_encoder.fit_transform(indice_encoded)
        return onehot_encoded