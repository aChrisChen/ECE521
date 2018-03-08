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
            self.trainDataset, self.validDataset, self.testDataset  = self.notMNISTDataset()
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


    def dataloader(self):
        features, labels = self.trainDataset[0], self.trainDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.batch_size)
        self.train_iter = dataset.make_one_shot_iterator()

        features, labels = self.validDataset[0], self.validDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.batch_size)
        self.valid_iter = dataset.make_one_shot_iterator()

        features, labels = self.trainDataset[0], self.trainDataset[1]
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(self.config.batch_size)
        self.test_iter = dataset.make_one_shot_iterator()

    def next_batch(self, split):
        if split == 'train':
            return self.sess.run(self.train_iter.get_next())
        if split == 'valid':
            return self.sess.run(self.valid_iter.get_next())
        else:
            return self.sess.run(self.test_iter.get_next())



    def oneHotEncoder(self, indice):
        onehot_encoder = OneHotEncoder(sparse=False)
        indice_encoded = indice.reshape(len(indice),1)
        onehot_encoded = onehot_encoder.fit_transform(indice_encoded)
        return onehot_encoded




