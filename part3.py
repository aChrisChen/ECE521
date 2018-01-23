import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Part 1
def dist(X, Z):
    return tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1)

# Part 3
def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255.0
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Data: N * 1024
# Target: N * 1
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy","target.npy",0)
trainTarget = trainTarget[:, np.newaxis]
validTarget = validTarget[:, np.newaxis]
testTarget = testTarget[:, np.newaxis]

# distVector: N1 * N2
# knnDist, knnIndex: N1 * K
def predKnn(distVector, K, trainTar):
    knnDist, knnIndex = tf.nn.top_k(-distVector, k=K)
    knnIndex = tf.reduce_sum(knnIndex, 1)
    knnTarget = tf.gather(trainTar, knnIndex)
    knnTarget = tf.reduce_sum(knnTarget, 1)
    y, idx, count = tf.unique_with_counts(knnTarget)
    predTargetCount, predIndex = tf.nn.top_k(count, k=1)
    predTarget = tf.gather(y, predIndex)
    return predTarget

def mse(y, y_):
    return tf.reduce_mean(tf.reduce_sum((y - y_)**2, 1)) / 2

# def plotImage(data, target):
#     d = np.reshape(data, [-1, 32, 32])
#     plt.figure(1)
#     plt.imshow(d[0], interpolation='nearest', cmap=plt.gray())
#     plt.grid(False)
#     plt.show()

trainX = tf.placeholder(tf.float32, [None, 1024], name="train_x")
trainY = tf.placeholder(tf.float32, [None, 1], name="train_y")
newX = tf.placeholder(tf.float32, [None, 1024], name="new_x")
newY = tf.placeholder(tf.float32, [None, 1], name="new_y")
k = tf.placeholder("int32", name="k")
predY = predKnn(dist(newX, trainX), k, trainY)
MSE = mse(predY, newY)
sess = tf.InteractiveSession()

kvec = [1, 5, 10, 25, 50, 100, 200]
mse_valid_list = []
mse_test_list = []

for kc in kvec:
    mse_valid = sess.run(MSE, feed_dict={trainX:trainData, trainY:trainTarget, newX:validData, newY:validTarget, k:kc})
    mse_test = sess.run(MSE, feed_dict={trainX:trainData, trainY:trainTarget, newX:testData, newY:testTarget, k:kc})
    mse_valid_list.append(mse_valid)
    mse_test_list.append(mse_test)
    print("K=%d\t validation MSE: %f, test MSE: %f"%(kc, mse_valid, mse_test))

k_best = kvec[np.argmin(mse_valid_list)]
print("Best k using validation set is k=%d"%k_best)