import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Part 1
def dist(X, Z):
    return tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1)

# Part 2
np.random.seed(521)
Data = np.linspace(1.0, 10.0, num=100) [:, np.newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

#                                                                distVector: N1 * N2
#                                                         knnDist, knnIndex: N1 * K
#                                               tf.expand_dims(knnIndex, 2): N1 * K * 1
#                               tf.reshape(tf.range(numTrainData),[1,1,-1]): 1 * 1 * N2
# tf.equal(tf.expand_dims(knnIndex, 2), tf.reshape(tf.range(N2), [1,1,-1])): N1 * K * N2
#                                                                         r: N1 * N2
def knn(distVector, K=1):
    knnDist, knnIndex = tf.nn.top_k(-distVector, k=K)
    numTrainData = tf.shape(distVector)[1]
    r = tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(knnIndex, 2), tf.reshape(tf.range(numTrainData), [1,1,-1]))), 1)
    return r/tf.to_float(K)

#                      r: N1 * N2
#               trainTar: N2 * 1
# tf.matmul(R, trainTar): N1 * 1
def predKnn(r, trainTar):
    return tf.matmul(r, trainTar)

#                                             y: N1 * 1
#                                            y_: N1 * 1
#                                        y - y_: N1 * 1
#                 tf.reduce_sum((y - y_)**2, 1): N1
# tf.reduce_mean(tf.reduce_sum((y - y_)**2, 1)): 1
def mse(y, y_):
    return tf.reduce_mean(tf.reduce_sum((y - y_)**2, 1)) / 2

trainX = tf.placeholder(tf.float32, [None, 1], name="train_x")
trainY = tf.placeholder(tf.float32, [None, 1], name="train_y")
newX = tf.placeholder(tf.float32, [None, 1], name="new_x")
newY = tf.placeholder(tf.float32, [None, 1], name="new_y")
k = tf.placeholder("int32", name="k")
predY = predKnn(knn(dist(newX, trainX), K = k), trainY)
MSE = mse(predY, newY)
sess = tf.InteractiveSession()

X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]

kvec = [1, 3, 5, 50]
mse_valid_list = []
mse_test_list = []

for kc in kvec:
    mse_valid = sess.run(MSE, feed_dict={trainX:trainData, trainY:trainTarget, newX:validData, newY:validTarget, k:kc})
    mse_test = sess.run(MSE, feed_dict={trainX:trainData, trainY:trainTarget, newX:testData, newY:testTarget, k:kc})
    mse_valid_list.append(mse_valid)
    mse_test_list.append(mse_test)
    print("K=%d\t validation MSE: %f, test MSE: %f"%(kc, mse_valid, mse_test))
    yp = sess.run(predY, feed_dict={trainX:trainData, trainY:trainTarget, newX:X, k:kc})

    plt.figure(kc+1)
    plt.plot(trainData, trainTarget, '.')
    plt.plot(X, yp, '-')
    plt.title("k-NN regression on data1D, k=%d"%kc)
    plt.show()

k_best = kvec[np.argmin(mse_valid_list)]
print("Best k using validation set is k=%d"%k_best)