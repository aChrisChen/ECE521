import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from part1 import dist
from part2 import mse
from collections import defaultdict


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

    trainTarget = trainTarget[:, np.newaxis]
    validTarget = validTarget[:, np.newaxis]
    testTarget = testTarget[:, np.newaxis]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def get_knn_target_matrix(distVector, K, trainTar):
    '''

    :param distVector: N1 * N2 (N1: new data; N2: train data)
    :param K:
    :param trainTar: N2 * 1
    :return: knnTarget: N1 * K (translate index to trainTar[index])
    '''
    knnDist, knnIndex = tf.nn.top_k(-distVector, k=K)
    knnIndex = tf.expand_dims(knnIndex, 2) # N1 * K * 1
    knnTarget = tf.gather_nd(trainTar, knnIndex) # N1 * K * 1
    knnTarget = tf.reduce_sum(knnTarget, 2) # N1 * K
    return knnTarget

def predict_1d_knn_target(knnTarget, rowIndex):
    # Apply majority vote to 'rowIndex'th row (K-length 1D vector)
    row = tf.gather_nd(knnTarget, rowIndex)
    y, idx, count = tf.unique_with_counts(row)
    majorCount, majorIndex = tf.nn.top_k(count, k=1)
    predRowTarget = tf.gather(y, majorIndex)
    return predRowTarget


def buildGraph():
    # Variable Creation
    trainX = tf.placeholder(tf.float32, [None, 1024], name="train_x")
    trainY = tf.placeholder(tf.float32, [None, 1], name="train_y")
    newX = tf.placeholder(tf.float32, [None, 1024], name="new_x")
    newY = tf.placeholder(tf.float32, [None, 1], name="new_y")
    predY = tf.placeholder(tf.float32, [None, 1], name="pred_y")
    k = tf.placeholder("int32", name="k")
    rowIndex = tf.placeholder("int32", name="row_index")

    # knn classification
    knnMatrix = get_knn_target_matrix(dist(newX, trainX), k, trainY)
    predRowY = predict_1d_knn_target(knnMatrix, rowIndex)
    lossMSE = mse(predY, newY)
    accuracy = tf.metrics.accuracy(newY, predY)

    return trainX, trainY, newX, newY, k, rowIndex, predY, knnMatrix,predRowY,lossMSE, accuracy


#
#
# # determine the best K value based on validation set
# k_best = kvec[np.argmin(mseValidList)]
# print("Best k using validation set is k=%d"%k_best)

if __name__ == '__main__':
    # Load Data
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy",0)
    kvec = [1, 5, 10, 25, 50, 100, 200]

    # Load computation_Graph
    trainX, trainY, newX, newY, k, rowIndex, predY, knnMatrix, predRowY, lossMSE, accuracy = buildGraph()

    # Initialization
    sess = tf.InteractiveSession()
    mseResult = defaultdict(list)


    for kc in kvec:
        # brand new 1-D prediction vector (N1-length) for each K value
        predictionTargetResult = defaultdict(list)

        # get (N1 * K) KNN_matrix
        knn_target_valid = sess.run(knnMatrix,
                                    feed_dict={newX: validData, trainX: trainData, trainY: trainTarget, k: kc})
        knn_target_test = sess.run(knnMatrix,
                                   feed_dict={newX: testData, trainX: trainData, trainY: trainTarget, k: kc})

        # for each row (K-length) in the KNN_matrix, get its majority vote from K values
        for idx in range(np.shape(knn_target_valid)[0]):
            pred_row = sess.run(predRowY, feed_dict={newX: validData, trainX: trainData, trainY: trainTarget, k: kc,
                                                     rowIndex: [idx]})
            predictionTargetResult['validation'].append(pred_row)

        for idx in range(np.shape(knn_target_test)[0]):
            pred_row = sess.run(predRowY, feed_dict={newX: testData, trainX: trainData, trainY: trainTarget, k: kc,
                                                     rowIndex: [idx]})
            predictionTargetResult['test'].append(pred_row)

        # compute MSE
        mse_valid = sess.run(lossMSE, feed_dict={predY: predictionTargetResult['validation'], newY: validTarget})
        mse_test = sess.run(lossMSE, feed_dict={predY: predictionTargetResult['test'], newY: testTarget})
        mseResult['validation'].append(mse_valid)
        mseResult['test'].append(mse_test)
        print("\nK=%d\t validation MSE: %f, test MSE: %f" % (kc, mse_valid, mse_test))

        # compute correction rate
        corr_valid = sess.run(accuracy, feed_dict={predY: predictionTargetResult['validation'], newY: validTarget})
        corr_test = sess.run(accuracy, feed_dict={predY: predictionTargetResult['test'], newY: testTarget})
        print("\t validation corr: %f, test corr: %f" % (corr_valid, corr_test))