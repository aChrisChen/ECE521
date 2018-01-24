import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Part 1: Euclidean Distance
def dist(X, Z):
    return tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1)

# Part 3: KNN Classification
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

#   Data: N * 1024
# Target: N
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy","target.npy",0)
# Target: N -> N * 1
trainTarget = trainTarget[:, np.newaxis] 
validTarget = validTarget[:, np.newaxis]
testTarget = testTarget[:, np.newaxis]

#        distVector: N1 * N2 (N1: new data number; N2: train data number)
#         (trainTar: N2 * 1)
# knnDist, knnIndex: N1 * K (each new data -> K nearest train data(s) )
#         knnTarget: N1 * K (translate index to trainTar[index])
def get_knn_target_matrix(distVector, K, trainTar):
    knnDist, knnIndex = tf.nn.top_k(-distVector, k=K)
    knnIndex = tf.expand_dims(knnIndex, 2) # N1 * K * 1
    # Replace knnIndex by trainTar[knnIndex]
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

def mse(y, _y):
    return tf.reduce_mean(tf.reduce_sum((y - _y)**2, 1)) / 2

def correct_rate(y, _y):
    return tf.reduce_sum(tf.to_float(tf.equal(y, _y))) / tf.to_float(tf.shape(y)[0])

trainX = tf.placeholder(tf.float32, [None, 1024], name="train_x")
trainY = tf.placeholder(tf.float32, [None, 1], name="train_y")
newX = tf.placeholder(tf.float32, [None, 1024], name="new_x")
newY = tf.placeholder(tf.float32, [None, 1], name="new_y")
k = tf.placeholder("int32", name="k")

knnMatrix = get_knn_target_matrix(dist(newX, trainX), k, trainY)

rowIndex = tf.placeholder("int32", name="row_index")
predRowY = predict_1d_knn_target(knnMatrix, rowIndex)

predY = tf.placeholder(tf.float32, [None, 1], name="pred_y")
MSE = mse(predY, newY)
CORR = correct_rate(predY, newY)

sess = tf.InteractiveSession()

kvec = [1, 5, 10, 25, 50, 100, 200]
mseValidList = []
mseTestList = []

for kc in kvec:
    # brand new 1-D prediction vector (N1-length) for each K value
    predValidTarget = []
    predTestTarget = []

    # get (N1 * K) KNN_matrix
    knn_target_valid = sess.run(knnMatrix, feed_dict={newX:validData, trainX:trainData, trainY:trainTarget, k:kc})
    knn_target_test = sess.run(knnMatrix, feed_dict={newX:testData, trainX:trainData, trainY:trainTarget, k:kc})

    # for each row (K-length) in the KNN_matrix, get its majority vote from K values
    for idx in range(np.shape(knn_target_valid)[0]):
        pred_row = sess.run(predRowY, feed_dict={newX:validData, trainX:trainData, trainY:trainTarget, k:kc, rowIndex:[idx]})
        predValidTarget.append(pred_row)
    for idx in range(np.shape(knn_target_test)[0]):
        pred_row = sess.run(predRowY, feed_dict={newX:testData, trainX:trainData, trainY:trainTarget, k:kc, rowIndex:[idx]})
        predTestTarget.append(pred_row)

    # compute MSE
    mse_valid = sess.run(MSE, feed_dict={predY:predValidTarget, newY:validTarget})
    mse_test = sess.run(MSE, feed_dict={predY:predTestTarget, newY:testTarget})
    mseValidList.append(mse_valid)
    mseTestList.append(mse_test)
    print("\nK=%d\t validation MSE: %f, test MSE: %f"%(kc, mse_valid, mse_test))

    # compute correction rate
    corr_valid = sess.run(CORR, feed_dict={predY:predValidTarget, newY:validTarget})
    corr_test = sess.run(CORR, feed_dict={predY:predTestTarget, newY:testTarget})
    print("\t validation corr: %f, test corr: %f"%(corr_valid, corr_test))

# determine the best K value based on validation set
k_best = kvec[np.argmin(mseValidList)]
print("Best k using validation set is k=%d"%k_best)