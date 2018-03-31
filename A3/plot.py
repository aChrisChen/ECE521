
import numpy as np
import matplotlib.pyplot as plt

train_epoch = 20001
valid_epoch = 20000
EPOCH = 20000

def main():

#****************************************************************************
# 1.1.2
#****************************************************************************
    r1 = np.load("data_output/1.1.2.1.npz")
    train_loss_x1 = np.arange(train_epoch) + 1
    train_loss_y1 = r1["train_loss"]
    valid_loss_x1 = np.arange(valid_epoch) + 1
    valid_loss_y1 = r1["valid_loss"]
    test_loss_x1 = np.arange(EPOCH) + 1
    test_loss_y1 = r1["test_loss"]
    train_acc_x1 = np.arange(train_epoch) + 1
    train_acc_y1 = r1["train_acc"]
    valid_acc_x1 = np.arange(valid_epoch) + 1
    valid_acc_y1 = r1["valid_acc"]
    test_acc_x1 = np.arange(EPOCH) + 1
    test_acc_y1 = r1["test_acc"]

    r2 = np.load("data_output/1.1.2.2.npz")
    train_loss_x2 = np.arange(train_epoch) + 1
    train_loss_y2 = r2["train_loss"]
    valid_loss_x2 = np.arange(valid_epoch) + 1
    valid_loss_y2 = r2["valid_loss"]
    test_loss_x2 = np.arange(EPOCH) + 1
    test_loss_y2 = r2["test_loss"]
    train_acc_x2 = np.arange(train_epoch) + 1
    train_acc_y2 = r2["train_acc"]
    valid_acc_x2 = np.arange(valid_epoch) + 1
    valid_acc_y2 = r2["valid_acc"]
    test_acc_x2 = np.arange(EPOCH) + 1
    test_acc_y2 = r2["test_acc"]

    r3 = np.load("data_output/1.1.2.3.npz")
    train_loss_x3 = np.arange(train_epoch) + 1
    train_loss_y3 = r3["train_loss"]
    valid_loss_x3 = np.arange(valid_epoch) + 1
    valid_loss_y3 = r3["valid_loss"]
    test_loss_x3 = np.arange(EPOCH) + 1
    test_loss_y3 = r3["test_loss"]
    train_acc_x3 = np.arange(train_epoch) + 1
    train_acc_y3 = r3["train_acc"]
    valid_acc_x3 = np.arange(valid_epoch) + 1
    valid_acc_y3 = r3["valid_acc"]
    test_acc_x3 = np.arange(EPOCH) + 1
    test_acc_y3 = r3["test_acc"]

    plt.figure(1)
    plt.clf()
    plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, learning rate = 0.001")
    plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, learning rate = 0.003")
    plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, learning rate = 0.01")
    plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, learning rate = 0.001")
    plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, learning rate = 0.003")
    plt.semilogx(valid_loss_x3, valid_loss_y3, label = "valid, learning rate = 0.01")
    plt.legend()

    plt.title("Training and Validation Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("picture/1.1.2/1.1.2_tv_loss.png")

    plt.figure(2)
    plt.clf()
    plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, learning rate = 0.001")
    plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, learning rate = 0.003")
    plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, learning rate = 0.01")
    plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, learning rate = 0.001")
    plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, learning rate = 0.003")
    plt.semilogx(valid_acc_x3, valid_acc_y3, label = "valid, learning rate = 0.01")
    plt.legend()

    plt.title("Training and Validation Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("picture/1.1.2/1.1.2_tv_acc.png")

    plt.figure(3)
    plt.clf()
    plt.semilogx(test_loss_x1, test_loss_y1, label = "test, learning rate = 0.001")
    plt.semilogx(test_loss_x2, test_loss_y2, label = "test, learning rate = 0.003")
    plt.semilogx(test_loss_x3, test_loss_y3, label = "test, learning rate = 0.01")
    plt.legend()

    plt.title("Test Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("picture/1.1.2/1.1.2_te_loss.png")

    plt.figure(4)
    plt.clf()
    plt.semilogx(test_acc_x1, test_acc_y1, label = "test, learning rate = 0.001")
    plt.semilogx(test_acc_x2, test_acc_y2, label = "test, learning rate = 0.003")
    plt.semilogx(test_acc_x3, test_acc_y3, label = "test, learning rate = 0.01")
    plt.legend()

    plt.title("Test Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("picture/1.1.2/1.1.2_te_acc.png")


#****************************************************************************
# 1.2.1
#****************************************************************************
    r1 = np.load("data_output/1.1.2.3.npz")
    train_loss_x1 = np.arange(train_epoch) + 1
    train_loss_y1 = r1["train_loss"]
    valid_loss_x1 = np.arange(valid_epoch) + 1
    valid_loss_y1 = r1["valid_loss"]
    test_loss_x1 = np.arange(EPOCH) + 1
    test_loss_y1 = r1["test_loss"]
    train_acc_x1 = np.arange(train_epoch) + 1
    train_acc_y1 = r1["train_acc"]
    valid_acc_x1 = np.arange(valid_epoch) + 1
    valid_acc_y1 = r1["valid_acc"]
    test_acc_x1 = np.arange(EPOCH) + 1
    test_acc_y1 = r1["test_acc"]

    r2 = np.load("data_output/1.2.1.2.npz")
    train_loss_x2 = np.arange(train_epoch) + 1
    train_loss_y2 = r2["train_loss"]
    valid_loss_x2 = np.arange(valid_epoch) + 1
    valid_loss_y2 = r2["valid_loss"]
    test_loss_x2 = np.arange(EPOCH) + 1
    test_loss_y2 = r2["test_loss"]
    train_acc_x2 = np.arange(train_epoch) + 1
    train_acc_y2 = r2["train_acc"]
    valid_acc_x2 = np.arange(valid_epoch) + 1
    valid_acc_y2 = r2["valid_acc"]
    test_acc_x2 = np.arange(EPOCH) + 1
    test_acc_y2 = r2["test_acc"]

    r3 = np.load("data_output/1.2.1.3.npz")
    train_loss_x3 = np.arange(train_epoch) + 1
    train_loss_y3 = r3["train_loss"]
    valid_loss_x3 = np.arange(valid_epoch) + 1
    valid_loss_y3 = r3["valid_loss"]
    test_loss_x3 = np.arange(EPOCH) + 1
    test_loss_y3 = r3["test_loss"]
    train_acc_x3 = np.arange(train_epoch) + 1
    train_acc_y3 = r3["train_acc"]
    valid_acc_x3 = np.arange(valid_epoch) + 1
    valid_acc_y3 = r3["valid_acc"]
    test_acc_x3 = np.arange(EPOCH) + 1
    test_acc_y3 = r3["test_acc"]

    plt.figure(1)
    plt.clf()
    plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, 1000 hidden units")
    plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, 500 hidden units")
    plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, 100 hidden units")
    plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, 1000 hidden units")
    plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, 500 hidden units")
    plt.semilogx(valid_loss_x3, valid_loss_y3, label = "valid, 100 hidden units")
    plt.legend()

    plt.title("Training and Validation Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("picture/1.2.1/1.2.1_tv_loss.png")

    plt.figure(2)
    plt.clf()
    plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, 1000 hidden units")
    plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, 500 hidden units")
    plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, 100 hidden units")
    plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, 1000 hidden units")
    plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, 500 hidden units")
    plt.semilogx(valid_acc_x3, valid_acc_y3, label = "valid, 100 hidden units")
    plt.legend()

    plt.title("Training and Validation Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("picture/1.2.1/1.2.1_tv_acc.png")



#****************************************************************************
# 1.2.2
#****************************************************************************
    r1 = np.load("data_output/1.1.2.3.npz")
    train_loss_x1 = np.arange(train_epoch) + 1
    train_loss_y1 = r1["train_loss"]
    valid_loss_x1 = np.arange(valid_epoch) + 1
    valid_loss_y1 = r1["valid_loss"]
    test_loss_x1 = np.arange(EPOCH) + 1
    test_loss_y1 = r1["test_loss"]
    train_acc_x1 = np.arange(train_epoch) + 1
    train_acc_y1 = r1["train_acc"]
    valid_acc_x1 = np.arange(valid_epoch) + 1
    valid_acc_y1 = r1["valid_acc"]
    test_acc_x1 = np.arange(EPOCH) + 1
    test_acc_y1 = r1["test_acc"]

    r2 = np.load("data_output/1.2.2.1.npz")
    train_loss_x2 = np.arange(train_epoch) + 1
    train_loss_y2 = r2["train_loss"]
    valid_loss_x2 = np.arange(valid_epoch) + 1
    valid_loss_y2 = r2["valid_loss"]
    test_loss_x2 = np.arange(EPOCH) + 1
    test_loss_y2 = r2["test_loss"]
    train_acc_x2 = np.arange(train_epoch) + 1
    train_acc_y2 = r2["train_acc"]
    valid_acc_x2 = np.arange(valid_epoch) + 1
    valid_acc_y2 = r2["valid_acc"]
    test_acc_x2 = np.arange(EPOCH) + 1
    test_acc_y2 = r2["test_acc"]


    plt.figure(1)
    plt.clf()
    plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, 1 layer")
    plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, 2 layers")
    plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, 1 layer")
    plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, 2 layers")
    plt.legend()

    plt.title("Training and Validation Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("picture/1.2.2/1.2.2_tv_loss.png")

    plt.figure(2)
    plt.clf()
    plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, 1 layer")
    plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, 2 layers")
    plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, 1 layer")
    plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, 2 layers")
    plt.legend()

    plt.title("Training and Validation Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("picture/1.2.2/1.2.2_tv_acc.png")



#****************************************************************************
# 1.3.1
#****************************************************************************
    r1 = np.load("data_output/1.1.2.3.npz")
    train_loss_x1 = np.arange(train_epoch) + 1
    train_loss_y1 = r1["train_loss"]
    valid_loss_x1 = np.arange(valid_epoch) + 1
    valid_loss_y1 = r1["valid_loss"]
    test_loss_x1 = np.arange(EPOCH) + 1
    test_loss_y1 = r1["test_loss"]
    train_acc_x1 = np.arange(train_epoch) + 1
    train_acc_y1 = r1["train_acc"]
    valid_acc_x1 = np.arange(valid_epoch) + 1
    valid_acc_y1 = r1["valid_acc"]
    test_acc_x1 = np.arange(EPOCH) + 1
    test_acc_y1 = r1["test_acc"]

    r2 = np.load("data_output/1.3.1.1.npz")
    train_loss_x2 = np.arange(train_epoch) + 1
    train_loss_y2 = r2["train_loss"]
    valid_loss_x2 = np.arange(valid_epoch) + 1
    valid_loss_y2 = r2["valid_loss"]
    test_loss_x2 = np.arange(EPOCH) + 1
    test_loss_y2 = r2["test_loss"]
    train_acc_x2 = np.arange(train_epoch) + 1
    train_acc_y2 = r2["train_acc"]
    valid_acc_x2 = np.arange(valid_epoch) + 1
    valid_acc_y2 = r2["valid_acc"]
    test_acc_x2 = np.arange(EPOCH) + 1
    test_acc_y2 = r2["test_acc"]


    plt.figure(1)
    plt.clf()
    plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, dropout_rate = 0")
    plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, dropout_rate = 0.5")
    plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, dropout_rate = 0")
    plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, dropout_rate = 0.5")
    plt.legend()

    plt.title("Training and Validation Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("picture/1.3.1/1.3.1_tv_loss.png")

    plt.figure(2)
    plt.clf()
    plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, dropout_rate = 0")
    plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, dropout_rate = 0.5")
    plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, dropout_rate = 0")
    plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, dropout_rate = 0.5")
    plt.legend()

    plt.title("Training and Validation Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("picture/1.3.1/1.3.1_tv_acc.png")


if __name__ == '__main__':
	main()
