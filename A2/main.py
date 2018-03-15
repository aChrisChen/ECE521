import tensorflow as tf

#from data_loader.data_generator import DataGenerator
from model import*
from train import*
from utils import*
from logger import*
from data_generator import *
def main():
    # capture the config
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.plot_dir])

    if config.gpu:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    training_data = DataGenerator(sess, config)
    print(training_data)
    if config.logistic:
        model = LogisticClassification(config)
    else:
        model = LinearRegression(config)
    logger = Logger(sess, config)

    with tf.device(device_name):
        train(sess, training_data, model, config, logger)
    '''
    logger.plot_npz(fig_name="loss", \
                    # dict1="plot/notMNIST_2/lr=0.005000_wd=0.000000_adam=False_log.npz", \
                    # tag1="train_loss", tag1_label="OOD implementation train loss", \
                    dict1="plot/notMNIST_2/5000_0.005000_0.000000_False_False(MSE).npz", \
                    tag1="train_loss", tag1_label="Linear train loss", \
                    dict2="plot/notMNIST_2/5000_0.005000_0.000000_False_False(CrE).npz", \
                    tag2="train_loss", tag2_label="Sigmoid train loss", \
                    x_scale=1, new_plot=True, log_x=True, x_label="Number of iterations", y_label="Loss", title="Loss versus number of iterations")
    logger.plot_npz(fig_name="loss", \
                    dict1="plot/notMNIST_2/5000_0.005000_0.000000_False_False(MSE).npz", \
                    tag1="valid_loss", tag1_label="Linear valid loss", \
                    dict2="plot/notMNIST_2/5000_0.005000_0.000000_False_False(CrE).npz", \
                    tag2="valid_loss", tag2_label="Sigmoid valid loss", \
                    x_scale=35, new_plot=False, log_x=True, x_label="Number of iterations", y_label="Loss", title="Loss versus number of iterations")
    logger.plot_npz(fig_name="acc", \
                    dict1="plot/notMNIST_2/5000_0.005000_0.000000_False_False(MSE).npz", \
                    tag1="train_acc", tag1_label="Linear train accuracy", \
                    dict2="plot/notMNIST_2/5000_0.005000_0.000000_False_False(CrE).npz", \
                    tag2="train_acc", tag2_label="Sigmoid train accuracy", \
                    x_scale=1, new_plot=True, log_x=True, x_label="Number of iterations", y_label="Accuracy", title="Accuracy versus number of iterations")
    logger.plot_npz(fig_name="acc", \
                    dict1="plot/notMNIST_2/5000_0.005000_0.000000_False_False(MSE).npz", \
                    tag1="valid_acc", tag1_label="Linear valid accuracy", \
                    dict2="plot/notMNIST_2/5000_0.005000_0.000000_False_False(CrE).npz", \
                    tag2="valid_acc", tag2_label="Sigmoid valid accuracy", \
                    x_scale=35, new_plot=False, log_x=True, x_label="Number of iterations", y_label="Accuracy", title="Accuracy versus number of iterations")
    '''

if __name__ == '__main__':
    main()