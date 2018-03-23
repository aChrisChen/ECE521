import tensorflow as tf

from model import *
from train import *
from utils import *
from logger import *
from data_loader import *
from plot import *

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

    # select device
    if config.gpu:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    # sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    data = DataLoader(sess, config)
    print(data)
    # if config.logistic:
    #     model = LogisticRegression(config)
    # else:
    #     model = LinearRegression(config)

    model = MLP(config)
    #model = LogisticRegression(config)
    logger = Logger(sess, config)

    if config.mode == "Train":
        with tf.device(device_name):
            train(sess, data, model, config, logger)
    elif config.mode == "Inference":
        name_checkpoint = 'example-375'
        model.load(sess, file_name=name_checkpoint)
        test_loss, test_acc = evaluate(sess, data, model, 'test', config)
        print("Test Loss: %f, Test Accuracy: %f" % (test_loss, test_acc))

        visualize_layer1(sess, config.idx_unit_hidden1_visualization,name_checkpoint)



if __name__ == '__main__':
    main()
