import json
from bunch import Bunch
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description = __doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default= None,
        help= 'The Configuration file'
    )
    args = argparser.parse_args()
    return args


def create_dirs(dirs):
    try:
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_config_from_jason(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config


def process_config(json_file):
    config = get_config_from_jason(json_file)
    config.summary_dir = os.path.join("experiments", config.exp_name, "summary")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoint")
    config.plot_dir = os.path.join("experiments", config.exp_name, "plot")
    return config

def visualize_layer1(sess, idx_unit, checkpoint_name):
    var_hidden1 = [v for v in tf.trainable_variables() if v.name == "hidden1/kernel:0"][0]
    weight_hidden1= sess.run(var_hidden1)
    weight_selected = weight_hidden1[:,idx_unit-1]
    weight_selected = weight_selected.reshape([28, 28])
    plt.imshow(weight_selected)
    #plt.show()
    plt.savefig('visulization_unit'+ str(idx_unit)+ '-'+ checkpoint_name +'.png')

