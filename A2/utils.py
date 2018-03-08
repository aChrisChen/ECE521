import json
from bunch import Bunch
import os
import argparse

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
    config.summary_dir = os.path.join(".../experiments", config.exp_name, "summary")
    config.checkpoint_dir = os.path.join(".../experiments", config.exp_name, "checkpoint")
    return config
