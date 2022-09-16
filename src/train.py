import sys
from src.utils.common import read_config
from src.utils.data import get_data
import argparse

def training(config_path):
    config = read_config(config_path)
    val = config['params']['validation_datasize']
    (x_train,y_train),(x_val,y_val),(x_test,y_test) = get_data(val)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',"-c",default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)


