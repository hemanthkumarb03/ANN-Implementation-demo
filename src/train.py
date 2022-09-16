import sys
from src.utils.common import read_config
from src.utils.data import get_data
import argparse

from src.utils.model import create_model

def training(config_path):

    #read config file
    config = read_config(config_path)

    #get data
    val = config['params']['validation_datasize']
    (x_train,y_train),(x_val,y_val),(x_test,y_test) = get_data(val)
    
    #modelcreation
    loss = config['params']['loss_function']
    optimizer = config['params']['optimizer']
    metrics = config['params']['metrics']
    classes = config['params']['classes']
    model = create_model(loss, optimizer, metrics, classes)

    #model training
    epochs = config['params']['epochs']
    history = model.fit(x_train,y_train,epochs=epochs,validation_data=(x_val,y_val))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',"-c",default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)


