import sys
from src.utils.common import read_config
from src.utils.data import get_data
from src.utils.model import create_model
from src.utils.model import save_model
import os
import argparse



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

    #save model
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_name = config['artifacts']['model']
    model_dir = config['artifacts']['model_dir']
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model,model_name,model_dir)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',"-c",default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)


