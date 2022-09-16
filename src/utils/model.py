import tensorflow as tf
import time
import os

def create_model(loss, optimizer, metrics, classes):

    layers = [
        tf.keras.layers.Flatten(input_shape=[28,28],name='input_layer'),
        tf.keras.layers.Dense(300,activation='relu',name='hidden_layer1'),
        tf.keras.layers.Dense(100,activation='relu',name='hidden_layer2'),
        tf.keras.layers.Dense(classes,activation='softmax',name='output_layer')
    ]
    model = tf.keras.models.Sequential(layers)
    model.summary()

    #compile model
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    return model ##<<< Untrained Model

def get_unique_filename(filename):
    filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return filename


def save_model(model,model_name,model_dir):
    filename = get_unique_filename(model_name)
    path = os.path.join(model_dir,filename)
    model.save(path)
